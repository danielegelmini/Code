#!/usr/bin/env python3
"""
Given a case study, this script:
  1. Loads the Petri net model (.pnml) and the test log.
  2. Learns the simulation parameters ONCE -- or loads them from a cached
     JSON file (case_studies/<case_study>/discovery_output/simulator_params_<case_study>.json)
     if one already exists, skipping the expensive discovery step (and even
     the .xes log parsing) entirely. Use --force_rediscover to bypass the
     cache and regenerate it.
  3. Simulates the baseline scenario.
  4. Simulates the recommendations for 'exhaustive' and 'nsga2' methods.
  5. Saves the simulated log(s) as CSV files under
     case_studies/<case_study>/prosit_simulation_results/<method>/

Usage:
    python 4_run_recommendation_simulation.py --case_study bpi17_before --n_sim 10
    python 4_run_recommendation_simulation.py --case_study bpi17_before --n_sim 10 --force_rediscover
"""

import argparse
import sys
import uuid
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

from utils.simulation_functions import build_recommender_df
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.get_features import get_features

from prosit.simulator import SimulatorParameters, SimulatorEngine

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Event log column names (constants)
# ---------------------------------------------------------------------------
CASE_ID_NAME = "case:concept:name"
START_DATE_NAME = "start:timestamp"
END_DATE_NAME = "time:timestamp"
ACTIVITY_COLUMN_NAME = "concept:name"
RESOURCE_COLUMN_NAME = "org:resource"

# model_bpi12.pnml is missing O_SENT_BACK and O_CANCELLED as transitions entirely --
# 578/633 (91%) of the exhaustive/nsga2 recommendations for BPI12 target one of these
# two activities, so they were structurally unreachable no matter what (not a simulator
# bug: the label simply doesn't exist in that net). Overriding to an alternative net
# also requires a distinct params-cache filename, since cached simulator params are
# keyed to specific Transition objects of the net they were discovered against --
# reusing the default model's cache against a different net would silently mismatch.
PNML_OVERRIDES = {
    "BPI12": "diem_log_BPI12.pnml",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full simulation pipeline (Baseline, Exhaustive, NSGA2) for a given case study."
    )
    parser.add_argument(
        "--case_study", type=str, default="bpi12",
        help="Name of the case study folder under case_studies/ (default: bpi12)",
    )
    parser.add_argument(
        "--n_sim", type=int, default=10,
        help="Number of simulation runs to perform per method (default: 10)",
    )
    parser.add_argument(
        "--case_ids", type=str, default=None,
        help="Optional path to a text file with one case id per line to filter log_rec on. "
             "If omitted, all cases in the test log are simulated.",
    )
    parser.add_argument(
        "--base_dir", type=str, default=".",
        help="Base directory containing the case_studies/ folder.",
    )
    parser.add_argument(
        "--force_rediscover", action="store_true",
        help="Ignore any cached simulation parameters and re-run discovery from "
             "the event log, overwriting the cache.",
    )
    return parser.parse_args()


def setup_simulator(case_dir: Path, case_study: str, force_rediscover: bool) -> SimulatorEngine:
    """Load the Petri net for this case study and build a SimulatorEngine for it, loading its simulation parameters from cache when possible.

    discover_from_eventlog is the expensive step (resource discovery, feature building/alignment, transition weights, calendars, execution/waiting/arrival time discovery -- typically 3-5 minutes). Since it depends only on the log (not on n_sim, case_ids, or the method being simulated), the discovered parameters are cached to disk the first time and reused on every subsequent run; when the cache hits, the .xes log isn't even parsed.

    Args:
        case_dir: case_studies/<case_study>/ directory.
        case_study: name of the case study (e.g. "BPI12", "bac").
        force_rediscover: if True, ignore any existing parameters cache and regenerate it.

    Returns:
        A ready-to-use SimulatorEngine.
    """
    if case_study in PNML_OVERRIDES:
        pnml_filename = PNML_OVERRIDES[case_study]
        pnml_path = case_dir / "discovery_output" / pnml_filename
        params_cache_path = case_dir / "discovery_output" / f"simulator_params_{case_study}_{Path(pnml_filename).stem}.json"
    else:
        pnml_path = case_dir / "discovery_output" / f"model_{case_study}.pnml"
        params_cache_path = case_dir / "discovery_output" / f"simulator_params_{case_study}.json"

    if not pnml_path.exists():
        raise FileNotFoundError(
            f"No Petri net found at {pnml_path}. Discovery from scratch is "
            f"disabled in this script (see commented-out block) -- generate "
            f"the .pnml first."
        )
    print(f"Loading Petri net: {pnml_path}")
    net, im, fm = pm4py.read_pnml(str(pnml_path))

    params = SimulatorParameters(net, im, fm)
    cache_valid = False
    if params_cache_path.exists() and not force_rediscover:
        print(f"Loading cached simulation parameters: {params_cache_path}")
        try:
            params.from_json(str(params_cache_path))
            cache_valid = True
            print("Simulation parameters loaded from cache successfully.\n")
        except ValueError as exc:
            print(f"WARNING: Cached simulator parameters are invalid: {exc}")
            print("Regenerating simulation parameters from the event log...\n")
            params_cache_path.unlink(missing_ok=True)

    if not cache_valid:
        print("No valid cached simulation parameters found.")


    return SimulatorEngine(params)


def load_inputs(case_dir: Path, case_study: str, case_ids_path: Optional[str]) -> tuple[pd.DataFrame, Optional[List[str]]]:
    """Load the test log, reduced to the columns needed for simulation, and the optional case-id filter.

    The test log is restricted to the base event-log columns plus whatever case-specific (continuous/categorical) feature columns get_features() declares for this case study -- everything else is dropped.

    Args:
        case_dir: case_studies/<case_study>/ directory.
        case_study: name of the case study, used to look up its feature list.
        case_ids_path: optional path to a text file with one case id per line.

    Returns:
        (clean_prev_log, case_ids): the reduced test log, and the case-id
        filter list (None if case_ids_path was not provided).
    """
    test_log_path = case_dir / "test_log.csv"
    print(f"Loading test log (running traces): {test_log_path}")
    prev_log = pd.read_csv(test_log_path, parse_dates=[END_DATE_NAME, START_DATE_NAME])

    base_columns = [
        CASE_ID_NAME, ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME,
        START_DATE_NAME, END_DATE_NAME, "NEXT_ACTIVITY", "NEXT_RESOURCE",
    ]
    _, _, _, continuous_features, categorical_features, _ = get_features(case_study)
    engineered_continuous_exact = {
        "time_from_start", "time_from_previous_event(start)", "event_duration",
    }
    engineered_continuous_prefix = "# ACTIVITY="
    base_categorical_already_included = {
        ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME,
        "NEXT_ACTIVITY", "NEXT_RESOURCE", "weekday",
    }
    raw_continuous = [
        c for c in continuous_features
        if c not in engineered_continuous_exact
        and not c.startswith(engineered_continuous_prefix)
    ]
    raw_categorical = [
        c for c in categorical_features
        if c not in base_categorical_already_included
    ]
    case_specific_columns = [c for c in (raw_continuous + raw_categorical) if c in prev_log.columns]
    missing = [c for c in (raw_continuous + raw_categorical) if c not in prev_log.columns]
    if missing:
        print(
            f"The following case-specific columns from get_features('{case_study}') "
            f"were NOT found in test_log.csv: {missing}"
        )
    selected_columns = base_columns + [c for c in case_specific_columns if c not in base_columns]
    clean_prev_log = prev_log[selected_columns]

    case_ids = None
    if case_ids_path:
        with open(case_ids_path) as f:
            case_ids = [line.strip() for line in f if line.strip()]
        print(f"Restricting test log to {len(case_ids)} case ids from {case_ids_path}")

    return clean_prev_log, case_ids


def save_engine_diagnostics(sim_engine, sim_folder: Path, run_index: int) -> None:
    """Persist the SimulatorEngine's post-run diagnostics as CSVs next to sim_<run_index>.csv
    (when non-empty), and print a terse case-count summary.

    - last_unreachable_recommendations: a requested recommendation could not legally be reached from the replayed prefix marking (exhaustive/nsga2 runs only). This should be 0 for baseline, since baseline never resolves a recommendation.
    - last_runaway_cases: a case fired >= max_events_per_case simulated events without reaching the final marking and was force-truncated by SimulatorEngine.apply()'s safety valve. Should be 0: every prefix marking is now reconstructed via alignment, which is always legally reachable, and the net is a sound workflow net -- so every case is structurally guaranteed to be able to complete.
    - last_non_fitting_prefixes: the historical prefix for a case needed at least one "log move" during alignment (a logged activity the model could not explain at all). Saved for reference only, not printed -- it does not affect completability.
    - last_model_inserted_activities: the historical prefix for a case needed at least one visible activity inserted by the model that is not actually present in the log (the model considers it a necessary step the log just didn't record). Never written to the output log; does affect that case's activity-history counts going forward.

    Args:
        sim_engine: the SimulatorEngine instance that just completed a run.
        sim_folder: output folder for this method's simulation results.
        run_index: 1-based index of the run just completed.
    """
    diagnostics = [
        ("last_unreachable_recommendations", "unreachable_recommendations"),
        ("last_runaway_cases", "runaway_cases"),
        ("last_non_fitting_prefixes", "non_fitting_prefixes"),
        ("last_model_inserted_activities", "model_inserted_activities"),
    ]
    for attr_name, file_suffix in diagnostics:
        records = getattr(sim_engine, attr_name, [])
        if records:
            out_path = sim_folder / f"sim_{run_index}_{file_suffix}.csv"
            pd.DataFrame(records).to_csv(out_path, index=False)

    print(f"Unreachable recommendations: {len(sim_engine.last_unreachable_recommendations)}")
    print(f"Interrupted cases: {len(sim_engine.last_runaway_cases)}")
    print(f"Cases with model-inserted activities: {len(sim_engine.last_model_inserted_activities)}")


def run_simulation_batch(sim_engine: SimulatorEngine, log_input: pd.DataFrame, out_folder: Path, n_sim: int) -> None:
    """Run SimulatorEngine.apply() n_sim times against the same input log, saving each run's result and diagnostics. Shared by the baseline and the exhaustive/nsga2 methods.

    Args:
        sim_engine: the SimulatorEngine to run.
        log_input: the prev_log to pass to sim_engine.apply() on every run.
        out_folder: where to save sim_<i>.csv and its diagnostic CSVs.
        n_sim: number of simulation runs to perform.
    """
    for i in range(n_sim):
        run_start = datetime.now()
        print(f"Starting run {i + 1}/{n_sim} at {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
        sim_log = sim_engine.apply(prev_log=log_input)
        sim_log = sim_log.sort_values(by=["case:concept:name", "time:timestamp"])
        out_path = out_folder / f"sim_{i + 1}.csv"
        sim_log.to_csv(out_path, index=False)
        save_engine_diagnostics(sim_engine, out_folder, i + 1)


def run_baseline_simulation(sim_engine: SimulatorEngine, clean_prev_log: pd.DataFrame, case_ids: Optional[List[str]], case_dir: Path, n_sim: int) -> None:
    """Run the baseline scenario (no recommendations applied) n_sim times and save the results under case_dir/prosit_simulation_results/baseline/.

    Every case is given a sentinel "no recommendation" value, so the simulator never resolves an actual recommendation and simply continues each case probabilistically.

    Args:
        sim_engine: the SimulatorEngine to run.
        clean_prev_log: the case-specific-columns-only test log.
        case_ids: optional case id filter.
        case_dir: case_studies/<case_study>/ directory.
        n_sim: number of simulation runs to perform.
    """
    print("=== STARTING BASELINE SIMULATION ===")
    baseline_folder = case_dir / "prosit_simulation_results" / "baseline"
    baseline_folder.mkdir(parents=True, exist_ok=True)

    sentinel_act = f"__NO_RECOMMENDATION__{uuid.uuid4().hex}"
    baseline_recommendations = {
        case_id: {"act": sentinel_act, "res": None}
        for case_id in clean_prev_log["case:concept:name"].unique()
    }
    log_baseline = build_recommender_df(clean_prev_log.copy(), baseline_recommendations)
    log_baseline["start:timestamp"] = pd.to_datetime(log_baseline["start:timestamp"], format="mixed")
    log_baseline["time:timestamp"] = pd.to_datetime(log_baseline["time:timestamp"], format="mixed")
    if case_ids:
        log_baseline = log_baseline[log_baseline["case:concept:name"].isin(case_ids)]

    run_simulation_batch(sim_engine, log_baseline, baseline_folder, n_sim)

    print("Baseline simulation finished successfully!\n")


def run_recommendation_simulations(
    sim_engine: SimulatorEngine, case_dir: Path, case_study: str, clean_prev_log: pd.DataFrame, case_ids: Optional[List[str]], n_sim: int
) -> None:
    """Run the 'exhaustive' and 'nsga2' recommendation methods (whichever have a recommendations file available) n_sim times each, saving the results under case_dir/prosit_simulation_results/<method>/.

    For each method: loads its recommendations pickle, merges it with the test log (applying BPI12-specific dtype conversions where needed, and filling any missing recommendation with the case's actual historical next activity/resource), then runs the simulation batch.

    Args:
        sim_engine: the SimulatorEngine to run.
        case_dir: case_studies/<case_study>/ directory.
        case_study: name of the case study.
        clean_prev_log: the case-specific-columns-only test log.
        case_ids: optional case id filter.
        n_sim: number of simulation runs to perform per method.

    Raises:
        ValueError: if case_ids filtering leaves no matching rows for a method.
    """
    for method in ["exhaustive", "nsga2"]:
        print(f"=== STARTING {method.upper()} SIMULATION ===")

        res_path_base = case_dir / "recommendations" / f"recommendations_{case_study}_{method}"
        pkl_path = Path(f"{res_path_base}.pkl")
        if not pkl_path.exists():
            print(f"WARNING: Recommendation file {pkl_path} not found. Skipping {method}.")
            print("-" * 50)
            continue

        sim_folder = case_dir / "prosit_simulation_results" / method
        sim_folder.mkdir(parents=True, exist_ok=True)

        print(f"Loading recommendations: {pkl_path}")
        res = pd.read_pickle(pkl_path)
        rec_df = pd.DataFrame.from_dict(
            res, orient="index", columns=["Next_activity", "Next_resource"]
        ).reset_index()
        rec_df.rename(columns={"index": CASE_ID_NAME}, inplace=True)
        rec_df.to_csv(f"{res_path_base}.csv", index=False)

        current_prev_log = clean_prev_log.copy()
        if case_study.upper() == "BPI12":
            print("Applying BPI12 specific data type conversions...")
            rec_df = convert_dtypes_bpi12(rec_df, "recommendation")
            current_prev_log = convert_dtypes_bpi12(current_prev_log, "simulation")

        # Keep case-id dtype consistent across logs/recommendations/filters.
        rec_df[CASE_ID_NAME] = rec_df[CASE_ID_NAME].astype(str)
        current_prev_log[CASE_ID_NAME] = current_prev_log[CASE_ID_NAME].astype(str)
        rec_df["Next_activity"] = rec_df["Next_activity"].fillna(current_prev_log["NEXT_ACTIVITY"])
        rec_df["Next_resource"] = rec_df["Next_resource"].fillna(current_prev_log["NEXT_RESOURCE"])
        rec_df.to_csv(f"{res_path_base}.csv", index=False)
        if rec_df.isna().sum().sum() > 0:
            print("Recommendations contain NaN values:")
            print(rec_df.isna().sum())

        print("Building recommendations dataframe...")
        recommendations = {
            row["case:concept:name"]: {"act": row["Next_activity"], "res": row["Next_resource"]}
            for _, row in rec_df.iterrows()
        }
        log_rec = build_recommender_df(current_prev_log, recommendations)
        log_rec["start:timestamp"] = pd.to_datetime(log_rec["start:timestamp"], format="mixed")
        log_rec["time:timestamp"] = pd.to_datetime(log_rec["time:timestamp"], format="mixed")

        if case_ids:
            str_case_ids = [str(c) for c in case_ids]
            log_rec = log_rec[log_rec["case:concept:name"].isin(str_case_ids)]
            if log_rec.empty:
                raise ValueError(
                    "No matching rows after --case_ids filtering. "
                    "Check case-id dtype/content and input file values."
                )

        run_simulation_batch(sim_engine, log_rec, sim_folder, n_sim)

        print(f"{method.capitalize()} simulation finished successfully!\n")


def main():
    args = parse_args()
    sys.path.append(args.base_dir)

    case_dir = Path(args.base_dir) / "case_studies" / args.case_study

    sim_engine = setup_simulator(case_dir, args.case_study, args.force_rediscover)
    clean_prev_log, case_ids = load_inputs(case_dir, args.case_study, args.case_ids)

    run_baseline_simulation(sim_engine, clean_prev_log, case_ids, case_dir, args.n_sim)
    run_recommendation_simulations(sim_engine, case_dir, args.case_study, clean_prev_log, case_ids, args.n_sim)


if __name__ == "__main__":
    main()
