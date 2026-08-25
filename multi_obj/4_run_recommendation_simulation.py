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
  4. Simulates the recommendations for 'exhaustive' and 'nsga2' methods, k
     ranks per method (k=1 by default -- one recommendation per case). Each
     rank reads its own recommendations_{case_study}_{method}_top{rank}of{k}.csv
     (see 3_run_experiment.py's run_experiment_top_k).
  5. Saves the simulated log(s) as CSV files under
     case_studies/<case_study>/prosit_simulation_results/<method>/<rank>/

Usage:
    python 4_run_recommendation_simulation.py --case_study bpi17_before --n_sim 10
    python 4_run_recommendation_simulation.py --case_study bpi17_before --n_sim 10 --force_rediscover
    python 4_run_recommendation_simulation.py --case_study bpi17_before --n_sim 10 --k 5 --force_rediscover
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

PNML_OVERRIDES = {}


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
        help="Number of simulation runs to perform per method/rank (default: 10)",
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Number of diverse recommendation ranks to simulate per method (default: 1). "
             "Expects recommendations_{case_study}_{method}_top{rank}of{k}.csv files "
             "(rank 1..k) and saves results under prosit_simulation_results/<method>/<rank>/.",
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


def _find_xes_log(case_dir: Path, case_study: str) -> Path:
    """Locate the historical .xes event log for this case study.

    Tries the case-sensitive convention `log_<case_study>.xes` first, then
    falls back to the single `log_*.xes` file in case_dir -- filename casing
    isn't consistent across case studies (e.g. case_studies/BPI12/ actually
    contains log_bpi12.xes, lowercase, not log_BPI12.xes).

    Raises:
        FileNotFoundError: if no .xes file can be found unambiguously.
    """
    exact = case_dir / f"log_{case_study}.xes"
    if exact.exists():
        return exact

    candidates = sorted(case_dir.glob("log_*.xes"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No .xes event log found under {case_dir} (tried {exact.name} "
            f"and log_*.xes). Discovery needs the historical log to learn "
            f"simulation parameters from."
        )
    raise FileNotFoundError(
        f"Ambiguous .xes event log under {case_dir}: found {[c.name for c in candidates]}, "
        f"none named {exact.name}. Rename the intended one or pass a matching case_study."
    )


def setup_simulator(case_dir: Path, case_study: str, force_rediscover: bool) -> SimulatorEngine:
    """Load the Petri net for this case study and build a SimulatorEngine for it, loading its simulation parameters from cache when possible, discovering them from the .xes log otherwise.

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
        pnml_path = case_dir / "discovery_output" / f"{case_study}_best_petri_net.pnml"
        params_cache_path = case_dir / "discovery_output" / f"simulator_params_{case_study}.json"

    if not pnml_path.exists():
        raise FileNotFoundError(
            f"No Petri net found at {pnml_path}. This script only discovers "
            f"the *simulation parameters* (from the .xes log, cached to "
            f"{params_cache_path.name}) -- it does not mine the Petri net "
            f"structure itself. Generate the .pnml first (see discovery/)."
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
        log_path = _find_xes_log(case_dir, case_study)
        print(f"Loading event log: {log_path}")
        log = xes_importer.apply(str(log_path))

        train_data_path = case_dir / "train_data.csv"
        if not train_data_path.exists():
            raise FileNotFoundError(
                f"No train_data.csv found at {train_data_path} -- needed to "
                f"restrict parameter discovery to training cases only (to "
                f"avoid leaking test cases into the simulator's learned "
                f"parameters, since {log_path.name} contains train+test+more)."
            )
        train_case_ids = set(
            pd.read_csv(train_data_path, usecols=[CASE_ID_NAME])[CASE_ID_NAME].astype(str)
        )
        n_before = len(log)
        # Trace-level case id attribute in a parsed EventLog is "concept:name"
        # (not "case:concept:name" -- that "case:" prefix only appears after
        # flattening to a DataFrame).
        log = pm4py.filter_trace_attribute_values(
            log, "concept:name", train_case_ids, retain=True, case_id_key="concept:name"
        )
        print(f"Restricted event log to {len(log)}/{n_before} training cases (from {train_data_path.name}).")

        print("Discovering simulation parameters from event log (this can take a few minutes)...")
        params.discover_from_eventlog(log, max_depth_tree=0)
        params_cache_path.parent.mkdir(parents=True, exist_ok=True)
        params.to_json(str(params_cache_path))
        print(f"Cached simulation parameters to {params_cache_path}\n")

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


def run_simulation_batch(sim_engine: SimulatorEngine, log_input: pd.DataFrame, out_folder: Path, n_sim: int, label: str = "") -> None:
    """Run SimulatorEngine.apply() n_sim times against the same input log, saving each run's result and diagnostics. Shared by the baseline and the exhaustive/nsga2 methods.

    Args:
        sim_engine: the SimulatorEngine to run.
        log_input: the prev_log to pass to sim_engine.apply() on every run.
        out_folder: where to save sim_<i>.csv and its diagnostic CSVs.
        n_sim: number of simulation runs to perform.
        label: short prefix identifying which scenario is running (e.g.
            "BASELINE" or "EXHAUSTIVE rank 2/5"), printed with every run so
            it's obvious which rank a given log line belongs to. Optional.
    """
    prefix = f"[{label}] " if label else ""
    for i in range(n_sim):
        run_start = datetime.now()
        print(f"{prefix}Starting run {i + 1}/{n_sim} at {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
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

    run_simulation_batch(sim_engine, log_baseline, baseline_folder, n_sim, label="BASELINE")

    print("Baseline simulation finished successfully!\n")


def _simulate_recommendation_file(
    sim_engine: SimulatorEngine,
    csv_path: Path,
    sim_folder: Path,
    case_study: str,
    clean_prev_log: pd.DataFrame,
    case_ids: Optional[List[str]],
    n_sim: int,
    label: str = "",
) -> None:
    """Load one recommendations CSV, merge it with the test log, and run n_sim simulations, saving the results under sim_folder.

    Cases with no recommendation (missing Next_activity/Next_resource) are NOT
    filled in with what actually happened historically -- simulation exists
    precisely so we don't need to know that. Instead, they are excluded from
    this simulation run (the simulator would otherwise silently drop them
    from the output anyway -- it only simulates cases where
    recommendation:act/res are not both null, see prosit/simulator.py's
    `cases_prefixes` filtering in SimulatorEngine.apply()), and reported both
    on stdout and as excluded_no_recommendation.csv under sim_folder, so the
    exclusion is visible rather than silent. This means the recommender found
    no valid next action for them at all (empty possible-activities set --
    even after next_possible_activities' trace-suffix fallback -- or empty
    Pareto front); it's a genuine cold-start gap in the training data, not
    something simulating around it can fix.

    Args:
        sim_engine: the SimulatorEngine to run.
        csv_path: path to the recommendations CSV (case:concept:name, Next_activity, Next_resource).
        sim_folder: where to save sim_<i>.csv and its diagnostic CSVs.
        case_study: name of the case study.
        clean_prev_log: the case-specific-columns-only test log.
        case_ids: optional case id filter.
        n_sim: number of simulation runs to perform.
        label: short prefix identifying which scenario is running (e.g.
            "EXHAUSTIVE rank 2/5"), printed on every log line and passed
            through to run_simulation_batch. Optional.

    Raises:
        ValueError: if case_ids filtering leaves no matching rows.
    """
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Loading recommendations: {csv_path}")
    # Next_activity/Next_resource forced to str at read time: with missing values present (some
    # cases have no recommendation at this rank) and numeric-looking resource ids (e.g. BPI12's),
    # pandas would otherwise infer that column as float64, silently mangling ids like "10629" into
    # "10629.0" -- which then doesn't match any key in the simulator's sampled_waiting_times dict
    # (built from the raw .xes log, where resource ids are plain strings) and crashes with a KeyError.
    rec_df = pd.read_csv(csv_path, dtype={CASE_ID_NAME: str, "Next_activity": str, "Next_resource": str})

    missing_mask = rec_df["Next_activity"].isna() | rec_df["Next_resource"].isna()
    if missing_mask.any():
        excluded_ids = rec_df.loc[missing_mask, CASE_ID_NAME].tolist()
        print(
            f"{prefix}WARNING: {len(excluded_ids)} case(s) have no recommendation "
            f"(missing Next_activity/Next_resource) in {csv_path} -- excluding "
            f"them from this simulation run instead of simulating them. "
            f"Affected case ids: {excluded_ids[:20]}" + (" ..." if len(excluded_ids) > 20 else "")
        )
        sim_folder.mkdir(parents=True, exist_ok=True)
        excluded_path = sim_folder / "excluded_no_recommendation.csv"
        pd.DataFrame({CASE_ID_NAME: excluded_ids}).to_csv(excluded_path, index=False)
        print(f"{prefix}Saved excluded case ids to {excluded_path}")
        rec_df = rec_df.loc[~missing_mask].reset_index(drop=True)

    current_prev_log = clean_prev_log.copy()
    if case_study.upper() == "BPI12":
        print(f"{prefix}Applying BPI12 specific data type conversions...")
        rec_df = convert_dtypes_bpi12(rec_df, "simulation_prep")
        current_prev_log = convert_dtypes_bpi12(current_prev_log, "simulation")

    # Keep case-id dtype consistent across logs/recommendations/filters.
    rec_df[CASE_ID_NAME] = rec_df[CASE_ID_NAME].astype(str)
    current_prev_log[CASE_ID_NAME] = current_prev_log[CASE_ID_NAME].astype(str)

    print(f"{prefix}Building recommendations dataframe...")
    recommendations = {
        row[CASE_ID_NAME]: {"act": row["Next_activity"], "res": row["Next_resource"]}
        for _, row in rec_df.iterrows()
    }
    log_rec = build_recommender_df(current_prev_log, recommendations)
    log_rec["start:timestamp"] = pd.to_datetime(log_rec["start:timestamp"], format="mixed")
    log_rec["time:timestamp"] = pd.to_datetime(log_rec["time:timestamp"], format="mixed")

    if case_ids:
        str_case_ids = [str(c) for c in case_ids]
        log_rec = log_rec[log_rec[CASE_ID_NAME].isin(str_case_ids)]
        if log_rec.empty:
            raise ValueError(
                "No matching rows after --case_ids filtering. "
                "Check case-id dtype/content and input file values."
            )

    sim_folder.mkdir(parents=True, exist_ok=True)
    run_simulation_batch(sim_engine, log_rec, sim_folder, n_sim, label=label)


def run_recommendation_simulations(
    sim_engine: SimulatorEngine,
    case_dir: Path,
    case_study: str,
    clean_prev_log: pd.DataFrame,
    case_ids: Optional[List[str]],
    n_sim: int,
    k: int = 1,
) -> None:
    """Run the 'exhaustive' and 'nsga2' recommendation methods (whichever have recommendation files available) n_sim times each per rank, saving the results under case_dir/prosit_simulation_results/<method>/<rank>/.

    k is the number of diverse recommendations per case produced upstream
    (see select_top_k_pareto_actions / compute_recommendations_top_k /
    run_experiment_top_k). For each method and each rank in 1..k, this loads
    `recommendations_{case_study}_{method}_top{rank}of{k}.csv` and saves
    results under prosit_simulation_results/<method>/<rank>/ -- one
    subfolder per rank, so runs don't overwrite each other.

    Each (method, rank) combination is independent: a missing file for one
    rank only skips that rank (with a warning), it doesn't stop the others.
    Cases with no recommendation at all within a loaded file are excluded
    from that simulation run (not filled in with historical fallback, not a
    hard failure) -- see _simulate_recommendation_file.

    Args:
        sim_engine: the SimulatorEngine to run.
        case_dir: case_studies/<case_study>/ directory.
        case_study: name of the case study.
        clean_prev_log: the case-specific-columns-only test log.
        case_ids: optional case id filter.
        n_sim: number of simulation runs to perform per method/rank.
        k: number of recommendation ranks to simulate per method. Defaults to 1.

    Raises:
        ValueError: if k < 1, or if case_ids filtering leaves no matching
            rows for a method/rank.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    for method in ["exhaustive", "nsga2"]:
        print(f"=== STARTING {method.upper()} SIMULATION (k={k}) ===")

        for rank in range(1, k + 1):
            csv_path = case_dir / "recommendations" / f"recommendations_{case_study}_{method}_top{rank}of{k}.csv"
            sim_folder = case_dir / "prosit_simulation_results" / method / str(rank)
            rank_label = f"{method.upper()} rank {rank}/{k}"

            print(f"--- {rank_label} ---")

            if not csv_path.exists():
                print(f"WARNING: Recommendation file {csv_path} not found. Skipping {rank_label}.")
                print("-" * 50)
                continue

            _simulate_recommendation_file(
                sim_engine, csv_path, sim_folder, case_study, clean_prev_log, case_ids, n_sim,
                label=rank_label,
            )

            print(f"{rank_label} simulation finished successfully!\n")


def main():
    args = parse_args()
    sys.path.append(args.base_dir)

    case_dir = Path(args.base_dir) / "case_studies" / args.case_study

    sim_engine = setup_simulator(case_dir, args.case_study, args.force_rediscover)
    clean_prev_log, case_ids = load_inputs(case_dir, args.case_study, args.case_ids)

    run_baseline_simulation(sim_engine, clean_prev_log, case_ids, case_dir, args.n_sim)
    run_recommendation_simulations(sim_engine, case_dir, args.case_study, clean_prev_log, case_ids, args.n_sim, args.k)


if __name__ == "__main__":
    main()
