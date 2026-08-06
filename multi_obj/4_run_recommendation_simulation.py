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
import json
import sys
import uuid
import warnings
from pathlib import Path
from datetime import datetime

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


def print_workload_estimate(
    sim_input_log: pd.DataFrame,
    run_label: str,
    n_sim: int,
    generate_new_cases_from_prev_log: bool,
) -> None:
    """Print an estimate of simulation scale to avoid mistaking long runs for hangs."""

    if sim_input_log is None or sim_input_log.empty:
        print(f"[{run_label}] Workload estimate unavailable: empty input log.")
        return

    log = sim_input_log.copy()
    log[START_DATE_NAME] = pd.to_datetime(log[START_DATE_NAME], format="mixed", errors="coerce")
    log[END_DATE_NAME] = pd.to_datetime(log[END_DATE_NAME], format="mixed", errors="coerce")
    log = log.dropna(subset=[START_DATE_NAME, END_DATE_NAME])

    if log.empty:
        print(f"[{run_label}] Workload estimate unavailable: invalid timestamps after parsing.")
        return

    trace_durations = log.groupby(CASE_ID_NAME).agg(
        trace_start=(START_DATE_NAME, "min"),
        trace_end=(END_DATE_NAME, "max"),
    )
    trace_durations["duration"] = trace_durations["trace_end"] - trace_durations["trace_start"]

    longest_case = trace_durations["duration"].idxmax()
    longest_trace = trace_durations.loc[longest_case]
    started_during_longest = trace_durations[
        (trace_durations["trace_start"] >= longest_trace["trace_start"])
        & (trace_durations["trace_start"] <= longest_trace["trace_end"])
        & (trace_durations.index != longest_case)
    ]
    if generate_new_cases_from_prev_log:
        n_traces = int(started_during_longest.shape[0])
    else:
        n_traces = 0

    if "recommendation:act" in log.columns or "recommendation:res" in log.columns:
        rec_act = log["recommendation:act"] if "recommendation:act" in log.columns else pd.Series([None] * len(log))
        rec_res = log["recommendation:res"] if "recommendation:res" in log.columns else pd.Series([None] * len(log))
        rec_cases = log[~rec_act.isna() | ~rec_res.isna()][CASE_ID_NAME].nunique()
    else:
        rec_cases = 0

    est_cases_per_run = rec_cases + n_traces
    print(
        f"[{run_label}] Workload estimate: input_cases={trace_durations.shape[0]}, "
        f"prefix_cases={rec_cases}, sampled_new_cases={n_traces}, "
        f"~cases_processed_per_run={est_cases_per_run}, runs={n_sim}."
    )
    if est_cases_per_run >= 1000:
        print(
            f"[{run_label}] Note: high workload detected. Progress may look slow for long periods; "
            "prefer waiting unless there is truly no CPU activity for a prolonged time."
        )


def _format_seconds_to_hms(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_run_timing(
    run_label: str,
    run_index: int,
    n_runs: int,
    run_start: datetime,
    run_end: datetime,
    durations_seconds: list[float],
) -> None:
    """Print timing for one run and ETA for remaining runs in the same batch."""

    run_duration = max(0.0, (run_end - run_start).total_seconds())
    durations_seconds.append(run_duration)
    avg_duration = sum(durations_seconds) / len(durations_seconds)
    runs_left = n_runs - run_index
    eta_seconds = avg_duration * runs_left

    print(
        f"[{run_label}] Run {run_index}/{n_runs} timing: "
        f"duration={_format_seconds_to_hms(run_duration)}, "
        f"avg={_format_seconds_to_hms(avg_duration)}, "
        f"remaining_runs={runs_left}, "
        f"estimated_time_left={_format_seconds_to_hms(eta_seconds)}."
    )


def main():
    args = parse_args()
    sys.path.append(args.base_dir)

    base_dir = Path(args.base_dir)
    case_study = args.case_study
    case_dir = base_dir / "case_studies" / case_study

    # -----------------------------------------------------------------
    # 1. Load event log, Petri net model, and test log

    # In this step, the script loads all the required process mining artifacts.
    # It reads the event log in XES format and the running traces from a CSV test log.
    # It also attempts to load an existing Petri net (PNML) model; .
    # Finally, it filters the test log to retain only the necessary features and target columns.
    # -----------------------------------------------------------------
    log_path = case_dir / f"log_{case_study}.xes"
    pnml_path = case_dir / "discovery_output" / f"model_{case_study}.pnml"
    test_log_path = case_dir / "test_log.csv"
    params_cache_path = case_dir / "discovery_output" / f"simulator_params_{case_study}.json"

    if pnml_path.exists():
        print(f"Loading Petri net: {pnml_path}")
        net, im, fm = pm4py.read_pnml(str(pnml_path))
    else:
        raise FileNotFoundError(
            f"No Petri net found at {pnml_path}. Discovery from scratch is "
            f"disabled in this script (see commented-out block) -- generate "
            f"the .pnml first."
        )

    print(f"Loading test log (running traces): {test_log_path}")
    prev_log = pd.read_csv(
        test_log_path, parse_dates=[END_DATE_NAME, START_DATE_NAME]
    )

    # -----------------------------------------------------------------
    # 2. Learn (or load cached) Simulation Parameters ONCE

    # discover_from_eventlog is the expensive step (resource discovery,
    # feature building/alignment, transition weights, calendars, execution/
    # waiting/arrival time discovery -- typically 3-5 minutes). Since it
    # depends only on the log (not on n_sim, case_ids, or the method being
    # simulated), we cache the discovered parameters to disk the first time
    # and reuse them on every subsequent run, unless --force_rediscover is
    # passed or no cache exists yet. When the cache hits, we don't even need
    # to parse the .xes log file again.
    # -----------------------------------------------------------------
    params = SimulatorParameters(net, im, fm)

    cache_valid = False
    if params_cache_path.exists() and not args.force_rediscover:
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
        print(f"Loading event log: {log_path}")
        log = xes_importer.apply(str(log_path))
        print("Discovering simulation parameters from event log (this can take a few minutes)...")
        params.discover_from_eventlog(log, max_depth_tree=0)
        params_cache_path.parent.mkdir(parents=True, exist_ok=True)
        params.to_json(str(params_cache_path))
        print(f"Cached simulation parameters to {params_cache_path}\n")

    sim_engine = SimulatorEngine(params)

    # Cleaning the test log to only include the columns we need for simulation
    base_columns = [
        CASE_ID_NAME, ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME,
        START_DATE_NAME, END_DATE_NAME, "NEXT_ACTIVITY", "NEXT_RESOURCE",
    ]
    
    _, _, _, continuous_features, categorical_features, _ = get_features(case_study)

    ENGINEERED_CONTINUOUS_EXACT = {
        "time_from_start", "time_from_previous_event(start)", "event_duration",
    }
    ENGINEERED_CONTINUOUS_PREFIX = "# ACTIVITY="
    BASE_CATEGORICAL_ALREADY_INCLUDED = {
        ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME,
        "NEXT_ACTIVITY", "NEXT_RESOURCE", "weekday",
    }
 
    raw_continuous = [
        c for c in continuous_features
        if c not in ENGINEERED_CONTINUOUS_EXACT
        and not c.startswith(ENGINEERED_CONTINUOUS_PREFIX)
    ]
    raw_categorical = [
        c for c in categorical_features
        if c not in BASE_CATEGORICAL_ALREADY_INCLUDED
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

    # Optional filtering to a specific subset of case ids
    case_ids = None
    if args.case_ids:
        with open(args.case_ids) as f:
            case_ids = [line.strip() for line in f if line.strip()]
        print(f"Restricting test log to {len(case_ids)} case ids from {args.case_ids}")

    # -----------------------------------------------------------------
    # 3. BASELINE SIMULATION

    # Executes the baseline scenario where no recommendations are applied.
    # It assigns a dummy/sentinel value to indicate the absence of an action,
    # builds the recommender dataframe, and runs the simulation engine 'n_sim' times.
    # The resulting simulated logs are sorted chronologically per case and exported as CSV files.
    # -----------------------------------------------------------------
    print("=== STARTING BASELINE SIMULATION ===")
    baseline_folder = case_dir / "prosit_simulation_results" / "baseline"
    baseline_folder.mkdir(parents=True, exist_ok=True)

    log_baseline_raw = clean_prev_log.copy()
    
    sentinel_act = f"__NO_RECOMMENDATION__{uuid.uuid4().hex}"
    baseline_recommendations = {
        case_id: {"act": sentinel_act, "res": None}
        for case_id in log_baseline_raw["case:concept:name"].unique()
    }
    
    log_baseline = build_recommender_df(log_baseline_raw, baseline_recommendations)
    log_baseline["start:timestamp"] = pd.to_datetime(log_baseline["start:timestamp"], format="mixed")
    log_baseline["time:timestamp"] = pd.to_datetime(log_baseline["time:timestamp"], format="mixed")

    if case_ids:
        log_baseline = log_baseline[log_baseline["case:concept:name"].isin(case_ids)]

    print_workload_estimate(
        log_baseline,
        "BASELINE",
        args.n_sim,
        generate_new_cases_from_prev_log=False,
    )

    print(f"Running {args.n_sim} baseline simulation(s)...")
    baseline_durations_seconds = []
    for i in range(args.n_sim):
        run_start = datetime.now()
        print(f"[BASELINE] Starting run {i + 1}/{args.n_sim} at {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
        sim_log = sim_engine.apply(
            prev_log=log_baseline,
            generate_new_cases_from_prev_log=False,
        )
        sim_log = sim_log.sort_values(by=["case:concept:name", "time:timestamp"])
        out_path = baseline_folder / f"sim_{i + 1}.csv"
        sim_log.to_csv(out_path, index=False)
        print(f"Saved baseline run {i + 1}/{args.n_sim} -> {out_path}")
        run_end = datetime.now()
        print_run_timing(
            "BASELINE",
            i + 1,
            args.n_sim,
            run_start,
            run_end,
            baseline_durations_seconds,
        )
    print("Baseline simulation finished successfully!\n")

    # -----------------------------------------------------------------
    # 4. EXHAUSTIVE AND NSGA2 SIMULATIONS
    
    # Iterates through the advanced recommendation strategies (e.g., 'exhaustive' and 'nsga2').
    # For each method, it loads the previously computed recommendations, merges them 
    # with the current test log (applying any necessary data type conversions, like for BPI12), 
    # and constructs the final DataFrame for the simulation. 
    # The simulator engine is then applied 'n_sim' times, saving the outputs in dedicated directories.
    # -----------------------------------------------------------------
    methods_to_run = ["exhaustive", "nsga2"]

    for method in methods_to_run:
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

        if args.case_study.upper() == "BPI12":
            print("Applying BPI12 specific data type conversions...")
            rec_df = convert_dtypes_bpi12(rec_df, "recommendation")
            current_prev_log = convert_dtypes_bpi12(current_prev_log, "simulation")

        result_df = rec_df.set_index(CASE_ID_NAME)
        result_df = result_df.reset_index()

        # Keep case-id dtype consistent across logs/recommendations/filters.
        result_df[CASE_ID_NAME] = result_df[CASE_ID_NAME].astype(str)
        current_prev_log[CASE_ID_NAME] = current_prev_log[CASE_ID_NAME].astype(str)
        
        result_df["Next_activity"] = result_df["Next_activity"].fillna(current_prev_log["NEXT_ACTIVITY"])
        result_df["Next_resource"] = result_df["Next_resource"].fillna(current_prev_log["NEXT_RESOURCE"])
        rec_df = result_df
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
            case_ids = [str(c) for c in case_ids]
            log_rec = log_rec[log_rec["case:concept:name"].isin(case_ids)]
            if log_rec.empty:
                raise ValueError(
                    "No matching rows after --case_ids filtering. "
                    "Check case-id dtype/content and input file values."
                )

        print_workload_estimate(
            log_rec,
            method.upper(),
            args.n_sim,
            generate_new_cases_from_prev_log=False,
        )

        print(f"Running {args.n_sim} {method} simulation(s)...")
        method_durations_seconds = []
        for i in range(args.n_sim):
            run_start = datetime.now()
            print(f"[{method.upper()}] Starting run {i + 1}/{args.n_sim} at {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
            sim_log = sim_engine.apply(
                prev_log=log_rec,
                generate_new_cases_from_prev_log=False,
            )
            sim_log = sim_log.sort_values(by=["case:concept:name", "time:timestamp"])
            out_path = sim_folder / f"sim_{i + 1}.csv"
            sim_log.to_csv(out_path, index=False)
            print(f"Saved {method} run {i + 1}/{args.n_sim} -> {out_path}")
            run_end = datetime.now()
            print_run_timing(
                method.upper(),
                i + 1,
                args.n_sim,
                run_start,
                run_end,
                method_durations_seconds,
            )

            unreachable = getattr(sim_engine, "last_unreachable_recommendations", [])
            if unreachable:
                print(
                    f"WARNING: {len(unreachable)} recommendation(s) were unreachable "
                    f"from replayed prefix marking in {method} run {i + 1}."
                )
                unreachable_df = pd.DataFrame(unreachable)
                unreachable_out = sim_folder / f"sim_{i + 1}_unreachable_recommendations.csv"
                unreachable_df.to_csv(unreachable_out, index=False)
                print(f"Saved unreachable recommendations report -> {unreachable_out}")
        
        print(f"{method.capitalize()} simulation finished successfully!\n")

if __name__ == "__main__":
    main()