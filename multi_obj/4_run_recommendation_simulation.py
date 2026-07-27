#!/usr/bin/env python3
"""
Given a case study, this script:
  1. Loads the event log (.xes) and the process model (.pnml)
  2. Learns the simulation parameters ONCE.
  3. Simulates the baseline scenario.
  4. Simulates the recommendations for 'exhaustive' and 'nsga2' methods.
  5. Saves the simulated log(s) as CSV files under
     case_studies/<case_study>/prosit_simulation_results/<method>/

Usage:
    python 4_run_recommendation_simulation.py --case_study bpi17_before --n_sim 10
"""

import argparse
import sys
import uuid
import warnings
from pathlib import Path

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
    return parser.parse_args()


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
    # It also attempts to load an existing Petri net (PNML) model; if not found, 
    # it discovers one from scratch using the Inductive Miner algorithm and saves it.
    # Finally, it filters the test log to retain only the necessary features and target columns.
    # -----------------------------------------------------------------
    log_path = case_dir / f"log_{case_study}.xes"
    pnml_path = case_dir / "discovery_output" / f"diem_log_{case_study}.pnml"
    test_log_path = case_dir / "test_log.csv"

    print(f"Loading event log: {log_path}")
    log = xes_importer.apply(str(log_path))

    if pnml_path.exists():
        print(f"Loading Petri net: {pnml_path}")
        net, im, fm = pm4py.read_pnml(str(pnml_path))
    else:
        print(f"Discovering Petri net (Inductive Miner):")
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        pnml_path.parent.mkdir(parents=True, exist_ok=True)
        pm4py.write_pnml(net, im, fm, str(pnml_path))
        print(f"Saved mined model to {pnml_path}.")

    print(f"Loading test log (running traces): {test_log_path}")
    prev_log = pd.read_csv(
        test_log_path, parse_dates=[END_DATE_NAME, START_DATE_NAME]
    )

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
    # 2. Learn Simulation Parameters ONCE

    # This section initializes the Prosit simulator engine.
    # It extracts the simulation parameters directly from the event log 
    # and the discovered Petri net exactly once. 
    # This avoids redundant resource-heavy computations during the subsequent multiple simulation loops.
    # -----------------------------------------------------------------
    print("Discovering simulation parameters from event log")
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(log, max_depth_tree=0)
    sim_engine = SimulatorEngine(params)
    print("Simulation parameters learned successfully.\n")

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

    print(f"Running {args.n_sim} baseline simulation(s)...")
    for i in range(args.n_sim):
        sim_log = sim_engine.apply(prev_log=log_baseline)
        sim_log = sim_log.sort_values(by=["case:concept:name", "time:timestamp"])
        out_path = baseline_folder / f"sim_{i + 1}.csv"
        sim_log.to_csv(out_path, index=False)
        print(f"Saved baseline run {i + 1}/{args.n_sim} -> {out_path}")
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
            log_rec = log_rec[log_rec["case:concept:name"].isin(case_ids)]

        print(f"Running {args.n_sim} {method} simulation(s)...")
        for i in range(args.n_sim):
            sim_log = sim_engine.apply(prev_log=log_rec)
            sim_log = sim_log.sort_values(by=["case:concept:name", "time:timestamp"])
            out_path = sim_folder / f"sim_{i + 1}.csv"
            sim_log.to_csv(out_path, index=False)
            print(f"Saved {method} run {i + 1}/{args.n_sim} -> {out_path}")
        
        print(f"{method.capitalize()} simulation finished successfully!\n")

if __name__ == "__main__":
    main()