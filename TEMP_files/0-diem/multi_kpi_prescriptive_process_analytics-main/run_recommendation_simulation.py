#!/usr/bin/env python3
"""
Given a case study and a recommendation method, this script:
  1. Loads the event log (.xes) and the process model (.pnml)
  2. Loads the previously generated recommendations (.pkl) and the test log of running traces
  3. Builds the "recommended" event log (log_rec) by injecting the recommended next activity/resource into the test log
  4. Runs the ProSit simulator on the recommended log
  5. Saves the simulated log(s) as CSV files under
     case_studies/<case_study>/simulation_results/<method>/

Usage:
    python run_recommendation_simulation.py --case_study bpi17_before --method exhaustive --n_sim 10
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

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
        description="Run the recommendation simulation pipeline for a given case study/method."
    )
    parser.add_argument(
        "--case_study", type=str, default="bpi12",
        help="Name of the case study folder under case_studies/ (default: bpi12)",
    )
    parser.add_argument(
        "--method", type=str, default="exhaustive",
        help="Recommendation method name, used to locate the recommendations file "
             "and to name the simulation_results subfolder (default: exhaustive)",
    )
    parser.add_argument(
        "--n_sim", type=int, default=10,
        help="Number of simulation runs to perform (default: 10)",
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
    from utils.simulation_functions import build_recommender_df
    from utils.pre_processing_functions import convert_dtypes_bpi12
    from utils.get_features import get_features

    base_dir = Path(args.base_dir)
    case_study = args.case_study
    method = args.method

    case_dir = base_dir / "case_studies" / case_study
    sim_folder = case_dir / "simulation_results" / method

    # -----------------------------------------------------------------
    # 1. Create all required folders up front
    # -----------------------------------------------------------------
    sim_folder.mkdir(parents=True, exist_ok=True)
    print(f"Ensured folder exists: {sim_folder}")

    # -----------------------------------------------------------------
    # 2. Load event log, Petri net model, and test log
    # -----------------------------------------------------------------
    log_path = case_dir / f"log_{case_study}.xes"
    pnml_path = case_dir / f"model_{case_study}.pnml"
    test_log_path = case_dir / "test_log.csv"
    res_path = case_dir / f"recommendations_{case_study}_{method}"

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
        CASE_ID_NAME,
        ACTIVITY_COLUMN_NAME,
        RESOURCE_COLUMN_NAME,
        START_DATE_NAME,
        END_DATE_NAME,
        "NEXT_ACTIVITY",
        "NEXT_RESOURCE",
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

    selected_columns = base_columns + [
        c for c in case_specific_columns if c not in base_columns]
    
    prev_log = prev_log[selected_columns]

    # -----------------------------------------------------------------
    # 3. Load recommendations, convert to DataFrame, save as CSV
    # -----------------------------------------------------------------
    print(f"Loading recommendations: {res_path}.pkl")
    res = pd.read_pickle(f"{res_path}.pkl")

    rec_df = pd.DataFrame.from_dict(
        res, orient="index", columns=["Next_activity", "Next_resource"]
    ).reset_index()
    rec_df.rename(columns={"index": CASE_ID_NAME}, inplace=True)
    rec_df.to_csv(f"{res_path}.csv", index=False)

    # -----------------------------------------------------------------
    # 4. Type conversion / alignment
    # -----------------------------------------------------------------
    if args.case_study == "bpi12":
        print("Applying BPI12 specific data type conversions...")
        rec_df = convert_dtypes_bpi12(rec_df, "recommendation")
        prev_log = convert_dtypes_bpi12(prev_log, "simulation")

    result_df = rec_df.set_index(CASE_ID_NAME)
    result_df = result_df.reset_index()
    # Fill in any missing recommendations with the original test log values
    result_df["Next_activity"] = result_df["Next_activity"].fillna(prev_log["NEXT_ACTIVITY"])
    result_df["Next_resource"] = result_df["Next_resource"].fillna(prev_log["NEXT_RESOURCE"])
    rec_df = result_df
    rec_df.to_csv(f"{res_path}.csv", index=False)

    if rec_df.isna().sum().sum() > 0:
        print("Recommendations contain NaN values:")
        print(rec_df.isna().sum())

    # -----------------------------------------------------------------
    # 5. Build the recommendation-injected log
    # -----------------------------------------------------------------
    print("Building recommendations dataframe...")
    recommendations = {
        row["case:concept:name"]: {"act": row["Next_activity"], "res": row["Next_resource"]}
        for _, row in rec_df.iterrows()
    }
    log_rec = build_recommender_df(prev_log, recommendations)

    log_rec["start:timestamp"] = pd.to_datetime(log_rec["start:timestamp"], format="mixed")
    log_rec["time:timestamp"] = pd.to_datetime(log_rec["time:timestamp"], format="mixed")

    # Optional filtering to a specific subset of case ids
    if args.case_ids:
        with open(args.case_ids) as f:
            case_ids = [line.strip() for line in f if line.strip()]
        print(f"Restricting log_rec to {len(case_ids)} case ids from {args.case_ids}")
        log_rec = log_rec[log_rec["case:concept:name"].isin(case_ids)]

    # -----------------------------------------------------------------
    # 6. Set up and run the simulator
    # -----------------------------------------------------------------
    print("Discovering simulation parameters from event log...")
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(log, max_depth_tree=0)
    sim_engine = SimulatorEngine(params)

    print(f"Running {args.n_sim} simulation(s)...")
    for i in range(args.n_sim):
        sim_log = sim_engine.apply(prev_log=log_rec)
        sim_log = sim_log.sort_values(by=["case:concept:name", "time:timestamp"])
        out_path = sim_folder / f"sim_{i + 1}.csv"
        sim_log.to_csv(out_path, index=False)
        print(f"Saved run {i + 1}/{args.n_sim} -> {out_path}")

    print("Simulation finished successfully!")


if __name__ == "__main__":
    main()
