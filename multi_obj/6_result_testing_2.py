#!/usr/bin/env python3
"""
6_result_testing_2.py  (rewritten)

Validates the ProSiT simulator's BASELINE runs against the ENTIRE REAL log
(not just the test set), mirroring the paper's own simulator-validation 
methodology (Table 2: rate of positive outcome, real vs simulated log; 
Figure 5: trace-duration distributions).

No delta_CO / delta_RT computation here on purpose: this script answers a
single question -- "does the baseline simulation reproduce what really
happened for these cases, or not?" -- so we can tell whether the negative
deltas come from the simulator itself or from something upstream
(recommendation injection, indexing, alignment...).

Usage:
    python 6_result_testing_2.py --base_dir . --n_sim 10
    python 6_result_testing_2.py --base_dir . --case_study BAC   # single case study
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py

from utils.simulation_functions import (
    getting_remaining_time,
    status_encoding,
    case_id_name,
    start_date_name,
    end_date_name,
    activity_column_name,
    resource_column_name,
)
from utils.pre_processing_functions import convert_dtypes_bpi12

CASE_STUDIES = ["BAC", "BPI12", "bpi17_after", "bpi17_before"]
SIM_SUBDIR = "prosit_simulation_results"
BASELINE_FOLDER_NAME = "baseline"

ENCODED_ACTIVITY_BY_CASE_STUDY = {
    "BPI12": "O_ACCEPTED",
    "bpi17_after": "O_Accepted",
    "bpi17_before": "O_Accepted",
}
BPI12_DTYPE_CASE_STUDIES = {"BPI12"}


# ---------------------------------------------------------------------------
# Baseline simulation loading (same as before, but we also keep per-row data
# to compute total trace duration, not just remaining_time from a split point)
# ---------------------------------------------------------------------------
def load_baseline_sim(case_dir: Path, case_study: str, n_sim: int, encoded_activity):
    baseline_folder = case_dir / SIM_SUBDIR / BASELINE_FOLDER_NAME
    dataframes = []
    for i in range(n_sim):
        sim_path = baseline_folder / f"sim_{i + 1}.csv"
        sim = pd.read_csv(sim_path, dtype={case_id_name: str})
        if case_study in BPI12_DTYPE_CASE_STUDIES:
            sim = convert_dtypes_bpi12(sim, "simulation")
        sim = sim[[case_id_name, start_date_name, end_date_name,
                    activity_column_name, resource_column_name]]
        sim = getting_remaining_time(sim, case_id_name, end_date_name)  # also parses timestamps
        sim = status_encoding(sim, case_study, encoded_activity)
        sim[case_id_name] = sim[case_id_name].astype(str) + "_" + str(i + 1)
        dataframes.append(sim)
    return pd.concat(dataframes, ignore_index=True).reset_index(drop=True)


def case_level_stats(df: pd.DataFrame, case_id_col: str = case_id_name) -> pd.DataFrame:
    """One row per case: duration_days, status (0/1)."""
    g = df.groupby(case_id_col)
    duration = (g[end_date_name].max() - g[start_date_name].min()).dt.total_seconds() / 86400.0
    status = g["status"].first()
    return pd.DataFrame({"duration_days": duration, "status": status})


# ---------------------------------------------------------------------------
# Real log loading (entire dataset, no longer restricted to test set)
# ---------------------------------------------------------------------------
def load_real_log(case_dir: Path, case_study: str, encoded_activity):
    log_path = case_dir / f"log_{case_study}.xes"
    if not log_path.exists():
        raise FileNotFoundError(f"Real event log not found: {log_path}")

    log = pm4py.read_xes(str(log_path))
    real_df = pm4py.convert_to_dataframe(log)

    real_df[case_id_name] = real_df[case_id_name].astype(str)

    if real_df.empty:
        raise ValueError(
            f"The real log {log_path} is empty after loading."
        )

    if start_date_name not in real_df.columns:
        raise ValueError(
            f"'{start_date_name}' column not found in the real log for {case_study}; "
            f"cannot compute per-event duration the same way as the simulated log."
        )

    real_df[start_date_name] = pd.to_datetime(real_df[start_date_name], utc=True, errors="coerce")
    real_df[end_date_name] = pd.to_datetime(real_df[end_date_name], utc=True, errors="coerce")

    real_df = status_encoding(real_df, case_study, encoded_activity)
    return real_df


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_case_study(base_dir: Path, case_study: str, n_sim: int) -> dict:
    case_dir = base_dir / "case_studies" / case_study
    encoded_activity = ENCODED_ACTIVITY_BY_CASE_STUDY.get(case_study)

    print(f"  Loading baseline simulations ({n_sim} runs)...")
    baseline_sim = load_baseline_sim(case_dir, case_study, n_sim, encoded_activity)
    sim_stats = case_level_stats(baseline_sim)

    print(f"  Loading entire real log dataset...")
    real_df = load_real_log(case_dir, case_study, encoded_activity)
    real_stats = case_level_stats(real_df)

    def summarize(stats: pd.DataFrame, label: str) -> dict:
        return {
            f"{label}_n_traces": len(stats),
            f"{label}_pct_positive": 100 * stats["status"].mean(),
            f"{label}_mean_duration_days": stats["duration_days"].mean(),
            f"{label}_median_duration_days": stats["duration_days"].median(),
            f"{label}_std_duration_days": stats["duration_days"].std(),
        }

    result = {"case_study": case_study}
    result.update(summarize(real_stats, "real"))
    result.update(summarize(sim_stats, "sim_baseline"))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate baseline simulations against the ENTIRE real log "
                     "(paper Table 2 / Figure 5 style check). No delta_CO/delta_RT here."
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--n_sim", type=int, default=10)
    parser.add_argument("--case_study", type=str, default=None,
                         help="Run for a single case study only (default: all).")
    parser.add_argument("--out_csv", type=str, default="6_baseline_vs_real_validation.csv")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    case_studies = [args.case_study] if args.case_study else CASE_STUDIES

    results = []
    for case_study in case_studies:
        print(f"\n=== {case_study} ===")
        try:
            res = compare_case_study(base_dir, case_study, args.n_sim)
            results.append(res)
            print(
                f"  Real:     n={res['real_n_traces']:5d}  "
                f"%positive={res['real_pct_positive']:.1f}%  "
                f"duration(days) mean={res['real_mean_duration_days']:.2f} "
                f"median={res['real_median_duration_days']:.2f} "
                f"std={res['real_std_duration_days']:.2f}"
            )
            print(
                f"  Baseline: n={res['sim_baseline_n_traces']:5d}  "
                f"%positive={res['sim_baseline_pct_positive']:.1f}%  "
                f"duration(days) mean={res['sim_baseline_mean_duration_days']:.2f} "
                f"median={res['sim_baseline_median_duration_days']:.2f} "
                f"std={res['sim_baseline_std_duration_days']:.2f}"
            )
            gap_pct = res['sim_baseline_pct_positive'] - res['real_pct_positive']
            gap_dur = res['sim_baseline_mean_duration_days'] - res['real_mean_duration_days']
            print(f"  --> gap: %positive {gap_pct:+.1f} pt | mean duration {gap_dur:+.2f} days")
        except (FileNotFoundError, ValueError) as e:
            print(f"  [SKIPPED] {e}")

    if not results:
        print("No results computed -- check your paths.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved comparison table to {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()