#!/usr/bin/env python3
"""
6_baseline_vs_real_validation.py  (rewritten)

Validates the ProSiT simulator's BASELINE runs against the ENTIRE REAL log
(not just the test set), mirroring the paper's own simulator-validation
methodology (Table 2: rate of positive outcome, real vs simulated log;
Figure 5: trace-duration distributions).

No delta_CO / delta_RT computation here on purpose: this script answers a
single question -- "does the baseline simulation reproduce what really
happened for these cases, or not?" -- so we can tell whether the negative
deltas come from the simulator itself or from something upstream
(recommendation injection, indexing, alignment...).

ALSO validates, separately, the FULLY SIMULATED dataset (an entire training
set simulated from scratch by 9_generate_simulated_training_set.py, e.g.
case_studies/BPI12_sim/simulated_event_log.csv) against the same
real log -- a different comparison from the baseline one above: the baseline
runs replay each real case's own PREFIX and simulate only its continuation,
while the fully simulated dataset has no real prefix at all, every trace is
generated from the arrival process onward. Both answer "does simulated data
look like the real log", just for two different generation modes. For each
entry in CASE_STUDIES, this looks for a sibling case study named
"<case_study>_sim" (the project's own naming convention -- see
FULLY_SIMULATED_SUFFIX) and, if its simulated_event_log.csv exists, adds a
sim_full_* block of columns to that row; case studies without a "_sim"
sibling (BAC, bpi17_before/after) simply get no sim_full_* columns.

Usage:
    python 6_baseline_vs_real_validation.py --base_dir . --n_sim 10
    python 6_baseline_vs_real_validation.py --base_dir . --case_study BAC   # single case study
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
    "BPI12_not_reordered": "O_ACCEPTED",
    "BPI12_not_reordered_sim": "O_ACCEPTED",
    "BPI12": "O_ACCEPTED",
    "BPI12_sim": "O_ACCEPTED",
    "bpi17_after": "O_Accepted",
    "bpi17_before": "O_Accepted",
}
BPI12_DTYPE_CASE_STUDIES = {"BPI12_not_reordered", "BPI12_not_reordered_sim", "BPI12", "BPI12_sim"}

# Naming convention already used across the pipeline (9_generate_simulated_training_set.py
# and friends): a fully-simulated case study is named "<source_case_study>_sim".
FULLY_SIMULATED_SUFFIX = "_sim"
TRAIN_DATA_FILENAME = "train_data.csv"


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


def load_fully_simulated_log(sim_case_dir: Path, sim_case_study: str, encoded_activity) -> pd.DataFrame:
    """Load a fully-simulated dataset built from scratch by
    9_generate_simulated_training_set.py (one flat CSV, not n_sim separate runs like the
    baseline: every trace is generated from the arrival process onward, no real prefix).

    Reads train_data.csv, NOT simulated_event_log.csv: the latter is the raw, oversampled
    simulation output BEFORE the '>3 events' filter and the trim back to the target training
    size (see 9_generate_simulated_training_set.py's --oversample, default 1.4x), so its trace
    count doesn't match anything meaningful on its own. train_data.csv is that same simulated
    log already trimmed to exactly the real training set's trace count -- the comparison this
    function feeds is only meaningful against that count, not the inflated raw one. train_data.csv
    still carries every raw event column (case id, activity, timestamps, resource) needed here,
    just with extra engineered feature columns alongside them, which are simply not selected below.
    """
    sim_log_path = sim_case_dir / TRAIN_DATA_FILENAME
    if not sim_log_path.exists():
        raise FileNotFoundError(f"Fully-simulated training set not found: {sim_log_path}")

    sim = pd.read_csv(sim_log_path, dtype={case_id_name: str})
    if sim_case_study in BPI12_DTYPE_CASE_STUDIES:
        sim = convert_dtypes_bpi12(sim, "simulation")
    sim = sim[[case_id_name, start_date_name, end_date_name,
                activity_column_name, resource_column_name]]
    sim = getting_remaining_time(sim, case_id_name, end_date_name)  # also parses timestamps
    sim = status_encoding(sim, sim_case_study, encoded_activity)
    return sim


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

    sim_full_case_study = case_study + FULLY_SIMULATED_SUFFIX
    sim_full_case_dir = base_dir / "case_studies" / sim_full_case_study
    try:
        print(f"  Loading fully-simulated dataset ({sim_full_case_study})...")
        sim_full_encoded_activity = ENCODED_ACTIVITY_BY_CASE_STUDY.get(sim_full_case_study, encoded_activity)
        sim_full_df = load_fully_simulated_log(sim_full_case_dir, sim_full_case_study, sim_full_encoded_activity)
        sim_full_stats = case_level_stats(sim_full_df)
        result.update(summarize(sim_full_stats, "sim_full"))
    except FileNotFoundError as e:
        print(f"  [no fully-simulated counterpart] {e}")

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

    print("\n" + "=" * 70)
    print("SEZIONE 1: SIMULAZIONI DA PREFISSO vs LOG REALE")
    print("=" * 70)

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

    results_with_sim_full = [r for r in results if "sim_full_n_traces" in r]
    if results_with_sim_full:
        print("\n" + "=" * 70)
        print("SEZIONE 2: DATASET TOTALMENTE SIMULATO vs LOG REALE")
        print("=" * 70)
        for res in results_with_sim_full:
            print(f"\n=== {res['case_study']} vs {res['case_study']}{FULLY_SIMULATED_SUFFIX} ===")
            print(
                f"  Real:            n={res['real_n_traces']:5d}  "
                f"%positive={res['real_pct_positive']:.1f}%  "
                f"duration(days) mean={res['real_mean_duration_days']:.2f} "
                f"median={res['real_median_duration_days']:.2f} "
                f"std={res['real_std_duration_days']:.2f}"
            )
            print(
                f"  Fully simulated: n={res['sim_full_n_traces']:5d}  "
                f"%positive={res['sim_full_pct_positive']:.1f}%  "
                f"duration(days) mean={res['sim_full_mean_duration_days']:.2f} "
                f"median={res['sim_full_median_duration_days']:.2f} "
                f"std={res['sim_full_std_duration_days']:.2f}"
            )
            gap_pct_full = res['sim_full_pct_positive'] - res['real_pct_positive']
            gap_dur_full = res['sim_full_mean_duration_days'] - res['real_mean_duration_days']
            print(f"  --> gap: %positive {gap_pct_full:+.1f} pt | mean duration {gap_dur_full:+.2f} days")

    if not results:
        print("No results computed -- check your paths.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved comparison table to {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()