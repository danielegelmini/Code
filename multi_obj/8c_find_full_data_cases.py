#!/usr/bin/env python3
"""
Find case_ids that have COMPLETE data in the "Simulated" panel of
8_plot_case_comparison.py / 8b_plot_case_comparison_vs_sim_model.py: all k
ranks have a usable ProSiT simulation result (sim_status_method_mean and
sim_remaining_time_method_mean_sigmoid_mm both non-null) AND the
no-recommendation baseline is present too.

Why this is needed: a rank's simulated point is missing whenever every one of
that rank's n_sim runs reported the recommendation unreachable from the
replayed prefix marking (sim_rec_applied_fraction == 0 -- see
5_result_computation.py / prosit/simulator.py's strict-recommendation
handling). case_id 198232 in BPI12 is one such example: 3 of its 5 ranks
(all recommending A_CANCELLED) have no simulated point at all, so the
"Simulated" panel only shows 2 of the 5 rank dots plus the baseline. This
script finds OTHER case_ids where all k ranks (not just some) plus the
baseline show up, so the 3-panel comparison figure is showing everything
there is to show for that case.

Among the qualifying (complete-data) case_ids, results are additionally
ranked by a "spread" score -- rewarding cases whose k ranks and baseline land
in visually separated positions in the (outcome, 1-sigmoid_mm) plane, since
(per prior experience picking examples for these figures) a case where every
point coincides makes for an uninformative plot. Grouped into short/medium/
long prefix-length bands (tertiles among the qualifying cases) so you can
pick one interesting example per band, the same way the BAC examples in
CODICI DA RUNNARE.txt were chosen.

Usage:
    python 8c_find_full_data_cases.py
    python 8c_find_full_data_cases.py --case_study BPI12 --method exhaustive --top_n 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.get_features import load_case_study
from utils.simulation_functions import case_id_name

EVAL_TABLES_SUBDIR = "evaluation_tables"
DEFAULT_CASE_STUDY = "BPI12"
DEFAULT_METHOD = "exhaustive"


def get_prefix_lengths(test_log: pd.DataFrame) -> pd.Series:
    """str(case_id) -> number of historical events in test_log.csv (the prefix length at the
    recommendation point). Same definition as 8_plot_case_comparison.py's get_prefix_length,
    vectorised over every case at once."""
    counts = test_log.groupby(case_id_name).size()
    counts.index = counts.index.astype(str)
    return counts


def find_complete_cases(eval_table: pd.DataFrame) -> pd.DataFrame:
    """One row per case_id that has a finite sim_status_method_mean AND
    sim_remaining_time_method_mean_sigmoid_mm for EVERY one of its k ranks, plus a finite
    baseline (sim_status_baseline_mean / sim_remaining_time_baseline_mean_sigmoid_mm).

    Returns a DataFrame with case_id, k_total, and a 'spread_score' column (higher = more
    visually separated rank/baseline points in the (outcome, 1-sigmoid_mm) plane -- range of
    the k ranks' positions plus their mean distance from the baseline).
    """
    rows = []
    for cid, g in eval_table.groupby(case_id_name):
        g = g.sort_values("rank")
        k_total = int(g["k_total"].iloc[0])
        if len(g) != k_total:
            continue  # some ranks missing entirely (no recommendation at all for that rank)

        sim_x = g["sim_status_method_mean"].to_numpy(dtype=float)
        sim_y = 1.0 - g["sim_remaining_time_method_mean_sigmoid_mm"].to_numpy(dtype=float)
        if not np.all(np.isfinite(sim_x)) or not np.all(np.isfinite(sim_y)):
            continue  # at least one rank has no usable simulation run

        base_x = float(g["sim_status_baseline_mean"].iloc[0])
        base_y = 1.0 - float(g["sim_remaining_time_baseline_mean_sigmoid_mm"].iloc[0])
        if not (np.isfinite(base_x) and np.isfinite(base_y)):
            continue  # baseline itself missing

        rank_spread = float(np.hypot(sim_x.max() - sim_x.min(), sim_y.max() - sim_y.min()))
        mean_dist_from_baseline = float(np.mean(np.hypot(sim_x - base_x, sim_y - base_y)))
        spread_score = rank_spread + mean_dist_from_baseline

        rows.append({
            case_id_name: str(cid),
            "k_total": k_total,
            "rank_spread": rank_spread,
            "mean_dist_from_baseline": mean_dist_from_baseline,
            "spread_score": spread_score,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="List case_ids with a full k-rank + baseline ProSiT-simulated panel, "
                    "ranked by how visually separated the points are."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_study", type=str, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--top_n", type=int, default=5,
                         help="How many case_ids to print per prefix-length band (default: 5).")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    case_dir = base_dir / "case_studies" / args.case_study
    table_path = case_dir / EVAL_TABLES_SUBDIR / f"{args.method}_all_ranks.csv"
    if not table_path.exists():
        raise SystemExit(f"No evaluation table at {table_path} -- run 5_result_computation.py first.")

    eval_table = pd.read_csv(table_path, dtype={case_id_name: str})
    complete = find_complete_cases(eval_table)
    if complete.empty:
        raise SystemExit(
            f"No case_id in {table_path} has simulated data for all its ranks plus the baseline."
        )

    _, _, test_log = load_case_study(args.case_study)
    prefix_lengths = get_prefix_lengths(test_log)
    complete["prefix_length"] = complete[case_id_name].map(prefix_lengths)
    complete = complete.dropna(subset=["prefix_length"]).copy()
    complete["prefix_length"] = complete["prefix_length"].astype(int)

    print(f"{len(complete)} / {eval_table[case_id_name].nunique()} case_ids in {table_path.name} "
          f"have a full k-rank + baseline simulated panel.\n")

    # Tertile bands over prefix length, same short/medium/long split used when picking the
    # existing example case_ids (see CODICI DA RUNNARE.txt).
    q1, q2 = complete["prefix_length"].quantile([1 / 3, 2 / 3])
    bands = [
        ("short", complete["prefix_length"] <= q1),
        ("medium", (complete["prefix_length"] > q1) & (complete["prefix_length"] <= q2)),
        ("long", complete["prefix_length"] > q2),
    ]

    for band_name, mask in bands:
        band_df = complete.loc[mask].sort_values("spread_score", ascending=False).head(args.top_n)
        print(f"--- {band_name} prefix (<= {q1:.0f} / <= {q2:.0f} / > {q2:.0f} events) "
              f"-- top {len(band_df)} by spread ---")
        if band_df.empty:
            print("  (none)")
        else:
            print(band_df[[case_id_name, "prefix_length", "k_total", "rank_spread",
                           "mean_dist_from_baseline", "spread_score"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()

    print("Try one with:")
    example_cid = complete.sort_values("spread_score", ascending=False)[case_id_name].iloc[0]
    print(f"  python 8_plot_case_comparison.py --case_study {args.case_study} --method {args.method} --case_id {example_cid}")
    print(f"  python 8b_plot_case_comparison_vs_sim_model.py --case_study {args.case_study} --method {args.method} --case_id {example_cid}")


if __name__ == "__main__":
    main()

# Running commands:
# python 8c_find_full_data_cases.py --case_study BPI12 --method exhaustive
