#!/usr/bin/env python3
"""
3_sumup_recommendations.py

Summary of the recommendations produced by 3_run_experiment.py, to read before
launching the simulations (4_run_recommendation_simulation.py): for every case
study and every rank 1..k, how many test cases actually received a
recommendation (activity AND resource) and which activities are recommended most.

A test case can have fewer than k recommendations: the transition system offered
no candidate, no candidate passed the confidence filter, or the Pareto front had
fewer than k points. The simulations of a rank only cover the cases that have a
recommendation at that rank, so this tells in advance how many cases each rank
of step 5 will be computed on.

Writes 3_recommendations_summary.csv (one row per case study, method and rank).

Usage:
    python 3_sumup_recommendations.py
    python 3_sumup_recommendations.py --case_studies "BAC,bpi17_before" --method exhaustive --k 5
"""

import argparse
from pathlib import Path

import pandas as pd

CASE_ID_NAME = "case:concept:name"
DEFAULT_CASE_STUDIES = "BAC,BPI12,bpi17_before,bpi17_after"


def summarize(base_dir: Path, case_study: str, method: str, k: int, n_top: int) -> list:
    case_dir = base_dir / "case_studies" / case_study
    n_test = pd.read_csv(case_dir / "test_log.csv", usecols=[CASE_ID_NAME])[CASE_ID_NAME].nunique()
    rows, n_recs_per_case = [], None
    for rank in range(1, k + 1):
        path = case_dir / "recommendations" / f"recommendations_{case_study}_{method}_top{rank}of{k}.csv"
        if not path.exists():
            print(f"  [missing] {path}")
            continue
        rec = pd.read_csv(path, dtype=str)
        has_rec = rec["Next_activity"].notna() & rec["Next_resource"].notna()
        per_case = has_rec.groupby(rec[CASE_ID_NAME]).any()
        n_recs_per_case = per_case.astype(int) if n_recs_per_case is None else n_recs_per_case.add(per_case.astype(int), fill_value=0)
        top = rec.loc[has_rec, "Next_activity"].value_counts(normalize=True).head(n_top)
        rows.append({
            "case_study": case_study,
            "method": method,
            "rank": rank,
            "test_cases": n_test,
            "with_recommendation": int(per_case.sum()),
            "pct_with_recommendation": 100 * per_case.sum() / n_test,
            "top_activities": "; ".join(f"{a} ({s:.0%})" for a, s in top.items()),
        })
    if n_recs_per_case is not None:
        dist = n_recs_per_case.value_counts().sort_index()
        print(f"  cases by number of recommendations: " + ", ".join(f"{int(n)}: {c}" for n, c in dist.items()))
    return rows


def main():
    parser = argparse.ArgumentParser(description="How many test cases have a recommendation at each rank, before simulating.")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--case_studies", type=str, default=DEFAULT_CASE_STUDIES,
                        help=f"Comma-separated case studies (default: {DEFAULT_CASE_STUDIES}).")
    parser.add_argument("--method", type=str, default="exhaustive")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_top", type=int, default=3, help="Most recommended activities listed per rank.")
    parser.add_argument("--out_csv", type=str, default="3_recommendations_summary.csv")
    args = parser.parse_args()

    rows = []
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        print(f"\n=== {cs} ===")
        rows += summarize(Path(args.base_dir), cs, args.method, args.k, args.n_top)
    if not rows:
        print("No recommendation files found.")
        return
    table = pd.DataFrame(rows)
    table.to_csv(args.out_csv, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print("\n" + table.round(1).to_string(index=False))
    print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()
