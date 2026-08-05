#!/usr/bin/env python3
"""
7_inspect_baseline_traces.py

Diagnostic tool: for a handful of randomly sampled case ids, shows side by
side, step by step:
  - the real prefix (test_log.csv)
  - what REALLY happened next (test_data.csv, the ground-truth continuation)
  - what each of the n_sim simulations generated next, for a chosen method
      (case_studies/<case_study>/prosit_simulation_results/<method>/sim_i.csv,
      where <method> is 'baseline', 'exhaustive', or 'nsga2')

This is meant purely for visual/manual inspection -- does a given method's
simulated continuation look at all plausible compared to reality? -- not for
computing any aggregate metric.

Usage:
    python 7_inspect_baseline_traces.py --base_dir . --case_study BAC --n_cases 5
    python 7_inspect_baseline_traces.py --base_dir . --case_study BAC --method exhaustive --n_cases 5
    python 7_inspect_baseline_traces.py --base_dir . --case_study BAC --method nsga2 --case_ids 201810001660,201810002650
"""

import argparse
import random
from pathlib import Path

import pandas as pd

from utils.simulation_functions import check_recommendation_following

CASE_ID_NAME = "case:concept:name"
ACTIVITY_COLUMN_NAME = "concept:name"
RESOURCE_COLUMN_NAME = "org:resource"
START_DATE_NAME = "start:timestamp"
END_DATE_NAME = "time:timestamp"

SIM_SUBDIR = "prosit_simulation_results"
BASELINE_FOLDER_NAME = "baseline"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={CASE_ID_NAME: str})
    df[START_DATE_NAME] = pd.to_datetime(df[START_DATE_NAME], format="mixed", utc=True, errors="coerce")
    df[END_DATE_NAME] = pd.to_datetime(df[END_DATE_NAME], format="mixed", utc=True, errors="coerce")
    return df


def case_sequence(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """All rows for one case, sorted chronologically."""
    return df[df[CASE_ID_NAME] == case_id].sort_values([START_DATE_NAME, END_DATE_NAME]).reset_index(drop=True)


def remove_prefix_rows(full_seq: pd.DataFrame, prefix_seq: pd.DataFrame) -> pd.DataFrame:
    """Remove the exact prefix rows from a full case sequence, preserving duplicates correctly."""
    if full_seq.empty or prefix_seq.empty:
        return full_seq.copy()

    key_cols = [ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME, START_DATE_NAME, END_DATE_NAME]

    full_keyed = full_seq.copy()
    prefix_keyed = prefix_seq.copy()

    full_keyed["__dup_idx"] = full_keyed.groupby(key_cols).cumcount()
    prefix_keyed["__dup_idx"] = prefix_keyed.groupby(key_cols).cumcount()
    prefix_keyed["__is_prefix"] = True

    merged = full_keyed.merge(
        prefix_keyed[key_cols + ["__dup_idx", "__is_prefix"]],
        on=key_cols + ["__dup_idx"],
        how="left",
    )

    out = merged[merged["__is_prefix"].isna()].drop(columns=["__dup_idx", "__is_prefix"]).copy()
    return out.sort_values(END_DATE_NAME).reset_index(drop=True)


def extract_continuation_after_prefix(
    full_seq: pd.DataFrame,
    prefix_seq: pd.DataFrame,
    prefix_end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Extract continuation rows after the prefix boundary and count temporal anomalies."""
    generated_only = remove_prefix_rows(full_seq, prefix_seq)
    if generated_only.empty:
        return generated_only, 0

    strictly_pre_boundary = generated_only[generated_only[END_DATE_NAME] < prefix_end_ts]
    continuation = generated_only[generated_only[END_DATE_NAME] >= prefix_end_ts].sort_values(
        [START_DATE_NAME, END_DATE_NAME]
    ).reset_index(drop=True)
    return continuation, len(strictly_pre_boundary)


def format_first_start_batch(sim_cont: pd.DataFrame) -> str:
    if sim_cont.empty:
        return "(Nessuna continuazione)"

    first_start = sim_cont[START_DATE_NAME].min()
    batch = sim_cont[sim_cont[START_DATE_NAME] == first_start]
    rendered = [format_step(row) for _, row in batch.iterrows()]
    return " | ".join(rendered)


def format_step(row) -> str:
    if row is None:
        return ""
    act = str(row[ACTIVITY_COLUMN_NAME])
    res = str(row[RESOURCE_COLUMN_NAME])
    return f"{act} / {res}"


def build_comparison_table(prefix_len: int, real_cont: pd.DataFrame,
                            sim_conts: list, max_steps: int, method_label: str) -> pd.DataFrame:
    n_steps = max(len(real_cont), *(len(s) for s in sim_conts)) if sim_conts else len(real_cont)
    n_steps = min(n_steps, max_steps)

    rows = []
    for step in range(n_steps):
        row = {"step_after_prefix": step + 1}
        row["REAL"] = format_step(real_cont.iloc[step]) if step < len(real_cont) else ""
        for i, sim_cont in enumerate(sim_conts):
            row[f"{method_label}_SIM_{i + 1}"] = format_step(sim_cont.iloc[step]) if step < len(sim_cont) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare, step by step, prefix + real continuation vs "
                    "baseline-simulated continuations for sampled case ids."
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--case_study", type=str, required=True)
    parser.add_argument("--method", type=str, default="nsga2",
                         choices=["baseline", "exhaustive", "nsga2"],
                         help="Which simulation folder to compare against the real "
                              "continuation: 'baseline' (no recommendation), "
                              "'exhaustive', or 'nsga2' (default: nsga2).")
    parser.add_argument("--n_cases", type=int, default=5,
                         help="Number of random case ids to inspect (ignored if --case_ids given).")
    parser.add_argument("--case_ids", type=str, default=None,
                         help="Comma-separated list of specific case ids to inspect, instead of random sampling.")
    parser.add_argument("--n_sim", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=8,
                         help="Max number of post-prefix steps to display per case.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="inspection_output")
    parser.add_argument(
        "--check_recommendations",
        action="store_true",
        help="Check whether each simulation contains the recommended activity/resource pair "
             "in the first post-prefix start-time batch and save a CSV summary.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    case_dir = base_dir / "case_studies" / args.case_study
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test_log.csv (prefixes) and test_data.csv (real continuations)...")
    test_log = load_events(case_dir / "test_log.csv")
    test_data = load_events(case_dir / "test_data.csv")

    print(f"Loading {args.n_sim} '{args.method}' simulation file(s)...")
    sim_folder = case_dir / SIM_SUBDIR / args.method
    if args.method == "baseline":
        print("NOTE: 'baseline' uses the no-recommendation simulation; use '--method exhaustive' or '--method nsga2' to inspect recommendation-aware behavior.")
    sim_dfs = [load_events(sim_folder / f"sim_{i + 1}.csv") for i in range(args.n_sim)]

    if args.case_ids:
        sampled_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        all_ids = test_log[CASE_ID_NAME].unique().tolist()
        random.seed(args.seed)
        sampled_ids = random.sample(all_ids, min(args.n_cases, len(all_ids)))

    if args.check_recommendations:
        if args.method == "baseline":
            print("\n[CHECK] Skipping recommendation check for baseline method.")
        else:
            rec_path = case_dir / "recommendations" / f"recommendations_{args.case_study}_{args.method}.csv"
            if not rec_path.exists():
                print(f"\n[CHECK] Recommendation file not found: {rec_path}")
            else:
                rec_df = pd.read_csv(rec_path, dtype={CASE_ID_NAME: str})
                recommendations = {
                    str(row[CASE_ID_NAME]): {
                        "act": row.get("Next_activity", None),
                        "res": row.get("Next_resource", None),
                    }
                    for _, row in rec_df.iterrows()
                }

                check_df = check_recommendation_following(test_log, recommendations, sim_dfs)
                check_out = out_dir / f"{args.case_study}_{args.method}_recommendation_check.csv"
                check_df.to_csv(check_out, index=False)

                ok_mask = check_df["status"].eq("ok")
                ok_total = int(ok_mask.sum())
                ok_match = int((check_df[ok_mask]["match_recommendation"] == True).sum())
                match_rate = (ok_match / ok_total * 100) if ok_total else 0.0
                print(f"\n[CHECK] Saved recommendation adherence report to {check_out}")
                print(f"[CHECK] Match on valid rows: {ok_match}/{ok_total} ({match_rate:.2f}%)")

    print(f"\nInspecting {len(sampled_ids)} case id(s): {sampled_ids}\n")

    for case_id in sampled_ids:
        print("=" * 100)
        print(f"CASE ID: {case_id}")
        print("=" * 100)

        prefix_seq = case_sequence(test_log, case_id)
        real_seq = case_sequence(test_data, case_id)

        if prefix_seq.empty:
            print(f"  [SKIPPED] case id not found in test_log.csv")
            continue

        prefix_len = len(prefix_seq)
        prefix_end_ts = prefix_seq[END_DATE_NAME].max()

        real_cont, real_pre_boundary = extract_continuation_after_prefix(real_seq, prefix_seq, prefix_end_ts)

        sim_conts = []
        sim_pre_boundary_counts = []
        for sim_df in sim_dfs:
            sim_seq = case_sequence(sim_df, case_id)
            sim_cont, pre_boundary_count = extract_continuation_after_prefix(sim_seq, prefix_seq, prefix_end_ts)
            sim_conts.append(sim_cont)
            sim_pre_boundary_counts.append(pre_boundary_count)

        # --- prefix summary (last few events for context) ---
        print(f"\n  Prefix length: {prefix_len} event(s). Last event(s) of the prefix:")
        tail = prefix_seq.tail(3)[[ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME, END_DATE_NAME]]
        for _, r in tail.iterrows():
            print(f"    ... {r[ACTIVITY_COLUMN_NAME]} / {r[RESOURCE_COLUMN_NAME]} @ {r[END_DATE_NAME]}")

        if real_cont.empty:
            print("\n  [NOTE] No real continuation found for this case in test_data.csv "
                  "(prefix might already be the whole trace).")

        if real_pre_boundary:
            print(f"\n  [NOTE] Found {real_pre_boundary} real event(s) ending before prefix boundary after prefix-removal; check data consistency.")

        empty_sims = sum(1 for s in sim_conts if s.empty)
        if empty_sims:
            print(f"\n  [NOTE] {empty_sims}/{args.n_sim} '{args.method}' simulation(s) have no post-prefix continuation after boundary filtering.")

        sims_with_temporal_anomaly = sum(1 for c in sim_pre_boundary_counts if c > 0)
        if sims_with_temporal_anomaly:
            print(
                f"\n  [NOTE] {sims_with_temporal_anomaly}/{args.n_sim} '{args.method}' simulation(s) generated "
                "event(s) ending before the prefix boundary (temporal anomaly)."
            )

        # --- AGGIUNTA: CHECK RAPIDO DELLA PRIMA ATTIVITÀ ---
        print(f"\n  [QUICK CHECK] Prima attività simulata (Step 1 by earliest completion):")
        for i, sim_cont in enumerate(sim_conts):
            if not sim_cont.empty:
                first_act = sim_cont.iloc[0][ACTIVITY_COLUMN_NAME]
                first_res = sim_cont.iloc[0][RESOURCE_COLUMN_NAME]
                print(f"    SIM_{i + 1}: {first_act} / {first_res}")
            else:
                print(f"    SIM_{i + 1}: (Nessuna continuazione)")

        print(f"\n  [QUICK CHECK] Eventi con start minimo post-prefix (tie-aware):")
        for i, sim_cont in enumerate(sim_conts):
            print(f"    SIM_{i + 1}: {format_first_start_batch(sim_cont)}")

        # --- step-by-step comparison table ---
        method_label = args.method.upper()
        table = build_comparison_table(prefix_len, real_cont, sim_conts, args.max_steps, method_label)

        out_path = out_dir / f"{args.case_study}_{args.method}_{case_id}_comparison.csv"
        table.to_csv(out_path, index=False)
        print(f"\n  Saved to {out_path}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()