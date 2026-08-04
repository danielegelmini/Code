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

CASE_ID_NAME = "case:concept:name"
ACTIVITY_COLUMN_NAME = "concept:name"
RESOURCE_COLUMN_NAME = "org:resource"
START_DATE_NAME = "start:timestamp"
END_DATE_NAME = "time:timestamp"

SIM_SUBDIR = "prosit_simulation_results"
BASELINE_FOLDER_NAME = "baseline"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={CASE_ID_NAME: str})
    df[END_DATE_NAME] = pd.to_datetime(df[END_DATE_NAME], format="mixed", utc=True, errors="coerce")
    return df


def case_sequence(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """All rows for one case, sorted chronologically."""
    return df[df[CASE_ID_NAME] == case_id].sort_values(END_DATE_NAME).reset_index(drop=True)


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
        real_cont = real_seq.iloc[prefix_len:].reset_index(drop=True)

        sim_conts = []
        for sim_df in sim_dfs:
            sim_seq = case_sequence(sim_df, case_id)
            sim_cont = sim_seq.iloc[prefix_len:].reset_index(drop=True) if len(sim_seq) >= prefix_len else pd.DataFrame()
            sim_conts.append(sim_cont)

        # --- prefix summary (last few events for context) ---
        print(f"\n  Prefix length: {prefix_len} event(s). Last event(s) of the prefix:")
        tail = prefix_seq.tail(3)[[ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME, END_DATE_NAME]]
        for _, r in tail.iterrows():
            print(f"    ... {r[ACTIVITY_COLUMN_NAME]} / {r[RESOURCE_COLUMN_NAME]} @ {r[END_DATE_NAME]}")

        if real_cont.empty:
            print("\n  [NOTE] No real continuation found for this case in test_data.csv "
                  "(prefix might already be the whole trace).")

        empty_sims = sum(1 for s in sim_conts if s.empty)
        if empty_sims:
            print(f"\n  [NOTE] {empty_sims}/{args.n_sim} '{args.method}' simulation(s) have fewer rows "
                  f"than the prefix length for this case -- check for missing/short traces.")

        # --- AGGIUNTA: CHECK RAPIDO DELLA PRIMA ATTIVITÀ ---
        print(f"\n  [QUICK CHECK] Prima attività simulata (Step 1):")
        for i, sim_cont in enumerate(sim_conts):
            if not sim_cont.empty:
                first_act = sim_cont.iloc[0][ACTIVITY_COLUMN_NAME]
                first_res = sim_cont.iloc[0][RESOURCE_COLUMN_NAME]
                print(f"    SIM_{i + 1}: {first_act} / {first_res}")
            else:
                print(f"    SIM_{i + 1}: (Nessuna continuazione)")

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