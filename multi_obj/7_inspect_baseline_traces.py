#!/usr/bin/env python3
"""
7_inspect_baseline_traces.py

Diagnostic tool: for a handful of randomly sampled case ids, shows side by
side, step by step:
  - the real prefix (test_log.csv)
  - what REALLY happened next (test_data.csv, the ground-truth continuation)
  - what each of the n_sim simulations generated next, for a chosen method
      (case_studies/<case_study>/prosit_simulation_results/<method>/<rank>/sim_i.csv
      for 'exhaustive'/'nsga2', or .../baseline/sim_i.csv for 'baseline')
  - the recommended (activity / resource) pair for that case, taken from
      recommendations_<case_study>_<method>_top<rank>of<k>.csv, so it is easy
      to eyeball whether the simulated continuation actually follows it.

This is meant purely for visual/manual inspection -- does a given method's
simulated continuation look at all plausible compared to reality, and does it
respect the recommendation? -- not for computing any aggregate metric.

Recommendations are now produced per Pareto rank (top1of5, top2of5, ...). By
default this script inspects rank 1 (the top recommendation); pass --rank 2
(or another value) to inspect a different one. 'baseline' has no
recommendation and no rank, so --rank is ignored for it.

With --validate the script switches from per-case inspection to a single
dataset-level check: for EVERY case in the recommendation file it verifies that
the recommended (activity / resource) pair is the first post-prefix event in
all n_sim simulations, and prints one pass/fail summary (listing the cases that
don't follow it, and excluding the ones legitimately not expected to -- no
recommendation at that rank, excluded_no_recommendation.csv, or unreachable
from the replayed prefix). No files are written; exit code is 1 on FAIL.

Usage:
    python 7_inspect_baseline_traces.py --case_study BAC --n_cases 5
    python 7_inspect_baseline_traces.py --case_study BAC --method exhaustive --n_cases 5
    python 7_inspect_baseline_traces.py --case_study BAC --method nsga2 --rank 2 --n_cases 5
    python 7_inspect_baseline_traces.py --case_study BAC --method nsga2 --case_ids 201810001660,201810002650
    python 7_inspect_baseline_traces.py --case_study BAC --method nsga2 --validate
"""

import argparse
import random
import re
import sys
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


def resolve_sim_folder(case_dir: Path, method: str, rank: int) -> Path:
    """Simulation folder for a method: rank-independent for baseline, <method>/<rank>/ otherwise."""
    if method == BASELINE_FOLDER_NAME:
        return case_dir / SIM_SUBDIR / BASELINE_FOLDER_NAME
    return case_dir / SIM_SUBDIR / method / str(rank)


def resolve_recommendation_file(case_dir: Path, case_study: str, method: str, rank: int) -> Path | None:
    """recommendations_<case_study>_<method>_top<rank>of<k>.csv for the requested rank (any k)."""
    rec_dir = case_dir / "recommendations"
    if not rec_dir.exists():
        return None
    pattern = re.compile(
        rf"^recommendations_{re.escape(case_study)}_{re.escape(method)}_top{rank}of(\d+)\.csv$"
    )
    matches = sorted(f for f in rec_dir.iterdir() if pattern.match(f.name))
    return matches[0] if matches else None


def _clean(value):
    """Normalise missing recommendation cells (NaN / empty string) to None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def load_recommendations(rec_path: Path) -> dict:
    """case_id -> {'act': Next_activity, 'res': Next_resource}. Missing cells become None."""
    rec_df = pd.read_csv(rec_path, dtype={CASE_ID_NAME: str})
    return {
        str(row[CASE_ID_NAME]): {
            "act": _clean(row.get("Next_activity", None)),
            "res": _clean(row.get("Next_resource", None)),
        }
        for _, row in rec_df.iterrows()
    }


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


def batch_follows_recommendation(sim_cont: pd.DataFrame, rec_act, rec_res) -> bool | None:
    """Whether the first post-prefix start-time batch contains the recommended (act[, res]) pair."""
    if rec_act is None or (isinstance(rec_act, float) and pd.isna(rec_act)):
        return None
    if sim_cont.empty:
        return False
    first_start = sim_cont[START_DATE_NAME].min()
    batch = sim_cont[sim_cont[START_DATE_NAME] == first_start]
    if rec_res is None or (isinstance(rec_res, float) and pd.isna(rec_res)):
        return bool((batch[ACTIVITY_COLUMN_NAME] == rec_act).any())
    return bool(((batch[ACTIVITY_COLUMN_NAME] == rec_act) & (batch[RESOURCE_COLUMN_NAME] == rec_res)).any())


def group_by_case(df: pd.DataFrame) -> dict:
    """case_id -> chronologically sorted sub-DataFrame, built in one pass."""
    return {
        cid: sub.sort_values([START_DATE_NAME, END_DATE_NAME]).reset_index(drop=True)
        for cid, sub in df.groupby(CASE_ID_NAME)
    }


def load_diagnostic_case_ids(sim_folder: Path, filename_glob: str) -> set:
    """Union of the case:concept:name column across every diagnostic CSV matching filename_glob."""
    ids: set = set()
    for path in sorted(sim_folder.glob(filename_glob)):
        try:
            diag = pd.read_csv(path, dtype={CASE_ID_NAME: str})
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if CASE_ID_NAME in diag.columns:
            ids.update(diag[CASE_ID_NAME].astype(str).tolist())
    return ids


def run_dataset_validation(case_study: str, method: str, rank: int, is_baseline: bool,
                            test_log: pd.DataFrame, sim_dfs: list, sim_folder: Path,
                            recommendations: dict, rec_path, n_sim: int) -> bool:
    """Check, for EVERY case in the recommendation file, whether the recommended
    (activity / resource) pair is the first post-prefix event in every simulation.

    Prints a dataset-level pass/fail summary to stdout (no files written).
    Returns True when every case that is expected to follow its recommendation
    does so in all n_sim runs, False otherwise.
    """
    print("=" * 100)
    print(f"DATASET VALIDATION: {case_study} / {method}" + ("" if is_baseline else f" / rank {rank}"))
    print("=" * 100)

    if is_baseline or not recommendations:
        print("Validation only applies to 'exhaustive'/'nsga2' with a recommendation file. Nothing to check.")
        return True

    excluded_ids = load_diagnostic_case_ids(sim_folder, "excluded_no_recommendation.csv")
    unreachable_ids = load_diagnostic_case_ids(sim_folder, "sim_*_unreachable_recommendations.csv")

    prefix_groups = group_by_case(test_log)
    sim_groups = [group_by_case(sdf) for sdf in sim_dfs]

    rows = []
    for cid, rec in recommendations.items():
        rec_act, rec_res = rec.get("act"), rec.get("res")

        if rec_act is None:
            rows.append({"case:concept:name": cid, "classification": "no_recommendation_at_rank",
                         "rec_activity": "", "rec_resource": "", "sims_followed": "", "sims_total": ""})
            continue
        if cid in excluded_ids:
            rows.append({"case:concept:name": cid, "classification": "excluded_no_recommendation",
                         "rec_activity": rec_act, "rec_resource": rec_res, "sims_followed": "", "sims_total": ""})
            continue
        if cid in unreachable_ids:
            rows.append({"case:concept:name": cid, "classification": "unreachable_from_prefix",
                         "rec_activity": rec_act, "rec_resource": rec_res, "sims_followed": "", "sims_total": ""})
            continue

        prefix_seq = prefix_groups.get(cid)
        if prefix_seq is None or prefix_seq.empty:
            rows.append({"case:concept:name": cid, "classification": "missing_from_test_log",
                         "rec_activity": rec_act, "rec_resource": rec_res, "sims_followed": "", "sims_total": ""})
            continue

        prefix_end_ts = prefix_seq[END_DATE_NAME].max()
        followed = 0
        for sg in sim_groups:
            sim_seq = sg.get(cid, prefix_seq.iloc[0:0])
            sim_cont, _ = extract_continuation_after_prefix(sim_seq, prefix_seq, prefix_end_ts)
            if batch_follows_recommendation(sim_cont, rec_act, rec_res):
                followed += 1

        if followed == n_sim:
            classification = "followed_all_sims"
        elif followed > 0:
            classification = "followed_some_sims"
        else:
            classification = "followed_no_sims"
        rows.append({"case:concept:name": cid, "classification": classification,
                     "rec_activity": rec_act, "rec_resource": rec_res,
                     "sims_followed": followed, "sims_total": n_sim})

    report = pd.DataFrame(rows)
    counts = report["classification"].value_counts().to_dict()
    total = len(report)
    n_no_rec = counts.get("no_recommendation_at_rank", 0)
    n_excluded = counts.get("excluded_no_recommendation", 0)
    n_unreachable = counts.get("unreachable_from_prefix", 0)
    n_missing = counts.get("missing_from_test_log", 0)
    n_all = counts.get("followed_all_sims", 0)
    n_some = counts.get("followed_some_sims", 0)
    n_none = counts.get("followed_no_sims", 0)
    n_expected = n_all + n_some + n_none

    print(f"\nRecommendation file : {rec_path.name}  ({total} cases)")
    print(f"Simulations         : {n_sim} run(s) from {sim_folder}")
    print("\nLegitimately not expected to follow the recommendation:")
    print(f"  no recommendation at this rank (empty) : {n_no_rec}")
    print(f"  excluded_no_recommendation.csv         : {n_excluded}")
    print(f"  unreachable from replayed prefix       : {n_unreachable}")
    if n_missing:
        print(f"  case id missing from test_log.csv      : {n_missing}  [!] unexpected")

    print(f"\nExpected to follow the recommendation    : {n_expected}")
    if n_expected:
        print(f"  followed in ALL {n_sim} sims : {n_all:6d}  ({n_all / n_expected * 100:.1f}%)")
        print(f"  followed in SOME sims  : {n_some:6d}  ({n_some / n_expected * 100:.1f}%)")
        print(f"  followed in NO sims    : {n_none:6d}  ({n_none / n_expected * 100:.1f}%)")

    offenders = report[report["classification"].isin(["followed_some_sims", "followed_no_sims"])]
    if len(offenders) or n_missing:
        print(f"\n>>> {len(offenders) + n_missing} case(s) do NOT fully follow the recommendation:")
        for _, r in offenders.head(50).iterrows():
            print(f"  {r['case:concept:name']}  followed {r['sims_followed']}/{r['sims_total']}  "
                  f"rec='{r['rec_activity']} / {r['rec_resource']}'")
        if len(offenders) > 50:
            print(f"  ... and {len(offenders) - 50} more")

    ok = (n_some == 0 and n_none == 0 and n_missing == 0)
    print("\n" + ("RESULT: OK -- every expected case follows its recommendation in all sims."
                   if ok else
                   f"RESULT: FAIL -- {n_some + n_none + n_missing} case(s) not fully adherent (see above)."))
    return ok


def build_comparison_table(prefix_len: int, real_cont: pd.DataFrame,
                            sim_conts: list, max_steps: int, method_label: str,
                            rec_act=None, rec_res=None) -> pd.DataFrame:
    n_steps = max(len(real_cont), *(len(s) for s in sim_conts)) if sim_conts else len(real_cont)
    n_steps = min(n_steps, max_steps)

    rec_str = f"{rec_act} / {rec_res}" if rec_act is not None else ""

    rows = []
    for step in range(n_steps):
        row = {"step_after_prefix": step + 1}
        # The recommendation only concerns the first post-prefix step.
        row["RECOMMENDED"] = rec_str if step == 0 else ""
        row["REAL"] = format_step(real_cont.iloc[step]) if step < len(real_cont) else ""
        for i, sim_cont in enumerate(sim_conts):
            row[f"{method_label}_SIM_{i + 1}"] = format_step(sim_cont.iloc[step]) if step < len(sim_cont) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare, step by step, prefix + real continuation vs "
                    "method-simulated continuations for sampled case ids, and "
                    "check whether the recommendation for each case is followed."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Directory that contains case_studies/ (run from multi_obj/).")
    parser.add_argument("--case_study", type=str, required=True)
    parser.add_argument("--method", type=str, default="nsga2",
                         choices=["baseline", "exhaustive", "nsga2"],
                         help="Which simulation folder to compare against the real "
                              "continuation: 'baseline' (no recommendation), "
                              "'exhaustive', or 'nsga2' (default: nsga2).")
    parser.add_argument("--rank", type=int, default=1,
                         help="Pareto rank of the recommendation to inspect (top<rank>of<k>). "
                              "Defaults to 1; ignored for --method baseline.")
    parser.add_argument("--n_cases", type=int, default=5,
                         help="Number of random case ids to inspect (ignored if --case_ids given).")
    parser.add_argument("--case_ids", type=str, default=None,
                         help="Comma-separated list of specific case ids to inspect, instead of random sampling.")
    parser.add_argument("--n_sim", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=8,
                         help="Max number of post-prefix steps to display per case.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="inspection_output")
    parser.add_argument("--validate", action="store_true",
                         help="Skip the per-case inspection. Instead check EVERY case in the "
                              "recommendation file and print one dataset-level pass/fail summary: "
                              "how many cases follow the recommendation as the first post-prefix "
                              "event in all simulations, and which ones don't. Exit code 1 on FAIL.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    case_dir = base_dir / "case_studies" / args.case_study
    out_dir = Path(args.out_dir)
    if not args.validate:
        out_dir.mkdir(parents=True, exist_ok=True)

    is_baseline = args.method == BASELINE_FOLDER_NAME
    rank = args.rank

    print("Loading test_log.csv (prefixes) and test_data.csv (real continuations)...")
    test_log = load_events(case_dir / "test_log.csv")
    test_data = load_events(case_dir / "test_data.csv")

    sim_folder = resolve_sim_folder(case_dir, args.method, rank)
    if is_baseline:
        print(f"Loading {args.n_sim} 'baseline' simulation file(s) from {sim_folder}...")
        print("NOTE: 'baseline' is the no-recommendation simulation; use "
              "'--method exhaustive' or '--method nsga2' to inspect recommendation-aware behavior.")
    else:
        print(f"Loading {args.n_sim} '{args.method}' rank-{rank} simulation file(s) from {sim_folder}...")
    if not sim_folder.exists():
        raise FileNotFoundError(f"Simulation folder not found: {sim_folder}")
    sim_dfs = [load_events(sim_folder / f"sim_{i + 1}.csv") for i in range(args.n_sim)]

    # --- recommendations for this (method, rank) ---
    recommendations: dict = {}
    rec_path = None
    if not is_baseline:
        rec_path = resolve_recommendation_file(case_dir, args.case_study, args.method, rank)
        if rec_path is None:
            print(f"\n[WARN] No recommendation file "
                  f"recommendations_{args.case_study}_{args.method}_top{rank}of*.csv found; "
                  "recommendation checks will be skipped.")
        else:
            print(f"Loading recommendations from {rec_path.name}...")
            recommendations = load_recommendations(rec_path)

    # --- dataset-wide validation mode: one pass/fail summary, no per-case dump, no files ---
    if args.validate:
        ok = run_dataset_validation(
            args.case_study, args.method, rank, is_baseline,
            test_log, sim_dfs, sim_folder, recommendations, rec_path, args.n_sim,
        )
        sys.exit(0 if ok else 1)

    if args.case_ids:
        sampled_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        all_ids = test_log[CASE_ID_NAME].unique().tolist()
        random.seed(args.seed)
        sampled_ids = random.sample(all_ids, min(args.n_cases, len(all_ids)))

    # --- aggregate recommendation-adherence report (across all n_sim runs) ---
    if recommendations:
        sampled_recs = {cid: recommendations[cid] for cid in sampled_ids if cid in recommendations}
        if sampled_recs:
            check_df = check_recommendation_following(test_log, sampled_recs, sim_dfs)
            check_out = out_dir / f"{args.case_study}_{args.method}_rank{rank}_recommendation_check.csv"
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
            print("  [SKIPPED] case id not found in test_log.csv")
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

        # --- recommendation for this case ---
        rec = recommendations.get(case_id)
        rec_act = rec_res = None
        if rec is not None:
            rec_act, rec_res = rec.get("act"), rec.get("res")
            if rec_act is None:
                print(f"\n  RECOMMENDED next (rank {rank}): (nessuna raccomandazione a questo rank per il caso)")
            else:
                print(f"\n  RECOMMENDED next (rank {rank}): {rec_act} / {rec_res}")
        elif not is_baseline and rec_path is not None:
            print(f"\n  RECOMMENDED next (rank {rank}): (case id not present in {rec_path.name})")

        # --- prefix summary (last few events for context) ---
        print(f"\n  Prefix length: {prefix_len} event(s). Last event(s) of the prefix:")
        tail = prefix_seq.tail(3)[[ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME, END_DATE_NAME]]
        for _, r in tail.iterrows():
            print(f"    ... {r[ACTIVITY_COLUMN_NAME]} / {r[RESOURCE_COLUMN_NAME]} @ {r[END_DATE_NAME]}")

        if real_cont.empty:
            print("\n  [NOTE] No real continuation found for this case in test_data.csv "
                  "(prefix might already be the whole trace).")

        if real_pre_boundary:
            print(f"\n  [NOTE] Found {real_pre_boundary} real event(s) ending before prefix boundary "
                  "after prefix-removal; check data consistency.")

        empty_sims = sum(1 for s in sim_conts if s.empty)
        if empty_sims:
            print(f"\n  [NOTE] {empty_sims}/{args.n_sim} '{args.method}' simulation(s) have no "
                  "post-prefix continuation after boundary filtering.")

        sims_with_temporal_anomaly = sum(1 for c in sim_pre_boundary_counts if c > 0)
        if sims_with_temporal_anomaly:
            print(
                f"\n  [NOTE] {sims_with_temporal_anomaly}/{args.n_sim} '{args.method}' simulation(s) generated "
                "event(s) ending before the prefix boundary (temporal anomaly)."
            )

        # --- quick check of the first simulated activity ---
        print("\n  [QUICK CHECK] Prima attività simulata (Step 1 by earliest completion):")
        for i, sim_cont in enumerate(sim_conts):
            if not sim_cont.empty:
                first_act = sim_cont.iloc[0][ACTIVITY_COLUMN_NAME]
                first_res = sim_cont.iloc[0][RESOURCE_COLUMN_NAME]
                print(f"    SIM_{i + 1}: {first_act} / {first_res}")
            else:
                print(f"    SIM_{i + 1}: (Nessuna continuazione)")

        print("\n  [QUICK CHECK] Eventi con start minimo post-prefix (tie-aware):")
        for i, sim_cont in enumerate(sim_conts):
            print(f"    SIM_{i + 1}: {format_first_start_batch(sim_cont)}")

        # --- recommendation-adherence per simulation for this case ---
        if rec_act is not None and not (isinstance(rec_act, float) and pd.isna(rec_act)):
            print(f"\n  [REC CHECK] La raccomandazione '{rec_act} / {rec_res}' compare nel primo batch post-prefix?")
            followed = 0
            checked = 0
            for i, sim_cont in enumerate(sim_conts):
                res = batch_follows_recommendation(sim_cont, rec_act, rec_res)
                if res is None:
                    continue
                checked += 1
                followed += int(res)
                print(f"    SIM_{i + 1}: {'SI' if res else 'NO'}")
            if checked:
                print(f"    -> Rispettata in {followed}/{checked} simulazioni "
                      f"({followed / checked * 100:.1f}%)")

        # --- step-by-step comparison table ---
        method_label = args.method.upper()
        table = build_comparison_table(prefix_len, real_cont, sim_conts, args.max_steps,
                                       method_label, rec_act, rec_res)

        suffix = "baseline" if is_baseline else f"{args.method}_rank{rank}"
        out_path = out_dir / f"{args.case_study}_{suffix}_{case_id}_comparison.csv"
        table.to_csv(out_path, index=False)
        print(f"\n  Saved to {out_path}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
