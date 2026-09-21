#!/usr/bin/env python3
"""
Creates a copy of case_studies/BPI12/log_BPI12.xes ("BPI12_reordered") where
groups of events that complete at the EXACT same instant (time:timestamp,
tied with at least one other event in the same case) are reordered to match
the relative order those same activities have in the new
"clean" log (case_studies/BPI12_clean_filtered/log_BPI12_clean_filtered.xes,
built by 0_prepare_bpi12_clean_log.py from the same 4685 cases).

Why this can matter for the whole duplicate-task investigation: if the old
log's tie-breaking was inconsistent across cases (e.g. sometimes listing
O_SELECTED before O_CANCELLED, sometimes after, for what is really always
the same causal order), Split Miner's directly-follows frequencies for that
pair would look artificially balanced -- exactly the concurrency-pruning
trigger documented in report_duplicate_task_gaps.tex/presentation
(r = |freq(a->b)-freq(b->a)| / (freq(a->b)+freq(b->a)) < epsilon). Fixing
ties to a consistent, source-backed order could remove that artifact for
some pairs without touching anything else about the log.

Scope, deliberately narrow:
  - Only the 17 activity labels shared VERBATIM by both logs are touched
    (all A_/O_ instantaneous milestone events -- see the vocabulary check
    this script runs and prints). The 6 W_ activities are NOT touched: the
    clean log renames AND re-splits them (e.g. old's single "W_Completeren
    aanvraag" corresponds to new's more granular "W_Complete_preaccepted_
    appl" occurrences with genuinely different, non-chained start times --
    a bigger, separate question from event ORDER, out of scope here).
  - Only rows that are part of a same-completion-instant tie (time:timestamp
    shared with at least one other row in the same case) are ever reordered.
    Ties are NOT required to also share start:timestamp: the old log's start
    is sometimes corrupted (chained to the previous event's end instead of a
    real value -- e.g. A_REGISTERED in case 173688 -- which would otherwise
    hide a genuine tie). Every other row -- different completion instant,
    any W_ row, any row in a case this script can't cleanly match -- is left
    byte-for-byte as in the original (start:timestamp included: this script
    never rewrites timestamps, only reorders which row gets which label).
  - A tie group is only reordered when the new log's shared-vocabulary
    subsequence has, at the SAME ordinal position, the exact same multiset
    of activity labels as the old tie group. Every case is verified for a
    matching total count of shared-vocabulary events before touching it;
    a mismatch excludes the WHOLE case from reordering rather than guessing.

Usage:
    python discovery/reorder_bpi12_ties_from_clean_log.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pm4py

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OLD_LOG = Path("case_studies/BPI12/log_BPI12.xes")
NEW_LOG = Path("case_studies/BPI12_clean_filtered/log_BPI12_clean_filtered.xes")
OUT_DIR = Path("case_studies/BPI12_reordered")
OUT_XES = OUT_DIR / "log_BPI12_reordered.xes"


def load(path: Path) -> pd.DataFrame:
    df = pm4py.convert_to_dataframe(pm4py.read_xes(str(path)))
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


def main():
    print(f"Loading {OLD_LOG} ...")
    df_old = load(OLD_LOG)
    print(f"Loading {NEW_LOG} ...")
    df_new = load(NEW_LOG)

    old_acts = set(df_old["concept:name"].unique())
    new_acts = set(df_new["concept:name"].unique())
    shared = old_acts & new_acts
    print(f"Shared activity labels ({len(shared)}): {sorted(shared)}")
    print(f"Old-only (never touched): {sorted(old_acts - new_acts)}")
    print(f"New-only (irrelevant here): {sorted(new_acts - old_acts)}\n")

    # Sort by completion instant, not start: old log's start:timestamp is sometimes
    # corrupted (see module docstring), so it's not a reliable ordering key.
    df_old = df_old.sort_values(["case:concept:name", "time:timestamp"], kind="stable").reset_index(drop=True)
    df_new = df_new.sort_values(["case:concept:name", "time:timestamp"], kind="stable").reset_index(drop=True)

    df_old["_global_row"] = df_old.index
    # Tie = same COMPLETION instant (time:timestamp) within a case. NOT start==end:
    # a first pass required zero duration (start==end) and missed cases where the old
    # log's start:timestamp is corrupted (chained to the previous event's end instead
    # of a real value, e.g. A_REGISTERED in case 173688: end matches the tied cluster
    # exactly, but its start was borrowed from the prior event, giving it a fake
    # multi-day "duration" that hid the tie). What matters for directly-follows order
    # is when the event is recorded as having happened -- time:timestamp -- not start.
    df_old["_tie_size"] = df_old.groupby(
        ["case:concept:name", "time:timestamp"]
    )["concept:name"].transform("size")
    df_old["_is_tied"] = df_old["_tie_size"] > 1

    new_concept = df_old["concept:name"].astype(object).to_numpy().copy()

    n_cases_total = 0
    n_cases_count_mismatch = 0
    n_groups_total = 0
    n_groups_reordered = 0
    n_groups_skipped_vocab = 0
    n_groups_skipped_mismatch = 0
    n_rows_changed = 0

    old_by_case = df_old.groupby("case:concept:name", sort=False)
    new_by_case = {cid: g for cid, g in df_new.groupby("case:concept:name", sort=False)}

    for case_id, old_g in old_by_case:
        n_cases_total += 1
        new_g = new_by_case.get(case_id)
        if new_g is None:
            continue

        old_shared_mask = old_g["concept:name"].isin(shared)
        new_shared_mask = new_g["concept:name"].isin(shared)
        old_shared = old_g[old_shared_mask]
        new_shared_labels = new_g.loc[new_shared_mask, "concept:name"].tolist()

        if len(old_shared) != len(new_shared_labels):
            n_cases_count_mismatch += 1
            continue
        if sorted(old_shared["concept:name"]) != sorted(new_shared_labels):
            n_cases_count_mismatch += 1
            continue

        # Ordinal position of each old-shared row within this case's shared subsequence.
        old_shared_positions = {row_pos: ordinal for ordinal, row_pos in enumerate(old_shared.index)}

        # Walk tie groups in original (file) order.
        for ts_end, grp in old_g[old_g["_is_tied"]].groupby(
            "time:timestamp", sort=False
        ):
            n_groups_total += 1
            if not set(grp["concept:name"]).issubset(shared):
                n_groups_skipped_vocab += 1
                continue

            ordinals = [old_shared_positions[i] for i in grp.index]
            lo, hi = min(ordinals), max(ordinals) + 1
            if hi - lo != len(grp):
                # Not contiguous in the shared subsequence (shouldn't happen given the
                # tie-group is contiguous in the full trace too) -- skip defensively.
                n_groups_skipped_mismatch += 1
                continue

            new_window = new_shared_labels[lo:hi]
            if sorted(new_window) != sorted(grp["concept:name"]):
                n_groups_skipped_mismatch += 1
                continue

            if new_window == list(grp["concept:name"]):
                n_groups_reordered += 1  # already matches, nothing to change
                continue

            # Assign the new log's relative order to this tie group's global rows,
            # keeping every other column (timestamps, resource, ...) attached to its
            # original row position -- only concept:name is permuted.
            global_rows = grp["_global_row"].tolist()
            for row_idx, label in zip(global_rows, new_window):
                if new_concept[row_idx] != label:
                    n_rows_changed += 1
                new_concept[row_idx] = label
            n_groups_reordered += 1

    print(f"Cases: {n_cases_total} total, {n_cases_count_mismatch} excluded "
          f"(shared-vocabulary count/multiset mismatch with the clean log).")
    print(f"Tie groups: {n_groups_total} total")
    print(f"  - {n_groups_reordered} resolved (order taken from the clean log; "
          f"{n_rows_changed} rows actually changed label position)")
    print(f"  - {n_groups_skipped_vocab} skipped (involve a non-shared, i.e. W_, activity)")
    print(f"  - {n_groups_skipped_mismatch} skipped (couldn't be matched cleanly)")

    df_out = df_old.drop(columns=["_global_row", "_tie_size", "_is_tied"]).copy()
    df_out["concept:name"] = new_concept

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(df_out, str(OUT_XES),
                     case_id_key="case:concept:name", activity_key="concept:name",
                     timestamp_key="time:timestamp")
    print(f"\nWrote {OUT_XES}")


if __name__ == "__main__":
    main()

# Running commands:
# python discovery/reorder_bpi12_ties_from_clean_log.py
