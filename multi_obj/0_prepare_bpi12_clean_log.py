#!/usr/bin/env python3
"""
Rebuilds case_studies/BPI12_reordered/log_BPI12_reordered.xes directly from the
raw "clean" BPI12 log (BPI_2012_log_eng_clean.xes), matching log_BPI12.xes's
OWN timestamp convention exactly -- deliberately, not "fixing" it:

  - Only COMPLETE rows are kept as events (SCHEDULE and START rows are
    dropped entirely, not used for anything -- not even to recover a real
    start:timestamp).
  - 'time:timestamp' = that COMPLETE row's own real completion instant.
  - 'start:timestamp' = the PREVIOUS event's time:timestamp within the same
    case (chained), and for the first event of a case, start:timestamp =
    time:timestamp (zero duration -- verified this is exactly how
    log_BPI12.xes's own first event of every trace behaves).

Why replicate this instead of using pm4py's interval_lifecycle.to_interval()
(the previous version of this script): the goal here isn't to recover the
"true" start/duration of each activity -- it's to get the EVENT ORDER right
(using the clean log's own real, granular COMPLETE timestamps, which do not
suffer from log_BPI12.xes's same-instant ties) while keeping the exact same
column/timestamp SHAPE as log_BPI12.xes, so every downstream script that
already works on that shape keeps working unchanged. Sorting by the true
completion instant (stable sort, so ties fall back to the raw file's own
row order) directly gives the corrected event order -- no separate tie
detection/permutation step needed (superseding discovery/reorder_bpi12_
ties_from_clean_log.py's approach, which only patched order within ties
already visible in the OLD log's own coarser timestamps).

Only the cases already used by case_studies/BPI12/log_BPI12.xes (4685 case
IDs) are kept, so every downstream script (discovery, training, 5/6/8b) sees
the exact same case population as the rest of this investigation.

The raw source file itself is left untouched.

Column handling (same as the previous version of this script):
  - 'case:amount' -> renamed to a bare 'AMOUNT_REQ' column (matches
    log_BPI12.xes's own layout; utils/get_features.py's BPI12 config
    references this bare name).
  - 'case:REG_DATE' -> dropped (a raw per-case timestamp attribute that,
    left in, gets picked up by ProSiT as a routing/timing feature and
    crashes sklearn -- hit and fixed once already this session).
  - '@@...' bookkeeping columns and 'lifecycle:transition' -> dropped.

Usage:
    python 0_prepare_bpi12_clean_log.py
"""
import warnings
from pathlib import Path

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

warnings.filterwarnings("ignore")

RAW_LOG = Path("case_studies/BPI12_reordered/other/clean_BPI12/BPI_2012_log_eng_clean.xes")
ORIGINAL_LOG = Path("case_studies/BPI12/log_BPI12.xes")
OUT_XES = Path("case_studies/BPI12_reordered/log_BPI12_reordered.xes")


def main():
    print(f"Reading raw lifecycle log: {RAW_LOG}")
    log = xes_importer.apply(str(RAW_LOG))
    df = pm4py.convert_to_dataframe(log)

    print(f"Reading original log for the case-ID filter: {ORIGINAL_LOG}")
    df_orig = pm4py.convert_to_dataframe(pm4py.read_xes(str(ORIGINAL_LOG)))
    keep_ids = set(df_orig["case:concept:name"].astype(str).unique())
    print(f"Original log has {len(keep_ids)} case IDs.")

    df["case:concept:name"] = df["case:concept:name"].astype(str)
    n_before = df["case:concept:name"].nunique()
    df = df[df["case:concept:name"].isin(keep_ids)].copy()
    missing = keep_ids - set(df["case:concept:name"].unique())
    if missing:
        print(f"WARNING: {len(missing)} original case IDs not found in the clean log: "
              f"{sorted(missing)[:10]}...")
    print(f"Filtered {n_before} -> {df['case:concept:name'].nunique()} cases.")

    is_complete = df["lifecycle:transition"].astype(str).str.upper() == "COMPLETE"
    print(f"Keeping COMPLETE rows only: {is_complete.sum()} / {len(df)} rows "
          f"(dropping SCHEDULE/START rows entirely).")
    df = df[is_complete].copy()
    df = df.sort_values(["case:concept:name", "time:timestamp"], kind="stable").reset_index(drop=True)

    # start:timestamp = previous event's time:timestamp within the same case; the
    # first event of each case gets its own time:timestamp (zero duration)
    df["start:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].shift(1)
    df["start:timestamp"] = df["start:timestamp"].fillna(df["time:timestamp"])

    drop_cols = [c for c in df.columns if c.startswith("@@") or c == "lifecycle:transition"]
    df = df.drop(columns=drop_cols, errors="ignore")

    if "case:amount" in df.columns:
        df["AMOUNT_REQ"] = df["case:amount"]
        df = df.drop(columns=["case:amount"])

    df = df.drop(columns=["case:REG_DATE"], errors="ignore")

    # A handful of W_ activities' COMPLETE rows have no org:resource in the raw log
    # (5.37% of rows, all W_ -- e.g. system-closed work items with no human assigned).
    # ProSiT's resource discovery builds a "resource = <name>" column per resource
    # (prosit/discovery/time_discovery.py's build_training_df_ex) and crashes trying to
    # concatenate a string with a bare NaN float; give missing resources their own
    # explicit category instead of dropping those events.
    n_missing_resource = df["org:resource"].isna().sum()
    if n_missing_resource:
        print(f"Filling {n_missing_resource} rows with missing org:resource as 'UNKNOWN'.")
        df["org:resource"] = df["org:resource"].fillna("UNKNOWN")

    ordered = ["case:concept:name", "concept:name", "start:timestamp", "time:timestamp", "org:resource"]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    n_cases = df["case:concept:name"].nunique()
    n_events = len(df)
    print(f"[BPI12_reordered] cases={n_cases}  events={n_events}  events/case={n_events / n_cases:.2f}  "
          f"activities={df['concept:name'].nunique()}  columns={list(df.columns)}")

    OUT_XES.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(df, str(OUT_XES),
                     case_id_key="case:concept:name", activity_key="concept:name",
                     timestamp_key="time:timestamp")
    print(f"Wrote {OUT_XES}")


if __name__ == "__main__":
    main()

# Running commands:
# python 0_prepare_bpi12_clean_log.py
