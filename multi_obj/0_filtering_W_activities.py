#!/usr/bin/env python3
"""
Pre-processing helper: remove every W_ activity (work items) from the case-study logs with fake start
times, keeping only the A_ (application) and O_ (offer) activities.

What it does (for each case study, under case_studies/<case_study>/)
----------------------------------------------------------------------
  * input:  log_<case_study>_no_start.xes             -- never overwritten
  * output: log_<case_study>_no_start_W_filtered.xes  -- same events minus the W_ ones

Only rows are removed: every attribute of the remaining events (start:timestamp included, so a fake
start may be the completion of a removed W_ event), the event order and the case set are unchanged
(no case starts with, or consists only of, W_ events).

Usage
-----
    python 0_filtering_W_activities.py
    python 0_filtering_W_activities.py --case_studies BPI12
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
import pm4py

warnings.filterwarnings("ignore")

CASE_ID_NAME = "case:concept:name"
ACTIVITY_COLUMN_NAME = "concept:name"
START_DATE_NAME = "start:timestamp"
END_DATE_NAME = "time:timestamp"

DEFAULT_CASE_STUDIES = ["BPI12", "bpi17_before", "bpi17_after"]
W_PREFIX = "W_"


def parse_args():
    parser = argparse.ArgumentParser(description="Remove the W_ activities from the case-study logs with fake start times.")
    parser.add_argument("--base_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--case_studies", default=",".join(DEFAULT_CASE_STUDIES),
                        help="Comma-separated case studies to process.")
    parser.add_argument("--input_suffix", default="_no_start",
                        help="Input log: case_studies/<cs>/log_<cs><input_suffix>.xes.")
    parser.add_argument("--output_suffix", default="_no_start_W_filtered",
                        help="Output log: case_studies/<cs>/log_<cs><output_suffix>.xes. Must differ from the input.")
    return parser.parse_args()


def process_case_study(base_dir: Path, cs: str, args) -> None:
    if args.input_suffix == args.output_suffix:
        raise ValueError("--input_suffix and --output_suffix must differ: the input log must never be overwritten")
    case_dir = base_dir / "case_studies" / cs
    in_path = case_dir / f"log_{cs}{args.input_suffix}.xes"
    out_path = case_dir / f"log_{cs}{args.output_suffix}.xes"
    print(f"\n=== {cs}: reading {in_path.name}")

    raw = pd.DataFrame(pm4py.read_xes(str(in_path), show_progress_bar=False)).reset_index(drop=True)
    original_columns = list(raw.columns)
    is_w = raw[ACTIVITY_COLUMN_NAME].astype(str).str.startswith(W_PREFIX)
    out = raw[~is_w].reset_index(drop=True)

    lost_cases = set(raw[CASE_ID_NAME]) - set(out[CASE_ID_NAME])
    if lost_cases:
        raise ValueError(f"{len(lost_cases)} cases consist only of W_ activities, e.g. {sorted(lost_cases)[:5]}")

    formatted = pm4py.format_dataframe(out, case_id=CASE_ID_NAME, activity_key=ACTIVITY_COLUMN_NAME,
                                       timestamp_key=END_DATE_NAME, start_timestamp_key=START_DATE_NAME)
    # format_dataframe adds helper columns (@@index, @@case_index, start_timestamp): keep only the original ones.
    pm4py.write_xes(formatted[original_columns], str(out_path))

    prefixes = sorted(out[ACTIVITY_COLUMN_NAME].astype(str).str.split("_").str[0].unique())
    print(f"  removed {is_w.sum()} W_ events ({raw.loc[is_w, ACTIVITY_COLUMN_NAME].nunique()} activities): "
          f"{len(raw)} -> {len(out)} events, {out[CASE_ID_NAME].nunique()} cases, "
          f"{out[ACTIVITY_COLUMN_NAME].nunique()} activities (prefixes {prefixes})")
    print(f"  wrote {out_path}")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        process_case_study(base_dir, cs, args)


if __name__ == "__main__":
    main()
