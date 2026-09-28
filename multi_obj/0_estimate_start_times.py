#!/usr/bin/env python3
"""
Pre-processing helper: re-estimate the activity start times of a case-study event log.

!! Run it from the dedicated conda env `pix_env` (pix-framework needs pandas<3 and
!! scikit-learn<1.6, incompatible with fbk_thesis_env), with the user site-packages
!! disabled so the shared AppData\\Roaming\\Python packages are not picked up:
!!     C:\\Users\\Utente\\.conda\\envs\\pix_env\\python.exe -s 0_estimate_start_times.py

Motivation
----------
In BPI12_reordered, bpi17_before and bpi17_after the recorded ``start:timestamp`` of
every event is not a real start: it is a copy of the ``time:timestamp`` (completion)
of the previous event of the same case (and the first event of each case is
instantaneous). So every activity "starts" the instant its predecessor ends, the
waiting time between two activities is always 0, and the whole gap ends up in the
execution time. (BAC records real start times -- about 25% of its events start
after their predecessor ends -- so it is not a target of this script.)

The start times are repaired with the technique of Chapela-Campa & Dumas,
"Repairing Activity Start Times to Improve Business Process Simulation", calling
pix-framework's implementation directly:
https://github.com/AutomatedProcessImprovement/pix-framework/tree/main/src/pix_framework/enhancement/start_time_estimator

    estimated_start(e) = max( enabled_time(e), available_time(e) )

  * enabled_time(e)   -- end time of the causal predecessor of e: the latest event
                         of the same case that ended before e ended and whose
                         activity is not concurrent with e's activity (concurrency
                         relations come from the configured oracle). First event of
                         a case: the case start.
  * available_time(e) -- end time of the latest event executed by the same
                         resource that ended before e ended (the resource works on
                         one activity at a time and starts the next one as soon as
                         it is free).

The fake start times are passed to the estimator unchanged (not reset): with
consider_start_times=False the estimator ignores them, except that the case start
(= enabled time of the first event) is min(start, end) over the case, i.e. the
first completion. So the first event of each case stays instantaneous and the case
arrival times -- and the cycle-time labels -- do not change.

What it produces (under case_studies/<case_study>/)
---------------------------------------------------
  * input:  log_<case_study>_no_start.xes -- the log with the fake start times
                                             (never overwritten)
  * output: log_<case_study>.xes          -- the same log with start:timestamp
                                             replaced by the estimate (every other
                                             attribute untouched); this is the file
                                             the rest of the pipeline reads

Default configuration (see DEFAULT_* below for the sources)
-----------------------------------------------------------
  * concurrency oracle: Heuristics Miner, thresholds 0.9 (the paper's proposal, sec. III.ii)
  * system accounts (start = end): 112 (BPI12_reordered), User_1 (bpi17_before/after)
  * unknown resource (start = enabled time): "0" (BPI12_reordered)
  * instantaneous activities (start = end): every A_ and O_ activity
  * outlier threshold: off

Usage C:\\Users\\Utente\\.conda\\envs\\pix_env\\python.exe -s 0_estimate_start_times.py
-----
    python -s 0_estimate_start_times.py
    python -s 0_estimate_start_times.py --case_studies bpi17_before
    python -s 0_estimate_start_times.py --concurrency df --output_suffix _est_df   # writes log_<cs>_est_df.xes
    python -s 0_estimate_start_times.py --instant_activity_prefixes "" --bot_resources "" --missing_resource ""   # package defaults
"""
import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
import pm4py
from pix_framework.enhancement.concurrency_oracle import (
    AlphaConcurrencyOracle,
    DirectlyFollowsConcurrencyOracle,
    HeuristicsConcurrencyOracle,
)
from pix_framework.enhancement.start_time_estimator.config import (
    ConcurrencyOracleType,
    ConcurrencyThresholds,
    Configuration,
    OutlierStatistic,
    ReEstimationMethod,
    ResourceAvailabilityType,
)
from pix_framework.enhancement.start_time_estimator.estimator import StartTimeEstimator
from pix_framework.io.event_log import EventLogIDs

warnings.filterwarnings("ignore")

CASE_ID_NAME = "case:concept:name"
ACTIVITY_COLUMN_NAME = "concept:name"
RESOURCE_COLUMN_NAME = "org:resource"
START_DATE_NAME = "start:timestamp"
END_DATE_NAME = "time:timestamp"

DEFAULT_CASE_STUDIES = ["BPI12_reordered", "bpi17_before", "bpi17_after"]
DEFAULT_BOT_RESOURCES = "BPI12_reordered=112;bpi17_before=User_1;bpi17_after=User_1"
DEFAULT_MISSING_RESOURCE = "BPI12_reordered=0"
DEFAULT_INSTANT_ACTIVITY_PREFIXES = "A_,O_"

LOG_IDS = EventLogIDs(
    case=CASE_ID_NAME,
    activity=ACTIVITY_COLUMN_NAME,
    resource=RESOURCE_COLUMN_NAME,
    start_time=START_DATE_NAME,
    end_time=END_DATE_NAME,
    enabled_time="enabled_time",
    enabling_activity="enabling_activity",
    available_time="available_time",
    estimated_start_time="estimated_start",
)

CONCURRENCY_TYPES = {
    "heuristics": ConcurrencyOracleType.HEURISTICS,
    "alpha": ConcurrencyOracleType.ALPHA,
    "df": ConcurrencyOracleType.DF,
    "deactivated": ConcurrencyOracleType.DEACTIVATED,
}
RE_ESTIMATION_METHODS = {
    "median": ReEstimationMethod.MEDIAN,
    "mean": ReEstimationMethod.MEAN,
    "mode": ReEstimationMethod.MODE,
    "instant": ReEstimationMethod.SET_INSTANT,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate activity start times (Chapela-Campa & Dumas, pix-framework) for case-study event logs.")
    parser.add_argument("--base_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--case_studies", default=",".join(DEFAULT_CASE_STUDIES),
                        help="Comma-separated case studies to process.")
    parser.add_argument("--input_suffix", default="_no_start",
                        help="The input log is case_studies/<cs>/log_<cs><input_suffix>.xes (the log with the fake "
                             "start times).")
    parser.add_argument("--output_suffix", default="",
                        help="The output log is case_studies/<cs>/log_<cs><output_suffix>.xes. Default: the canonical "
                             "log_<cs>.xes read by the rest of the pipeline. Must differ from the input.")
    parser.add_argument("--concurrency", default="heuristics", choices=list(CONCURRENCY_TYPES),
                        help="Concurrency oracle used to find each event's causal predecessor. 'df' = no concurrency "
                             "(predecessor = previous event of the case); 'deactivated' = ignore enablement, use "
                             "resource availability only. ('overlapping' is not offered: it relies on the recorded "
                             "start times, which here are fake.)")
    parser.add_argument("--df_threshold", type=float, default=0.9, help="Heuristics oracle: |dependency| threshold.")
    parser.add_argument("--l2l_threshold", type=float, default=0.9, help="Heuristics oracle: length-2-loop threshold.")
    parser.add_argument("--l1l_threshold", type=float, default=0.9, help="Heuristics oracle: length-1-loop threshold.")
    parser.add_argument("--bot_resources", default=DEFAULT_BOT_RESOURCES,
                        help="Resources executing instantaneously, per case study, e.g. "
                             "\"BPI12_reordered=112;bpi17_before=User_1\". Their events get start = end. "
                             "Pass \"\" to disable.")
    parser.add_argument("--missing_resource", default=DEFAULT_MISSING_RESOURCE,
                        help="Placeholder resource id meaning 'unknown resource', per case study, e.g. "
                             "\"BPI12_reordered=0\". Its events get start = enabled time. Pass \"\" to disable.")
    parser.add_argument("--instant_activities", default="",
                        help="Activities forced to be instantaneous, per case study, e.g. \"bpi17_before=A_Submitted,A_Complete\".")
    parser.add_argument("--instant_activity_prefixes", default=DEFAULT_INSTANT_ACTIVITY_PREFIXES,
                        help="Comma-separated label prefixes: every activity whose label starts with one of them is "
                             "instantaneous, in every case study. Pass \"\" to disable.")
    parser.add_argument("--outlier_threshold", type=float, default=float("nan"),
                        help="If set (e.g. 2.0), estimated durations above threshold * median duration of the "
                             "activity are capped to that value. Default: off (package default).")
    parser.add_argument("--re_estimation", default="median", choices=list(RE_ESTIMATION_METHODS),
                        help="How to fill the events whose start could not be estimated (package default: median).")
    return parser.parse_args()


def parse_per_case_study(spec: str) -> dict:
    """'CS1=a,b;CS2=c' -> {'CS1': {'a', 'b'}, 'CS2': {'c'}}"""
    out = {}
    for block in filter(None, (b.strip() for b in spec.split(";"))):
        cs, values = block.split("=", 1)
        out[cs.strip()] = {v.strip() for v in values.split(",") if v.strip()}
    return out


def build_configuration(args, cs: str, activities) -> Configuration:
    missing = parse_per_case_study(args.missing_resource).get(cs, set())
    if len(missing) > 1:
        raise ValueError(f"pix-framework accepts a single missing-resource id per log, got {missing} for {cs}")
    prefixes = tuple(p.strip() for p in args.instant_activity_prefixes.split(",") if p.strip())
    instant = parse_per_case_study(args.instant_activities).get(cs, set())
    if prefixes:
        instant |= {a for a in activities if a.startswith(prefixes)}
    return Configuration(
        log_ids=LOG_IDS,
        concurrency_oracle_type=CONCURRENCY_TYPES[args.concurrency],
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
        missing_resource=next(iter(missing)) if missing else "NOT_SET",
        re_estimation_method=RE_ESTIMATION_METHODS[args.re_estimation],
        bot_resources=parse_per_case_study(args.bot_resources).get(cs, set()),
        instant_activities=instant,
        concurrency_thresholds=ConcurrencyThresholds(df=args.df_threshold, l2l=args.l2l_threshold, l1l=args.l1l_threshold),
        reuse_current_start_times=False,
        consider_start_times=False,  # the recorded start times are fake: never trust them
        outlier_statistic=OutlierStatistic.MEDIAN,
        outlier_threshold=args.outlier_threshold,
    )


def print_concurrency(df: pd.DataFrame, config: Configuration, name: str) -> None:
    """Print the concurrent activity pairs the chosen oracle detects (the estimator builds the same oracle internally)."""
    oracle_class = {"heuristics": HeuristicsConcurrencyOracle, "alpha": AlphaConcurrencyOracle,
                    "df": DirectlyFollowsConcurrencyOracle}.get(name)
    if oracle_class is None:
        print(f"  concurrency oracle '{name}': enablement not used")
        return
    concurrency = oracle_class(df, config).concurrency
    pairs = sorted({tuple(sorted((a, b))) for a, bs in concurrency.items() for b in bs})
    print(f"  concurrency oracle '{name}': {len(pairs)} concurrent activity pairs")
    for a, b in pairs:
        print(f"    {a} || {b}")


def process_case_study(base_dir: Path, cs: str, args) -> None:
    case_dir = base_dir / "case_studies" / cs
    log_path = case_dir / f"log_{cs}{args.input_suffix}.xes"
    if args.input_suffix == args.output_suffix:
        raise ValueError("--input_suffix and --output_suffix must differ: the input log must never be overwritten")
    print(f"\n=== {cs}: reading {log_path.name}")
    raw = pd.DataFrame(pm4py.read_xes(str(log_path))).reset_index(drop=True)
    original_columns = list(raw.columns)

    # Per-case order by completion time (the package's sort_by_end_time); file order breaks ties.
    # The directly-follows counts behind the concurrency oracle depend on this order.
    df = raw[[CASE_ID_NAME, ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME, START_DATE_NAME, END_DATE_NAME]]
    df = df.sort_values([CASE_ID_NAME, END_DATE_NAME], kind="stable").copy()
    df[RESOURCE_COLUMN_NAME] = df[RESOURCE_COLUMN_NAME].astype(str)

    config = build_configuration(args, cs, df[ACTIVITY_COLUMN_NAME].unique())
    print(f"  {len(df)} events, {df[CASE_ID_NAME].nunique()} cases | bots={sorted(config.bot_resources)} "
          f"missing={config.missing_resource} | {len(config.instant_activities)} instant activities: "
          f"{sorted(config.instant_activities)}")
    print_concurrency(df, config, args.concurrency)

    est = StartTimeEstimator(df, config).estimate()
    est["estimated_start"] = pd.to_datetime(est["estimated_start"], utc=True)
    assert (est["estimated_start"] <= est[END_DATE_NAME]).all()

    # Output log: same rows, order and attributes as the input; only start:timestamp changes.
    out = raw.copy()
    out[START_DATE_NAME] = est["estimated_start"].reindex(out.index)
    out_xes = case_dir / f"log_{cs}{args.output_suffix}.xes"
    formatted = pm4py.format_dataframe(out, case_id=CASE_ID_NAME, activity_key=ACTIVITY_COLUMN_NAME,
                                       timestamp_key=END_DATE_NAME, start_timestamp_key=START_DATE_NAME)
    # format_dataframe adds helper columns (@@index, @@case_index, start_timestamp): keep only the original ones.
    pm4py.write_xes(formatted[original_columns], str(out_xes))
    print(f"  wrote {out_xes}")

    # Console summary only (no diagnostics file is written).
    waiting = est["estimated_start"] - pd.to_datetime(est["enabled_time"], utc=True)
    later = (est["estimated_start"] > est[START_DATE_NAME]).mean()
    earlier = (est["estimated_start"] < est[START_DATE_NAME]).mean()
    print(f"  start changed for {later + earlier:.1%} of events (later: {later:.1%}, earlier: {earlier:.1%}); "
          f"events with waiting > 0: {(waiting > pd.Timedelta(0)).mean():.1%}")
    kept = (est[END_DATE_NAME] - est["estimated_start"]).sum() / (est[END_DATE_NAME] - est[START_DATE_NAME]).sum()
    print(f"  total processing time kept: {kept:.1%} of the original")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        process_case_study(base_dir, cs, args)


if __name__ == "__main__":  # required: pix-framework parallelises with ProcessPoolExecutor (spawn on Windows)
    main()
