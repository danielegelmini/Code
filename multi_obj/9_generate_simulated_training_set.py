#!/usr/bin/env python3
"""
Experiment helper: build a *fully simulated* training set for a case study.

Motivation
----------
The downstream numbers for BPI12 do not reconcile the way we expect. To isolate
whether the prosit simulator is the cause, this script replaces the real training
log with a synthetic one: it draws exactly as many brand-new traces as the real
training split contains, straight from the simulation parameters already
discovered from the original event log (control-flow weights, resource/calendar
model, execution/waiting/arrival time distributions, trace-attribute marginals).
No historical prefix is replayed -- every trace starts from the Petri net's
initial marking and is simulated to completion.

The models can then be retrained on this synthetic training set (script 2) and
evaluated against the *real* test set, so any gap between "trained on real data"
and "trained on simulated data" is attributable to the simulator.

What it produces (under case_studies/<target_case_study>/)
--------------------------------------------------------
  * log_<target_case_study>.xes        -- the raw simulated event log
  * simulated_event_log.csv            -- same, flat CSV, for quick inspection
  * preprocessed_data.csv              -- after the standard feature pipeline
  * train_data.csv                     -- 100% of the simulated traces (this is
                                          the whole point: no train/test split is
                                          applied to the synthetic data)
  * test_data.csv / test_log.csv /
    test_log_with_last_act.csv         -- copied from the REAL source case study,
                                          with sigmoid_mm / outcome recomputed
                                          from the simulated-train scaler so the
                                          regression target lives in the same
                                          space the model is trained on
                                          (--no-reuse_source_test to skip)

It does NOT create the case-study folder skeleton (discovery_output/, model/,
recommendations/, ...) or touch scripts 1-8's per-case-study configuration
(get_features, data_labelling). The folder and the Petri
net / params cache under discovery_output/ are expected to be in place already
-- copy them from the source case study.

Usage
-----
    python 9_generate_simulated_training_set.py
    python 9_generate_simulated_training_set.py --source_case_study BPI12 --target_case_study BPI12_sim
    python 9_generate_simulated_training_set.py --n_train_traces 3737 --seed 7
    python 9_generate_simulated_training_set.py --force_rediscover   # ignore the params cache
"""
import argparse
import math
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

from prosit.simulator import SimulatorParameters, SimulatorEngine

from utils.get_features import get_features
from utils.data_normalization import fit_remaining_time_scalers, remaining_time_to_sigmoid_mm
from utils.pre_processing_functions import (
    getting_total_time,
    preprocessing_activity_frequency,
    add_next_act_res,
    prepare_data_and_add_features,
    data_labelling,
    linear_combination,
)

warnings.filterwarnings("ignore")

CASE_ID_NAME = "case:concept:name"
START_DATE_NAME = "start:timestamp"
END_DATE_NAME = "time:timestamp"
ACTIVITY_COLUMN_NAME = "concept:name"
RESOURCE_COLUMN_NAME = "org:resource"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
LAMBDA_VALUE = 0.5  # linear_combination weight, matching 1_data_preprocessing.py's default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a fully simulated training set for a case study."
    )
    parser.add_argument("--source_case_study", type=str, default="BPI12",
                        help="Case study whose discovered simulation parameters, Petri net, "
                             "training-set size and real test set are used (default: BPI12).")
    parser.add_argument("--target_case_study", type=str, default="BPI12_sim",
                        help="Output case-study folder under case_studies/ (default: BPI12_sim). "
                             "Must already exist with a populated discovery_output/.")
    parser.add_argument("--n_train_traces", type=int, default=None,
                        help="Number of simulated traces to keep as training data. "
                             "Defaults to the distinct-case count of the source train_data.csv.")
    parser.add_argument("--label_rule", type=str, default=None,
                        help="Case-study key passed to data_labelling() for the outcome label "
                             "(default: --source_case_study). Use it because the target folder "
                             "name is not known to data_labelling().")
    parser.add_argument("--t_start", type=str, default=None,
                        help="Arrival timestamp of the first simulated trace "
                             "(e.g. '2011-10-01 00:38:44'). Defaults to the earliest "
                             "start:timestamp of the source event log.")
    parser.add_argument("--oversample", type=float, default=1.0,
                        help="Simulate ceil(n_train_traces * oversample) traces (default: 1.0, i.e. "
                             "simulate exactly n_train_traces, no surplus). trim_to_exact() still runs "
                             "afterwards as a safety net -- if the feature pipeline happens to lose a "
                             "few traces for any reason, it randomly subsamples whatever remains down "
                             "to n_train_traces, and warns instead of failing if fewer survived than "
                             "the target. Raise this above 1.0 only if you've observed real losses for "
                             "a given case study/net; empirically (BPI12_sim) the '>3 events' "
                             "filter alone drops ~0 traces, so oversampling was pure surplus there.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the simulator's RNG and the final trim (default: 42). "
                             "Note: the prosit simulator has residual run-to-run variation from "
                             "hash-seed-dependent ordering; set PYTHONHASHSEED=0 in the environment "
                             "as well if you need bit-for-bit reproducibility.")
    parser.add_argument("--reuse_source_test", dest="reuse_source_test", action="store_true", default=True,
                        help="Copy the source case study's test_data.csv / test_log.csv / "
                             "test_log_with_last_act.csv into the target, recomputing "
                             "sigmoid_mm/outcome from the simulated-train scaler (default: on).")
    parser.add_argument("--no-reuse_source_test", dest="reuse_source_test", action="store_false",
                        help="Do not touch the target's test files.")
    parser.add_argument("--force_rediscover", action="store_true",
                        help="Ignore any cached simulation parameters and re-run discovery "
                             "from the source .xes event log.")
    parser.add_argument("--max_depth_tree", type=int, default=0,
                        help="0 (default) discovers CONTEXT-INDEPENDENT simulation parameters: a "
                             "single global empirical probability per transition/duration, "
                             "ignoring case history and attributes entirely (rules_mode=False, "
                             "the 4_run_recommendation_simulation.py default -- see "
                             "prosit/discovery/cf_discovery.py's 'if not max_depths_cv' branch). "
                             ">0 discovers a small decision tree per transition/activity/resource, "
                             "CONDITIONED on case history and attributes (rules_mode=True) -- this "
                             "is what lets simulated routing actually depend on the case instead of "
                             "drawing from fixed odds. Cached separately from the depth-0 params "
                             "(simulator_params_<source>_depth<N>.json), so the depth-0 cache used "
                             "elsewhere in the pipeline is never touched.",
                        )
    parser.add_argument("--pnml_path", type=str, default=None,
                        help="Override the Petri net used for discovery+simulation (default: "
                             "case_studies/<source>/discovery_output/<source>_best_petri_net.pnml). "
                             "Use this to try a manually-patched net (e.g. "
                             "BPI12_best_petri_net_patched.pnml) without touching the original -- "
                             "the simulation-parameters cache filename includes the pnml's stem, so "
                             "a patched net's discovery is cached separately and never collides with "
                             "the original net's cache.")
    return parser.parse_args()


def _find_xes_log(case_dir: Path, case_study: str) -> Path:
    """Locate the historical .xes log for a case study (filename casing is not consistent)."""
    exact = case_dir / f"log_{case_study}.xes"
    if exact.exists():
        return exact
    candidates = sorted(case_dir.glob("log_*.xes"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"No unambiguous .xes event log under {case_dir} (tried {exact.name} and log_*.xes)."
    )


def load_simulator(source_dir: Path, source_case_study: str, force_rediscover: bool,
                   max_depth_tree: int = 0, pnml_path: str | None = None) -> SimulatorEngine:
    """Rebuild the SimulatorEngine for the source case study from its Petri net and its
    (cached, or freshly discovered) simulation parameters. Mirrors 4_run_recommendation_simulation.py's
    setup_simulator, minus the PNML_OVERRIDES special-casing (unused for BPI12).

    max_depth_tree selects the discovery mode (see parse_args' help) and the cache filename, so
    a depth>0 (context-dependent, "rules") discovery never overwrites the depth-0 cache the rest
    of the pipeline (3_/4_/5_/6_) reads for the source case study. pnml_path, when given, overrides
    which .pnml is used (e.g. a manually-patched net); the cache filename then also includes the
    net's own stem, so a patched net's discovery is never confused with the original net's.
    """
    if pnml_path:
        resolved_pnml_path = Path(pnml_path)
        if not resolved_pnml_path.is_absolute():
            resolved_pnml_path = source_dir / "discovery_output" / resolved_pnml_path
        net_tag = f"_{resolved_pnml_path.stem}"
    else:
        resolved_pnml_path = source_dir / "discovery_output" / f"{source_case_study}_best_petri_net.pnml"
        net_tag = ""
    pnml_path = resolved_pnml_path

    depth_tag = f"_depth{max_depth_tree}" if max_depth_tree > 0 else ""
    if net_tag or depth_tag:
        params_cache_path = source_dir / "discovery_output" / f"simulator_params_{source_case_study}{net_tag}{depth_tag}.json"
    else:
        params_cache_path = source_dir / "discovery_output" / f"simulator_params_{source_case_study}.json"

    if not pnml_path.exists():
        raise FileNotFoundError(
            f"No Petri net at {pnml_path}. Generate it first (see discovery/)."
        )
    print(f"Loading Petri net: {pnml_path}")
    net, im, fm = pm4py.read_pnml(str(pnml_path))

    params = SimulatorParameters(net, im, fm)
    if params_cache_path.exists() and not force_rediscover:
        print(f"Loading cached simulation parameters: {params_cache_path}")
        params.from_json(str(params_cache_path))
    else:
        log_path = _find_xes_log(source_dir, source_case_study)
        print(f"Discovering simulation parameters (max_depth_tree={max_depth_tree}) from the "
              f"full event log: {log_path}")
        log = xes_importer.apply(str(log_path))
        print(f"  ({len(log)} cases; this can take a few minutes"
              f"{' -- longer with max_depth_tree>0, a decision tree is fit per transition/activity/resource' if max_depth_tree > 0 else ''})...")
        params.discover_from_eventlog(log, max_depth_tree=max_depth_tree)
        params_cache_path.parent.mkdir(parents=True, exist_ok=True)
        params.to_json(str(params_cache_path))
        print(f"Cached simulation parameters to {params_cache_path}")

    return SimulatorEngine(params)


def collapse_trace_attribute_dummies(df: pd.DataFrame, params: SimulatorParameters) -> pd.DataFrame:
    """The simulator emits every categorical trace attribute as a block of one-hot
    columns ('AMOUNT_REQ = 20000', 'AMOUNT_REQ = 5000', ...). Collapse each block
    back to the single raw column ('AMOUNT_REQ') the preprocessing pipeline and
    get_features() expect."""
    df = df.copy()
    for attr in params.label_data_attributes_categorical:
        prefix = f"{attr} = "
        dummies = [c for c in df.columns if c.startswith(prefix)]
        if not dummies:
            continue
        active = df[dummies].to_numpy().argmax(axis=1)
        df[attr] = [dummies[i][len(prefix):] for i in active]
        df = df.drop(columns=dummies)
        print(f"Collapsed {len(dummies)} '{prefix}...' dummy columns into '{attr}'.")
    return df


def simulate_event_log(engine: SimulatorEngine, n_sim: int, t_start: pd.Timestamp,
                       seed: int, params: SimulatorParameters) -> pd.DataFrame:
    """Simulate n_sim fresh traces from scratch and return a flat event log with the
    standard columns plus the raw trace-attribute column(s)."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    print(f"\nSimulating {n_sim} traces from scratch (t_start={t_start})...")
    sim = engine.apply(n_traces=n_sim, t_start=t_start.to_pydatetime())
    sim = collapse_trace_attribute_dummies(sim, params)

    # Drop the simulator-only 'enabled:timestamp'; keep the event-log columns in the
    # canonical order, trace attributes last (same layout as the real .xes once flattened).
    attr_cols = [c for c in params.label_data_attributes if c in sim.columns]
    ordered = [CASE_ID_NAME, ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME,
               START_DATE_NAME, END_DATE_NAME] + attr_cols
    sim = sim[ordered].copy()

    # Give the synthetic cases an unmistakable id so they can never be confused with
    # (or collide with) real case ids in any later concatenation/analysis.
    width = len(str(sim[CASE_ID_NAME].nunique()))
    renumber = {cid: f"SIM_{i + 1:0{width}d}"
                for i, cid in enumerate(sim[CASE_ID_NAME].drop_duplicates())}
    sim[CASE_ID_NAME] = sim[CASE_ID_NAME].map(renumber)

    sim = sim.sort_values([START_DATE_NAME, END_DATE_NAME]).reset_index(drop=True)
    print(f"  -> {len(sim)} events, {sim[CASE_ID_NAME].nunique()} traces, "
          f"span {sim[START_DATE_NAME].min()} .. {sim[END_DATE_NAME].max()}")
    return sim


def write_xes(sim: pd.DataFrame, path: Path) -> None:
    formatted = pm4py.format_dataframe(
        sim.copy(), case_id=CASE_ID_NAME, activity_key=ACTIVITY_COLUMN_NAME,
        timestamp_key=END_DATE_NAME, start_timestamp_key=START_DATE_NAME,
    )
    pm4py.write_xes(formatted, str(path))
    print(f"Wrote {path}")


def run_feature_pipeline(sim: pd.DataFrame, label_rule: str) -> pd.DataFrame:
    """Reproduce 1_data_preprocessing.py's data_pre_processing() on an in-memory event
    log, with the outcome label computed for `label_rule` (the target folder name is
    unknown to data_labelling)."""
    data = sim.copy()
    ordered_cols = [CASE_ID_NAME, ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME,
                    START_DATE_NAME, END_DATE_NAME]
    remaining_cols = [c for c in data.columns if c not in ordered_cols]
    data = data[ordered_cols + remaining_cols]

    df = pm4py.utils.format_dataframe(
        data, case_id=CASE_ID_NAME, activity_key=ACTIVITY_COLUMN_NAME,
        timestamp_key=END_DATE_NAME, start_timestamp_key=START_DATE_NAME,
    )
    df["time_timestamp"] = df[END_DATE_NAME]
    df = df.drop(columns=["@@index", "@@case_index"])

    print("...adding total time...")
    df = getting_total_time(df, CASE_ID_NAME, START_DATE_NAME, END_DATE_NAME)
    print("...adding activity frequency...")
    df = preprocessing_activity_frequency(df, ACTIVITY_COLUMN_NAME, CASE_ID_NAME, START_DATE_NAME)
    print("...adding next activity and next resource...")
    df = add_next_act_res(df, ACTIVITY_COLUMN_NAME, RESOURCE_COLUMN_NAME, CASE_ID_NAME)

    print("...adding features...")
    case_id_position = df.columns.get_loc(CASE_ID_NAME)
    start_date_position = df.columns.get_loc(START_DATE_NAME)
    end_date_position = df.columns.get_loc(END_DATE_NAME)
    df = prepare_data_and_add_features(df, case_id_position, start_date_position,
                                      DATE_FORMAT, end_date_position)

    print(f"...adding outcome label (rule: {label_rule})...")
    df = data_labelling(df, label_rule)
    if "label" not in df.columns:
        raise SystemExit(
            f"data_labelling('{label_rule}') did not produce a 'label' column. "
            f"Pass --label_rule with one of the keys data_labelling knows "
            f"(bpi12, bpi17_before, bpi17_after, bac)."
        )

    # Drop cases with <= 3 events (not meaningful for recommendation), same as the pipeline.
    df = df[df.groupby(CASE_ID_NAME)[CASE_ID_NAME].transform("count") > 3].reset_index(drop=True)

    df = df.rename(columns={"time_timestamp": END_DATE_NAME, "start_timestamp": START_DATE_NAME,
                            "leadtime": "total_time"})
    del df["time_from_midnight"]
    return df


def reconcile_activity_columns(df: pd.DataFrame, source_case_study: str) -> pd.DataFrame:
    """Guarantee every '# ACTIVITY=' column get_features(source) references exists, even
    if the simulated log happened not to contain that activity at all (fill with 0)."""
    _, _, _, continuous_features, _, _ = get_features(source_case_study)
    expected = [c for c in continuous_features if c.startswith("# ACTIVITY=")]
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = 0
    if missing:
        print(f"NOTE: simulated log had no events for {missing} -- added as all-zero columns.")
    return df


def match_column_order(df: pd.DataFrame, reference_csv: Path) -> pd.DataFrame:
    """Reorder df's columns to match an existing reference CSV (same column set expected),
    so the synthetic files are drop-in comparable with the real ones. Columns not present
    in the reference are appended in their current order."""
    if not reference_csv.exists():
        return df
    ref_cols = list(pd.read_csv(reference_csv, nrows=0).columns)
    ordered = [c for c in ref_cols if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    if extra:
        print(f"NOTE: columns not in {reference_csv.name}, appended last: {extra}")
    return df[ordered + extra]


def trim_to_exact(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """Keep exactly n_target traces (random subset, seeded). Warn if fewer are available."""
    ids = df[CASE_ID_NAME].drop_duplicates().tolist()
    if len(ids) < n_target:
        print(f"WARNING: only {len(ids)} traces survived preprocessing (< target {n_target}). "
              f"Re-run with a larger --oversample.")
        return df
    keep = set(pd.Series(ids).sample(n=n_target, random_state=seed))
    return df[df[CASE_ID_NAME].isin(keep)].reset_index(drop=True)


def build_test_files(target_dir: Path, source_dir: Path, std_scaler, mm_scaler) -> None:
    """Copy the source case study's test frames into the target, recomputing sigmoid_mm
    (and the derived outcome) from the simulated-train scaler so the regression target
    is in the same space the model trained on BPI12_sim will predict in. Raw
    remaining_time and the classification label are left untouched."""
    for name in ("test_data.csv", "test_log.csv", "test_log_with_last_act.csv"):
        src = source_dir / name
        if not src.exists():
            print(f"NOTE: {src} not found -- skipping {name}.")
            continue
        df = pd.read_csv(src)
        if "remaining_time" in df.columns:
            df["sigmoid_mm"] = remaining_time_to_sigmoid_mm(df["remaining_time"], std_scaler, mm_scaler)
            df = linear_combination(df, lambda_weight=LAMBDA_VALUE)
        df.to_csv(target_dir / name, index=False)
        print(f"Wrote {target_dir / name}  (from {src.name}, sigmoid_mm/outcome recomputed)")


def main():
    args = parse_args()
    source_case_study = args.source_case_study
    target_case_study = args.target_case_study
    label_rule = args.label_rule or source_case_study

    source_dir = Path("case_studies") / source_case_study
    target_dir = Path("case_studies") / target_case_study
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise SystemExit(f"Source case study folder not found: {source_dir}")

    # --- training-set size -------------------------------------------------
    if args.n_train_traces is not None:
        n_target = args.n_train_traces
    else:
        src_train = pd.read_csv(source_dir / "train_data.csv", usecols=[CASE_ID_NAME])
        n_target = int(src_train[CASE_ID_NAME].nunique())
    print(f"Target simulated training traces: {n_target}")

    # --- t_start ---------------------------------------------------------
    if args.t_start is not None:
        t_start = pd.Timestamp(args.t_start)
    else:
        src_log = pd.read_csv(source_dir / "preprocessed_data.csv", usecols=[START_DATE_NAME])
        t_start = pd.to_datetime(src_log[START_DATE_NAME], format="mixed").min().tz_localize(None)
    print(f"First-arrival timestamp: {t_start}")

    # --- simulate --------------------------------------------------------
    engine = load_simulator(source_dir, source_case_study, args.force_rediscover, args.max_depth_tree, args.pnml_path)
    n_sim = math.ceil(n_target * args.oversample)
    sim = simulate_event_log(engine, n_sim, t_start, args.seed, engine.simulation_parameters)

    write_xes(sim, target_dir / f"log_{target_case_study}.xes")
    sim.to_csv(target_dir / "simulated_event_log.csv", index=False)
    print(f"Wrote {target_dir / 'simulated_event_log.csv'}")

    # --- feature pipeline ----------------------------------------------
    print("\n" + "=" * 60)
    print(" FEATURE PIPELINE ")
    print("=" * 60)
    feats = run_feature_pipeline(sim, label_rule)
    feats = reconcile_activity_columns(feats, source_case_study)
    feats = trim_to_exact(feats, n_target, args.seed)
    feats = match_column_order(feats, source_dir / "preprocessed_data.csv")
    feats.to_csv(target_dir / "preprocessed_data.csv", index=False)
    print(f"Wrote {target_dir / 'preprocessed_data.csv'}  "
          f"({feats[CASE_ID_NAME].nunique()} traces, {len(feats)} events)")

    # --- targets: normalise remaining_time on the simulated training set --
    train_data = feats.copy()
    std_scaler, mm_scaler = fit_remaining_time_scalers(train_data)
    train_data["sigmoid_mm"] = remaining_time_to_sigmoid_mm(train_data["remaining_time"], std_scaler, mm_scaler)
    train_data = linear_combination(train_data, lambda_weight=LAMBDA_VALUE)
    train_data = match_column_order(train_data, source_dir / "train_data.csv")
    train_data.to_csv(target_dir / "train_data.csv", index=False)
    print(f"Wrote {target_dir / 'train_data.csv'}  "
          f"(100% of the simulated traces: {train_data[CASE_ID_NAME].nunique()})")

    # --- test set ------------------------------------------------------
    if args.reuse_source_test:
        print("\n" + "=" * 60)
        print(" TEST SET (from real source case study) ")
        print("=" * 60)
        build_test_files(target_dir, source_dir, std_scaler, mm_scaler)

    print("\n" + "*" * 60)
    print(" DONE ")
    print("*" * 60)
    print(f"Next: train on the synthetic set with\n"
          f"  python 2_training_predictive_model.py\n"
          f"('{target_case_study}' is already wired into get_features / data_labelling / "
          f"the scripts 2-6 case-study configs and the BPI12 dtype handling.)")


if __name__ == "__main__":
    main()

# Running commands:
# python 9_generate_simulated_training_set.py --source_case_study BPI12 --target_case_study BPI12_sim
