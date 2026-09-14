#!/usr/bin/env python3
"""
Isolated experiment: relabel several BPI12 activities based on whether their real
immediate predecessor is the one Split Miner's discovered net already connects them to,
re-mine a Petri net with Split Miner from the relabeled log, discover ProSiT simulation
parameters (rules mode) from the same relabeled log, simulate a fresh batch of traces,
collapse the relabeled activities back to their original names, and compare
activity-level statistics against the real log.

The activities and their ALREADY-MODELED predecessor(s) (from
discovery/analyze_duplicate_task_candidates.py's "clean split candidate" triage on
BPI12) are in RELABEL_SPECS below. Every occurrence gets one of two new labels:
"<activity>__modeled" if its real immediate predecessor is already a legal predecessor
in the current net, "<activity>__other" otherwise (this is where the missing real path
-- e.g. O_CANCELLED for A_CANCELLED -- ends up). Split Miner, seeing two distinct
labels instead of one ambiguous one, discovers a dedicated transition for each.

This is deliberately narrow in scope (log-level fidelity only, NOT the ML training
dataset / predictive models -- see 9_generate_simulated_training_set.py for that once
this is validated) and uses FIXED (epsilon, eta) = (0.1, 0.4), Split Miner's own
paper-recommended defaults, instead of the full Optuna hyperparameter search
(discovery/petri_net_discovery.py's default), to keep the alignment-based
fitness/precision evaluation to a single pass instead of ~30.

Everything is written under case_studies/BPI12_relabeled/ -- the original BPI12
case study folder is never touched.

Usage:
    python discovery/relabel_and_test_duplicate_task.py
    python discovery/relabel_and_test_duplicate_task.py --n_traces 5232 --epsilon 0.1 --eta 0.4
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discovery.petri_net_discovery import discover_petri_net, save_petri_net_image  # noqa: E402
from prosit.simulator import SimulatorParameters, SimulatorEngine  # noqa: E402

SOURCE_LOG = Path("case_studies/BPI12/log_BPI12.xes")
TARGET_DIR = Path("case_studies/BPI12_relabeled")
RELABELED_XES = TARGET_DIR / "log_BPI12_relabeled.xes"
PNML_PATH = TARGET_DIR / "discovery_output" / "BPI12_relabeled_best_petri_net.pnml"

# activity -> list of predecessor labels ALREADY legal in the original net (verified via
# discovery/analyze_duplicate_task_candidates.py's model_predecessor_labels against
# case_studies/BPI12/discovery_output/BPI12_best_petri_net.pnml). Everything else becomes
# "__other" -- that catch-all is where the real, currently-missing path lands.
RELABEL_SPECS = {
    "A_CANCELLED": ["W_Nabellen offertes"],
    "O_SENT_BACK": ["W_Completeren aanvraag"],
    "O_DECLINED": ["W_Nabellen offertes"],
    "O_SELECTED": ["A_ACCEPTED", "O_CANCELLED"],
    "A_DECLINED": ["W_Nabellen offertes", "O_DECLINED"],
}

SUFFIX_MODELED = "__modeled"
SUFFIX_OTHER = "__other"
COMPARISON_ACTIVITIES = list(RELABEL_SPECS.keys()) + [
    "O_CANCELLED", "O_SELECTED", "O_CREATED", "O_SENT", "W_Nabellen offertes",
]


def build_relabeled_log() -> pd.DataFrame:
    log = pm4py.read_xes(str(SOURCE_LOG))
    df = pm4py.convert_to_dataframe(log)
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    prev = df.groupby("case:concept:name")["concept:name"].shift(1)

    for activity, modeled_preds in RELABEL_SPECS.items():
        is_target = df["concept:name"] == activity
        is_modeled = is_target & prev.isin(modeled_preds)
        is_other = is_target & ~is_modeled

        df.loc[is_modeled, "concept:name"] = activity + SUFFIX_MODELED
        df.loc[is_other, "concept:name"] = activity + SUFFIX_OTHER
        print(f"  {activity}: {is_modeled.sum()} -> {activity}{SUFFIX_MODELED}, "
              f"{is_other.sum()} -> {activity}{SUFFIX_OTHER}")

    return df


def collapse_labels(df: pd.DataFrame, col: str = "concept:name") -> pd.DataFrame:
    df = df.copy()
    mapping = {}
    for activity in RELABEL_SPECS:
        mapping[activity + SUFFIX_MODELED] = activity
        mapping[activity + SUFFIX_OTHER] = activity
    df[col] = df[col].replace(mapping)
    return df


def trace_stats(df: pd.DataFrame, label: str) -> None:
    df = df.copy()
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], format="mixed", utc=True)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="mixed", utc=True)
    n_traces = df["case:concept:name"].nunique()
    events_per_trace = len(df) / n_traces
    g = df.groupby("case:concept:name").agg(start=("start:timestamp", "min"), end=("time:timestamp", "max"))
    duration_days = (g["end"] - g["start"]).dt.total_seconds() / 86400
    print(f"[{label}] traces={n_traces}  events/trace={events_per_trace:.2f}  "
          f"duration mean={duration_days.mean():.2f}d median={duration_days.median():.2f}d max={duration_days.max():.1f}d")

    for act in COMPARISON_ACTIVITIES:
        counts = df.groupby("case:concept:name")["concept:name"].apply(lambda s: (s == act).sum())
        print(f"    {act:24s} mean={counts.mean():.3f}  max={counts.max()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_traces", type=int, default=5232, help="Traces to simulate (default: matches earlier experiments).")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Split Miner epsilon (default: paper's recommended 0.1).")
    parser.add_argument("--eta", type=float, default=0.4, help="Split Miner eta (default: paper's recommended 0.4).")
    parser.add_argument("--max_depth_tree", type=int, default=2, help="ProSiT rules-mode depth (default: 2, matching prior fixed runs).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pnml_path", type=str, default=None,
                        help="Use an already-mined net instead of mining one with fixed "
                             "epsilon/eta (e.g. the Optuna-tuned net from "
                             "discovery/petri_net_discovery.py). Skips STEP 2 entirely.")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "discovery_output").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" STEP 1: relabel the log")
    print("=" * 60)
    relabeled_df = build_relabeled_log()
    formatted = pm4py.format_dataframe(
        relabeled_df.copy(), case_id="case:concept:name", activity_key="concept:name",
        timestamp_key="time:timestamp", start_timestamp_key="start:timestamp",
    )
    # format_dataframe() adds bookkeeping columns (@@index, @@case_index) and a duplicate
    # 'start_timestamp' (underscore, alongside the real 'start:timestamp') -- left in, these
    # leak through prosit's return_label_data_attributes() (it treats any non-standard column
    # as a trace/event attribute to condition the routing model on) as spurious "features",
    # one of them a raw Timestamp DecisionTreeClassifier can't fit on at all. Drop them before
    # writing, same as 9_generate_simulated_training_set.py's run_feature_pipeline does.
    formatted = formatted.drop(columns=["@@index", "@@case_index", "start_timestamp"], errors="ignore")
    pm4py.write_xes(formatted, str(RELABELED_XES))
    print(f"Wrote {RELABELED_XES}")

    if args.pnml_path:
        print("\n" + "=" * 60)
        print(f" STEP 2: load the already-mined net {args.pnml_path}")
        print("=" * 60)
        net, im, fm = pm4py.read_pnml(args.pnml_path)
    else:
        print("\n" + "=" * 60)
        print(f" STEP 2: mine a Petri net (Split Miner, epsilon={args.epsilon}, eta={args.eta}, no Optuna search)")
        print("=" * 60)
        relabeled_log = pm4py.read_xes(str(RELABELED_XES))
        net, im, fm = discover_petri_net(relabeled_log, args.epsilon, args.eta)
        pm4py.write_pnml(net, im, fm, str(PNML_PATH))
        print(f"Wrote {PNML_PATH}")
        try:
            save_petri_net_image(net, im, fm, str(PNML_PATH.with_suffix(".jpg")))
        except Exception as exc:
            print(f"(skipping net image -- Graphviz 'dot' executable not available: {exc})")
    for activity in RELABEL_SPECS:
        found = [t.label for t in net.transitions if t.label and t.label.startswith(activity)]
        print(f"  Transitions for {activity}*: {found}")

    print("\n" + "=" * 60)
    print(f" STEP 3: discover ProSiT simulation parameters (max_depth_tree={args.max_depth_tree})")
    print("=" * 60)
    # discover_from_eventlog needs the classic EventLog object (iterates trace/event as
    # dict-like), not pm4py.read_xes()'s return type used above for Split Miner -- mirrors
    # 9_generate_simulated_training_set.py's load_simulator.
    classic_log = xes_importer.apply(str(RELABELED_XES))
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(classic_log, max_depth_tree=args.max_depth_tree)
    engine = SimulatorEngine(params)

    print("\n" + "=" * 60)
    print(f" STEP 4: simulate {args.n_traces} fresh traces")
    print("=" * 60)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    t_start = pd.to_datetime(relabeled_df["start:timestamp"], format="mixed", utc=True).min().tz_localize(None)
    sim = engine.apply(n_traces=args.n_traces, t_start=t_start.to_pydatetime())
    sim = sim[["case:concept:name", "concept:name", "org:resource", "start:timestamp", "time:timestamp"]]
    sim.to_csv(TARGET_DIR / "simulated_event_log_relabeled.csv", index=False)

    sim_collapsed = collapse_labels(sim)
    sim_collapsed.to_csv(TARGET_DIR / "simulated_event_log.csv", index=False)
    print(f"Wrote {TARGET_DIR / 'simulated_event_log.csv'} (relabeled activities collapsed back to originals)")

    print("\n" + "=" * 60)
    print(" STEP 5: compare real vs simulated")
    print("=" * 60)
    real = pm4py.convert_to_dataframe(pm4py.read_xes(str(SOURCE_LOG)))
    trace_stats(real, "REAL")
    trace_stats(sim_collapsed, "SIM (relabeled + re-mined)")


if __name__ == "__main__":
    main()

# Running commands:
# python discovery/relabel_and_test_duplicate_task.py
