#!/usr/bin/env python3
"""
petri_net_discovery_fitness1.py
===============================

For every case study, builds a Petri net that replays its WHOLE event log with fitness 1, and is
the net the simulations of 4_run_recommendation_simulation.py use (since 2026-10-09).

The net is the directly-follows graph (DFG) of the full log without W_ activities
`case_studies/<dataset>/log_<dataset>_no_start_W_filtered.xes` (written by 0_filtering_W_activities.py;
it is the net the start-time estimation uses to build the final log_<dataset>.xes; only the order of
the completions matters, so its fake start times are irrelevant), converted with pm4py's "activity defines place" variant: one place per activity ("the last activity
executed was X"), one visible transition per observed step X -> Y, plus the start and end activities.
So:
  - every trace of the log is replayable (fitness 1), whatever its loops or rare steps;
  - unlike a flower model it only allows the steps X -> Y actually observed, so it keeps a reasonable
    precision (it forgets only the long-range constraints, i.e. what happened before the last activity);
  - every recommendation of step 3 can be applied in the simulation: the recommender proposes, after a
    prefix ending with X, only next activities Y observed after a window ending with X in the training
    log, hence steps X -> Y that are in the DFG of the full log.

For every case study it saves, in `case_studies/<dataset>/discovery_output/`:
    - petri_net_fitness1_<dataset>.pnml   -> the Petri net (PNML format)
    - petri_net_fitness1_<dataset>.jpg    -> a rendered image of it (needs Graphviz)
and records its quality as one row (net = "fitness1") of `discovery/petri_net_quality.csv`, next to the
rows of the best-F-score Split Miner nets of petri_net_discovery.py (net = "bestfscore").

Fitness and precision are computed with TOKEN-BASED replay: alignment-based precision runs out of memory
on these loop-rich nets (> 10 GB on BPI12). The best-F-score nets were measured with alignments, so the
two kinds of row are not exactly comparable (column `metric`).

Usage (from multi_obj/)
-----
    python discovery/petri_net_discovery_fitness1.py
    python discovery/petri_net_discovery_fitness1.py --case_studies "BAC,BPI12"
"""

import argparse
import importlib.util
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pm4py
from pm4py.objects.conversion.dfg import converter as dfg_converter

HERE = Path(__file__).resolve().parent
DEFAULT_CASE_STUDIES = "BAC,BPI12,bpi17_before,bpi17_after"
LOG_FILE_NAME_TEMPLATE = "log_{dataset}_no_start_W_filtered.xes"

# shared helpers (log discovery, image rendering, quality CSV) from the Split Miner discovery script
_spec = importlib.util.spec_from_file_location("petri_net_discovery", HERE / "petri_net_discovery.py")
pnd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pnd)


def build_fitness1_net(log):
    """DFG of the log -> Petri net, 'activity defines place' variant. Returns (net, im, fm, n_dfg_edges)."""
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    variant = dfg_converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE
    params = variant.value.Parameters
    net, im, fm = dfg_converter.apply(dfg, variant=variant, parameters={
        params.START_ACTIVITIES: start_activities, params.END_ACTIVITIES: end_activities})
    return net, im, fm, len(dfg)


def _ensure_graphviz_on_path():
    """Graphviz's `dot` ships with the conda environment (Library/bin on Windows) but is not always on PATH;
    the `dot.BAT` wrapper in Scripts/ that shutil.which may find cannot be run by the graphviz package."""
    candidate = Path(sys.prefix) / "Library" / "bin"
    if (candidate / "dot.exe").exists():
        os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
    elif shutil.which("dot") is None:
        print("  [!] Graphviz 'dot' not found: the .jpg images will not be saved (the .pnml nets are)")


def process_case_study(base_dir: Path, case_study: str):
    xes_path = base_dir / "case_studies" / case_study / LOG_FILE_NAME_TEMPLATE.format(dataset=case_study)
    if not xes_path.exists():
        # a simulated case study (1b_generate_simulated_training_set.py, e.g. BPI12_sim) has only its simulated
        # log_<dataset>.xes, already without W_ activities (it is simulated from a net discovered without them)
        fallback = base_dir / "case_studies" / case_study / f"log_{case_study}.xes"
        if not fallback.exists():
            print(f"[!] {case_study}: no log at {xes_path} nor {fallback}, skipped")
            return
        print(f"[i] {case_study}: no {xes_path.name}, using {fallback.name}")
        xes_path = fallback
    t0 = time.time()
    print(f"=== {case_study} ({xes_path}) ===")
    log = pnd.load_event_log(str(xes_path))
    net, im, fm, n_edges = build_fitness1_net(log)

    out_dir = xes_path.parent / "discovery_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pnml_path = out_dir / f"petri_net_fitness1_{case_study}.pnml"
    jpg_path = out_dir / f"petri_net_fitness1_{case_study}.jpg"
    pm4py.write_pnml(net, im, fm, str(pnml_path))
    print(f"  [ok] Saved: {pnml_path}")
    try:
        pnd.save_petri_net_image(net, im, fm, jpg_path)
        print(f"  [ok] Saved: {jpg_path}")
    except Exception as exc:  # the image is only for reading; the .pnml is what the simulator uses
        print(f"  [!] Image not saved ({type(exc).__name__}: {exc})")

    fit = pm4py.fitness_token_based_replay(log, net, im, fm)
    precision = pm4py.precision_token_based_replay(log, net, im, fm)
    fitness = fit["log_fitness"]
    f_score = 2 * fitness * precision / (fitness + precision) if fitness + precision > 0 else 0.0
    print(f"  [ok] token-based fitness {fitness:.4f} (fitting traces {fit['perc_fit_traces']:.1f}%) | "
          f"precision {precision:.4f} | F-score {f_score:.4f} | {(time.time() - t0) / 60:.1f} min")
    if fitness < 1 - 1e-9:
        print("  [!] WARNING: fitness below 1 -- some recommendations may not be applicable in the simulation")

    n_silent = sum(t.label is None for t in net.transitions)
    pnd.save_quality_row(pnd.QUALITY_CSV_PATH, {
        "dataset": case_study,
        "net": "fitness1",
        "miner": "dfg_activity_defines_place",
        "metric": "token_based",
        "epsilon": float("nan"),
        "eta": float("nan"),
        "fitness": fitness,
        "fitting_traces_pct": fit["perc_fit_traces"],
        "precision": precision,
        "f_score": f_score,
        "n_places": len(net.places),
        "n_visible_transitions": len(net.transitions) - n_silent,
        "n_silent_transitions": n_silent,
        "n_dfg_edges": n_edges,
        "n_traces": log["case:concept:name"].nunique() if hasattr(log, "columns") else len(log),
        "pnml": str(pnml_path),
        "discovered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    print()


def main():
    parser = argparse.ArgumentParser(description="Build, for every case study, a fitness-1 Petri net "
                                                 "(directly-follows graph, 'activity defines place') from its full log.")
    parser.add_argument("--case_studies", default=DEFAULT_CASE_STUDIES,
                        help=f"Comma-separated case studies (default: {DEFAULT_CASE_STUDIES}).")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ.pop("GV_FILE_PATH", None)
    _ensure_graphviz_on_path()
    base_dir = pnd.resolve_base_dir()
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        process_case_study(base_dir, cs)
    print("Done.")


if __name__ == "__main__":
    main()
