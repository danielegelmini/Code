#!/usr/bin/env python3
"""
petri_net_discovery_fitness_constrained.py
==========================================

Experiment: select the Split Miner net by a fitness CONSTRAINT instead of the F-score
used by petri_net_discovery.py -- among the (epsilon, eta) combinations whose
alignment-based fitness reaches --min_fitness, take the one with the highest
alignment-based precision -- and compare it with the net currently used by the
simulator (<case>_best_petri_net.pnml).

Writes only new files, never the ones the pipeline reads:
    case_studies/<case>/discovery_output/<case>_fitness_constrained_petri_net.pnml
    case_studies/<case>/discovery_output/simulator_params_<case>_fitness_constrained.json
    discovery/fitness_constrained_<case>_grid.csv        (screening of every combination)
    discovery/fitness_constrained_<case>_comparison.csv  (current vs selected net)

Steps:
    1. Screening: every (epsilon, eta) on a 0.1 grid, scored with token-based replay
       fitness/precision (seconds per net; alignment-based metrics take minutes on BPI12).
    2. Selection: alignment-based fitness/precision for the best screened candidates;
       the selected net has the highest precision among those with fitness >= min_fitness
       (highest fitness if none reaches it).
    3. Generalization: 5-fold cross-validation over traces -- discover on 4 folds with
       the same (epsilon, eta), alignment fitness on the held-out fold.
    4. Behaviour: simulator parameters discovered for the selected net, baseline
       continuation of the test prefixes (same set-up as 4_run_recommendation_simulation.py)
       compared with the real test cases, plus reachability checks.

Usage (from multi_obj/):
    python discovery/petri_net_discovery_fitness_constrained.py --case_study BPI12
"""

import argparse
import importlib
import sys
import time
import uuid
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.split_miner import algorithm as split_miner
from pm4py.algo.discovery.split_miner.variants import classic as split_miner_classic

MULTI_OBJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MULTI_OBJ_DIR))

from prosit.simulator import SimulatorParameters, SimulatorEngine  # noqa: E402
from prosit.utils.common_utils import update_current_marking  # noqa: E402
from utils.simulation_functions import build_recommender_df  # noqa: E402

C, A, ST, E = "case:concept:name", "concept:name", "start:timestamp", "time:timestamp"
GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

# Outcome configuration of the case studies this experiment supports: the positive
# activity, and the activities that close a case with a decision.
OUTCOMES = {
    "BPI12": ("O_ACCEPTED", {"O_ACCEPTED", "A_CANCELLED", "A_DECLINED", "O_CANCELLED", "O_DECLINED"}),
    "bpi17_before": ("O_Accepted", {"O_Accepted", "O_Cancelled", "O_Refused", "A_Cancelled", "A_Denied"}),
    "bpi17_after": ("O_Accepted", {"O_Accepted", "O_Cancelled", "O_Refused", "A_Cancelled", "A_Denied"}),
}


def discover(log, eps, eta):
    params = {split_miner_classic.Parameters.EPSILON: eps, split_miner_classic.Parameters.ETA: eta}
    bpmn = split_miner.apply(log, parameters=params, variant=split_miner.Variants.CLASSIC)
    return pm4py.convert_to_petri_net(bpmn)


def signature(net):
    """Label-level structure of a net, to recognise identical nets across (epsilon, eta)."""
    arcs = sorted((str(a.source.label if hasattr(a.source, "label") else "P"),
                   str(a.target.label if hasattr(a.target, "label") else "P")) for a in net.arcs)
    return (len(net.places), len(net.transitions), tuple(arcs))


def alignment_quality(log, net, im, fm):
    fit = pm4py.fitness_alignments(log, net, im, fm)
    prec = pm4py.precision_alignments(log, net, im, fm)
    return fit["log_fitness"], fit["percentage_of_fitting_traces"] / 100, prec


def screen(log):
    rows, seen = [], {}
    for eps in GRID:
        for eta in GRID:
            t0 = time.time()
            try:
                net, im, fm = discover(log, eps, eta)
            except Exception as exc:  # noqa: BLE001 - a failing combination is just skipped
                print(f"  eps={eps} eta={eta}: discovery failed ({type(exc).__name__})", flush=True)
                continue
            sig = signature(net)
            if sig in seen:
                row = dict(seen[sig], epsilon=eps, eta=eta)
            else:
                fit = pm4py.fitness_token_based_replay(log, net, im, fm)
                prec = pm4py.precision_token_based_replay(log, net, im, fm)
                row = {"epsilon": eps, "eta": eta, "n_places": len(net.places),
                       "n_transitions": len(net.transitions),
                       "tbr_fitness": fit["log_fitness"], "tbr_precision": prec, "net_id": len(seen)}
                seen[sig] = row
            rows.append(row)
            print(f"  eps={eps} eta={eta}: tbr fitness {row['tbr_fitness']:.3f} precision {row['tbr_precision']:.3f}"
                  f" (net {row['net_id']}, {time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def cross_validated_fitness(log, eps, eta, folds=5, seed=0):
    cases = np.array(sorted(log[C].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    scores = []
    for k in range(folds):
        held_out = set(cases[k::folds])
        train, test = log[~log[C].isin(held_out)], log[log[C].isin(held_out)]
        net, im, fm = discover(train, eps, eta)
        scores.append(pm4py.fitness_alignments(test, net, im, fm)["log_fitness"])
    return float(np.mean(scores)), float(np.std(scores))


def ends_without_decision(eng, net, im, fm, decisions):
    key = lambda m: frozenset((p.name, n) for p, n in m.items())
    seen, queue = {key(im)}, deque([im])
    while queue:
        m = queue.popleft()
        if m == fm:
            return True
        for t in eng._get_enabled_transitions_sorted(m):
            if t.label in decisions:
                continue
            m2 = update_current_marking(m, t)
            if key(m2) not in seen and sum(m2.values()) <= 20:
                seen.add(key(m2)); queue.append(m2)
    return False


def behaviour(case_study, case_dir, log, net, im, fm, params, n_runs):
    positive_act, decisions = OUTCOMES[case_study]
    s4 = importlib.import_module("4_run_recommendation_simulation")
    eng = SimulatorEngine(params)
    prev, _ = s4.load_inputs(case_dir, case_study, None)
    prev[C] = prev[C].astype(str)
    t_split = s4.load_split_time(case_dir)
    groups = dict(list(prev.sort_values([C, E], kind="stable").groupby(C)))
    real = pd.read_csv(case_dir / "test_data.csv", usecols=[C, A, E], dtype={C: str}).sort_values([C, E], kind="stable")
    real = real[real[C].isin(groups)]

    def reachable(cid, act):
        m = eng._reconstruct_prefix_state(cid, groups[cid])["marking"]
        return act in [t.label for t in eng._get_enabled_transitions_sorted(m)] or \
            eng._bfs_path_to_activity(m, act, only_invisible=True)[0] is not None

    real_seq = real.groupby(C, sort=False)[A].agg(list)
    nxt = [reachable(c, real_seq[c][len(g)]) for c, g in groups.items() if len(real_seq[c]) > len(g)]
    non_fit = np.mean([not eng._reconstruct_prefix_state(c, g)["is_fit"] for c, g in groups.items()])
    recs = pd.read_csv(case_dir / f"recommendations/recommendations_{case_study}_exhaustive_top1of5.csv", dtype=str).dropna()
    recs = recs[recs[C].isin(groups)]
    rank1 = np.mean([reachable(c, a) for c, a in zip(recs[C], recs["Next_activity"])])

    base = build_recommender_df(prev.copy(), {c: {"act": f"__NO_RECOMMENDATION__{uuid.uuid4().hex}", "res": None} for c in groups})
    for col in (ST, E):
        base[col] = pd.to_datetime(base[col], format="mixed")
    pos, und, dur = [], [], []
    for _ in range(n_runs):
        out = eng.apply(prev_log=base, resume_time=t_split)
        out[C] = out[C].astype(str)
        acts = out.groupby(C)[A].agg(set)
        pos.append(acts.apply(lambda x: positive_act in x).mean())
        und.append(acts.apply(lambda x: not (x & decisions)).mean())
        d = (pd.to_datetime(out[E]).groupby(out[C]).max() - pd.to_datetime(out[ST]).groupby(out[C]).min())
        dur.append(d.dt.total_seconds().median() / 86400)
    return {
        "test_prefixes_non_fitting": non_fit,
        "real_next_activity_reachable": float(np.mean(nxt)),
        "rank1_recommendation_reachable": rank1,
        "ends_without_decision_possible": ends_without_decision(eng, net, im, fm, decisions),
        "baseline_positive": float(np.mean(pos)),
        "baseline_undecided": float(np.mean(und)),
        "baseline_median_duration_days": float(np.mean(dur)),
    }


def real_reference(case_study, case_dir):
    positive_act, decisions = OUTCOMES[case_study]
    ids = set(pd.read_csv(case_dir / "test_log.csv", usecols=[C], dtype={C: str})[C])
    real = pd.read_csv(case_dir / "test_data.csv", usecols=[C, A, ST, E], dtype={C: str})
    real = real[real[C].isin(ids)]
    acts = real.groupby(C)[A].agg(set)
    d = pd.to_datetime(real[E], format="mixed").groupby(real[C]).max() - pd.to_datetime(real[ST], format="mixed").groupby(real[C]).min()
    return {
        "baseline_positive": acts.apply(lambda x: positive_act in x).mean(),
        "baseline_undecided": acts.apply(lambda x: not (x & decisions)).mean(),
        "baseline_median_duration_days": d.dt.total_seconds().median() / 86400,
    }


def main():
    parser = argparse.ArgumentParser(description="Fitness-constrained Split Miner selection, compared with the current net.")
    parser.add_argument("--case_study", default="BPI12", choices=sorted(OUTCOMES))
    parser.add_argument("--min_fitness", type=float, default=0.95)
    parser.add_argument("--n_shortlist", type=int, default=4, help="screened candidates scored with alignments")
    parser.add_argument("--n_runs", type=int, default=3, help="baseline simulation runs per net")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    cs = args.case_study
    case_dir = MULTI_OBJ_DIR / "case_studies" / cs
    out_dir = case_dir / "discovery_output"
    here = Path(__file__).resolve().parent
    grid_csv = here / f"fitness_constrained_{cs}_grid.csv"
    log = pm4py.read_xes(str(case_dir / f"log_{cs}.xes"))

    # 1. screening (cached: the grid CSV is reused if present)
    if grid_csv.exists():
        grid = pd.read_csv(grid_csv)
        print(f"Screening loaded from {grid_csv}")
    else:
        print("=== 1. Screening (token-based replay) ===", flush=True)
        grid = screen(log)
        grid.to_csv(grid_csv, index=False)

    current = pm4py.read_pnml(str(out_dir / f"{cs}_best_petri_net.pnml"))
    cur_sig = signature(current[0])
    cur_match = None
    for _, r in grid.drop_duplicates("net_id").iterrows():
        if signature(discover(log, r.epsilon, r.eta)[0]) == cur_sig:
            cur_match = r
            break
    print("Current net corresponds to", "(not found in grid)" if cur_match is None else f"eps={cur_match.epsilon}, eta={cur_match.eta}")

    # 2. alignment-based selection among the most precise screened nets with high token-based fitness
    print("\n=== 2. Selection (alignment-based) ===", flush=True)
    unique = grid.drop_duplicates("net_id")
    pool = unique[unique.tbr_fitness >= args.min_fitness]
    if pool.empty:
        pool = unique.nlargest(args.n_shortlist, "tbr_fitness")
    shortlist = pool.nlargest(args.n_shortlist, "tbr_precision")
    scored = []
    for _, r in shortlist.iterrows():
        net, im, fm = discover(log, r.epsilon, r.eta)
        f, pf, p = alignment_quality(log, net, im, fm)
        scored.append(dict(r, fitness=f, fitting_traces=pf, precision=p))
        print(f"  eps={r.epsilon} eta={r.eta}: fitness {f:.3f} ({pf:.1%} traces) precision {p:.3f}", flush=True)
    scored = pd.DataFrame(scored)
    ok = scored[scored.fitness >= args.min_fitness]
    best = ok.loc[ok.precision.idxmax()] if not ok.empty else scored.loc[scored.fitness.idxmax()]
    print(f"Selected: eps={best.epsilon} eta={best.eta}" + ("" if not ok.empty else f" (no net reaches fitness {args.min_fitness}: highest fitness taken)"))
    net, im, fm = discover(log, best.epsilon, best.eta)
    pm4py.write_pnml(net, im, fm, str(out_dir / f"{cs}_fitness_constrained_petri_net.pnml"))

    # 3-4. generalization and behaviour, current vs selected
    real = real_reference(cs, case_dir)
    rows = [dict(net="real test cases", **real)]
    candidates = [("current (F-score)", current, cur_match, out_dir / f"simulator_params_{cs}.json"),
                  ("fitness-constrained", (net, im, fm), best, out_dir / f"simulator_params_{cs}_fitness_constrained.json")]
    for name, (n, i, f_), params_row, params_path in candidates:
        print(f"\n=== 3-4. {name} ===", flush=True)
        fit, fit_pct, prec = alignment_quality(log, n, i, f_)
        cv_mean, cv_std = cross_validated_fitness(log, params_row.epsilon, params_row.eta) if params_row is not None else (np.nan, np.nan)
        params = SimulatorParameters(n, i, f_)
        if params_path.exists():
            params.from_json(str(params_path))
        else:
            # same set-up as 4_run_recommendation_simulation.py's setup_simulator
            params.discover_from_eventlog(xes_importer.apply(str(case_dir / f"log_{cs}.xes")), max_depth_tree=3)
            params.to_json(str(params_path))
        row = dict(net=name, epsilon=None if params_row is None else params_row.epsilon,
                   eta=None if params_row is None else params_row.eta,
                   fitness=fit, fitting_traces=fit_pct, precision=prec,
                   cv_heldout_fitness=cv_mean, cv_heldout_fitness_std=cv_std,
                   **behaviour(cs, case_dir, log, n, i, f_, params, args.n_runs))
        rows.append(row)
        print(row, flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(here / f"fitness_constrained_{cs}_comparison.csv", index=False)
    print("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
