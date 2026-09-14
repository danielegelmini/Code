#!/usr/bin/env python3
"""
Screens every case study's discovered Petri net for "duplicate task" gaps: activities
that Split Miner (like Inductive Miner and Heuristics Miner) can only place at ONE
position in the model -- by construction, since these algorithms map activity labels
to model tasks via a BIJECTIVE function (Split Miner paper, Augusto, Conforti, Dumas,
La Rosa, Polyvyanyy, Definition 3: "its Directly-Follows Graph (DFG) is a directed
graph G=(N,E), where N is the non-empty set of nodes, for which exists a bijective
function l : N -> L" -- footnote: "Each node of the graph represents a task."
Verbatim text confirmed against the local copy, Articoli/11 - split_miner_journal.pdf;
the paper is the journal extension of the ICDM 2017 conference paper -- double-check
the exact venue/year before citing it as "ICDM 2017" specifically). When an
activity genuinely occurs from more than one structurally distinct point in the real
process, no single position can represent all of its real occurrences, and the ones
reached from an "unexpected" predecessor become alignment deviations (log moves) that
the ProSiT simulator's parameter discovery never learns from -- see
9_generate_simulated_training_set.py and the A_CANCELLED investigation it prompted.

This is a CHEAP proxy check (no alignment, just a Petri net traversal + a groupby on the
raw log): for every labeled transition, it walks backward through invisible/silent
transitions to find the set of activity labels the model allows immediately before it,
then compares that against the real log's empirical immediate-predecessor distribution.
A high "mismatch fraction" (real occurrences preceded by something the model doesn't
allow) is strong evidence of a duplicate-task gap for that activity -- confirmed
definitively (as done for BPI12's A_CANCELLED: 0 of 21621 opportunities, across the
entire aligned log) only by the much slower alignment-based check in
verify_transition_alignment_gap.py.

Each flagged activity is also triaged by its real predecessor distribution, to gauge
whether relabeling-before-discovery (splitting the one activity label into
context-specific sub-labels so Split Miner naturally gives each its own transition) is
likely to help:
  - "clean split candidate": a small number of clearly dominant, comparably-sized
    predecessor contexts (top-2 cover >=70%, each >=15%) -- a natural 2-3-way split.
  - "mostly one predecessor": >=85% from a single predecessor -- the mismatch is more
    likely rare deviations/noise than a real duplicate-task gap; probably not worth
    splitting.
  - "diffuse": neither of the above -- immediate predecessor alone doesn't cleanly
    separate the contexts; would need a richer split criterion (or isn't a duplicate
    task at all).

Usage:
    python discovery/analyze_duplicate_task_candidates.py
    python discovery/analyze_duplicate_task_candidates.py --case_study BPI12
    python discovery/analyze_duplicate_task_candidates.py --mismatch_threshold 0.20 --out_csv suspects.csv
"""
import argparse
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import pm4py

warnings.filterwarnings("ignore")

# case_study -> (petri net path, xes log path), relative to multi_obj/
CASE_STUDIES = {
    "BAC": ("case_studies/BAC/discovery_output/BAC_best_petri_net.pnml", "case_studies/BAC/log_BAC.xes"),
    "BPI12": ("case_studies/BPI12/discovery_output/BPI12_best_petri_net.pnml", "case_studies/BPI12/log_BPI12.xes"),
    "bpi17_before": ("case_studies/bpi17_before/discovery_output/bpi17_before_best_petri_net.pnml", "case_studies/bpi17_before/log_bpi17_before.xes"),
    "bpi17_after": ("case_studies/bpi17_after/discovery_output/bpi17_after_best_petri_net.pnml", "case_studies/bpi17_after/log_bpi17_after.xes"),
}


def model_predecessor_labels(transition, max_depth: int = 6) -> set:
    """Real (labeled) activities the net allows immediately before `transition`, walking
    backward through invisible/silent transitions (BFS, depth-capped, visited-place-safe
    against cycles)."""
    visited_places = set()
    frontier = [a.source for a in transition.in_arcs]
    preds = set()
    depth = 0
    while frontier and depth < max_depth:
        next_frontier = []
        for place in frontier:
            if place in visited_places:
                continue
            visited_places.add(place)
            for a in place.in_arcs:
                t = a.source
                if t.label is not None:
                    preds.add(t.label)
                else:
                    next_frontier.extend(arc.source for arc in t.in_arcs)
        frontier = next_frontier
        depth += 1
    return preds


def classify_split_candidate(pred_dist: pd.Series) -> str:
    """Triage an activity's real predecessor distribution (normalized value_counts) into
    a recommendation for whether relabeling-before-discovery is likely to help."""
    top = pred_dist.sort_values(ascending=False)
    if len(top) >= 2 and top.iloc[0] >= 0.15 and top.iloc[1] >= 0.15 and top.iloc[:2].sum() >= 0.70:
        return "clean split candidate"
    if len(top) >= 1 and top.iloc[0] >= 0.85:
        return "mostly one predecessor (not worth splitting)"
    return "diffuse (needs richer context than immediate predecessor)"


def analyze_case_study(name: str, pnml_path: str, xes_path: str, mismatch_threshold: float) -> pd.DataFrame:
    """One row per activity whose mismatch fraction exceeds `mismatch_threshold` and
    that doesn't already have >1 transition in the net (i.e. isn't already
    duplicate-task-aware)."""
    net, im, fm = pm4py.read_pnml(pnml_path)
    log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(log)
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    df["prev_activity"] = df.groupby("case:concept:name")["concept:name"].shift(1)

    label_counts = Counter(t.label for t in net.transitions if t.label)
    already_duplicated = {lbl for lbl, c in label_counts.items() if c > 1}

    rows = []
    for label in sorted(label_counts.keys()):
        if label in already_duplicated:
            continue
        transitions = [t for t in net.transitions if t.label == label]
        allowed = set()
        for t in transitions:
            allowed |= model_predecessor_labels(t)

        real_rows = df[df["concept:name"] == label]
        n = len(real_rows)
        if n == 0:
            continue

        prev = real_rows["prev_activity"]
        is_start = prev.isna()
        mismatch = (~prev.isin(allowed)) & (~is_start)
        mismatch_frac = mismatch.sum() / n

        if mismatch_frac <= mismatch_threshold:
            continue

        pred_dist = prev.dropna().value_counts(normalize=True)
        top3 = pred_dist.sort_values(ascending=False).head(3)

        rows.append({
            "case_study": name,
            "activity": label,
            "n_occurrences": n,
            "mismatch_fraction": round(mismatch_frac, 4),
            "triage": classify_split_candidate(pred_dist),
            "top_predecessors": "; ".join(f"{k}={v:.1%}" for k, v in top3.items()),
        })

    return pd.DataFrame(rows).sort_values("mismatch_fraction", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Screen discovered Petri nets for duplicate-task gaps (a proxy check, no alignment)."
    )
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_study", type=str, default=None, choices=list(CASE_STUDIES),
                        help="Analyze only this case study (default: all of them).")
    parser.add_argument("--mismatch_threshold", type=float, default=0.15,
                        help="Flag an activity only if its mismatch fraction exceeds this (default: 0.15).")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Optional path to save the combined results table as CSV.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    targets = {args.case_study: CASE_STUDIES[args.case_study]} if args.case_study else CASE_STUDIES

    all_results = []
    for name, (pnml, xes) in targets.items():
        print(f"=== {name} ===")
        result = analyze_case_study(name, str(base_dir / pnml), str(base_dir / xes), args.mismatch_threshold)
        if result.empty:
            print("  (no suspects above threshold)")
        else:
            for _, r in result.iterrows():
                print(f"  {r['activity']:30s} mismatch={r['mismatch_fraction']:.1%}  n={r['n_occurrences']:6d}  [{r['triage']}]")
                print(f"    predecessors: {r['top_predecessors']}")
        all_results.append(result)
        print()

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    print("=== SUMMARY ===")
    for name in targets:
        res = combined[combined["case_study"] == name] if not combined.empty else combined
        clean = (res["triage"] == "clean split candidate").sum() if not res.empty else 0
        print(f"{name:15s} n_suspects={len(res):2d}  clean_split_candidates={clean}")

    if args.out_csv and not combined.empty:
        combined.to_csv(args.out_csv, index=False)
        print(f"\nSaved combined results to {args.out_csv}")


if __name__ == "__main__":
    main()

# Running commands:
# python discovery/analyze_duplicate_task_candidates.py
# python discovery/analyze_duplicate_task_candidates.py --case_study BPI12 --out_csv bpi12_duplicate_task_suspects.csv
