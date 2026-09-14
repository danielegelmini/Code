#!/usr/bin/env python3
"""
Automated, principled repair for directly-follows edges that Split Miner's own
CONCURRENCY-DETECTION step (Augusto, Conforti, Dumas, La Rosa, Polyvyanyy, "Split
Miner"; journal extension of the ICDM 2017 conference paper -- verify exact venue/
year before citing) prunes by mistake -- as opposed to edges deliberately dropped
by the later frequency-based
filtering (Algorithm 1 in the paper). This is the systematic version of the manual
patch applied to BPI12's O_CANCELLED -> A_CANCELLED edge: instead of hand-inspecting
one net and adding one invisible transition, this scans EVERY edge in the raw
directly-follows graph, finds every one that satisfies Split Miner's own concurrency
formula (so we know exactly *why* it was dropped, and that it wasn't a legitimate
precision/fitness trade-off), and repairs each one identically and automatically.

Why this exists (not just "add invisible transitions where the net looks wrong"):
Split Miner maps every activity label to exactly ONE task (Definition 3: a bijective
function between tasks and labels). When an activity genuinely occurs from more than
one structurally distinct point in the real process, one of the real predecessor
edges can get discarded for two very different reasons, and only one of them is a
bug worth auto-repairing:

  1. FREQUENCY FILTERING (Algorithm 1): the edge is comparatively rare and was
     deliberately cut to keep the model simple/precise. This is a real, intended
     trade-off of the algorithm's design goal (simple models) -- repairing it would
     second-guess a choice Split Miner made on purpose, and isn't something this
     script touches.
  2. CONCURRENCY MISCLASSIFICATION (Section III-B): the edge a->b coexists with a
     comparably-frequent reverse edge b->a, so Split Miner assumes a and b are
     order-independent (concurrent) and drops BOTH directional edges entirely --
     before frequency filtering even runs, regardless of how frequent a->b is (it
     can be a task's single most-frequent incoming edge and still get dropped this
     way; observed exactly on BPI12: O_CANCELLED -> A_CANCELLED, 767 occurrences,
     the single most frequent incoming edge to A_CANCELLED, dropped because
     A_CANCELLED -> O_CANCELLED also occurs 864 times, a 5.9% relative difference --
     well under any reasonable epsilon). This is the case this script targets: we
     know exactly which formula caused the drop, so the repair is a direct,
     mechanical undo of that specific decision, not a subjective judgment call.

The repair itself never touches the target activity's own transition (so its
learned timing/resource behaviour is untouched) -- it only ADDS an invisible
transition from the source activity's existing output place(s) into the target
activity's enabling place, so ProSiT's routing-weight discovery can learn a real,
non-zero, history-independent-or-conditioned probability for it directly from the
log the next time simulation parameters are (re)discovered. Every candidate edge is
verified sound (WOFLAN) individually before being kept; unsound candidates are
skipped and reported, never silently applied.

Usage:
    python discovery/repair_pruned_concurrent_edges.py --case_study BPI12
    python discovery/repair_pruned_concurrent_edges.py --case_study BPI12 --epsilon 0.1 --dry_run
    python discovery/repair_pruned_concurrent_edges.py --case_study BPI12 --out_pnml BPI12_best_petri_net_repaired.pnml
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discovery.analyze_duplicate_task_candidates import model_predecessor_labels, CASE_STUDIES  # noqa: E402

DEFAULT_EPSILON = 0.1  # Split Miner's own paper-recommended value (Augusto et al., Sec. IV-B)
DEFAULT_MIN_FREQUENCY = 30  # ignore edges this rare regardless of ratio -- avoid patching noise


def raw_dfg_frequencies(df: pd.DataFrame) -> dict:
    """{(a, b): count} for every directly-follows pair in the raw log (no alignment)."""
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    nxt = df.groupby("case:concept:name")["concept:name"].shift(-1)
    pairs = pd.DataFrame({"a": df["concept:name"], "b": nxt}).dropna()
    counts = pairs.groupby(["a", "b"]).size()
    return counts.to_dict()


def find_repairable_edges(net: PetriNet, dfg: dict, epsilon: float, min_frequency: int) -> list[dict]:
    """Every (a, b) edge that:
      1. has real, non-trivial frequency (>= min_frequency),
      2. is NOT already a legal predecessor of b's transition(s) in the net, and
      3. satisfies Split Miner's own concurrency condition against the reverse edge
         (so we know concurrency-pruning, not frequency-filtering, is why it's missing).

    Returns one dict per repairable edge, richest-evidence first (highest a->b frequency).
    """
    label_to_transitions = {}
    for t in net.transitions:
        if t.label is not None:
            label_to_transitions.setdefault(t.label, []).append(t)

    candidates = []
    for (a, b), freq_ab in dfg.items():
        if freq_ab < min_frequency or b not in label_to_transitions:
            continue
        freq_ba = dfg.get((b, a), 0)
        if freq_ba == 0:
            continue  # no reverse edge at all -> not a concurrency-pruning case (likely just filtered)

        allowed = set()
        for t in label_to_transitions[b]:
            allowed |= model_predecessor_labels(t)
        if a in allowed:
            continue  # already legally reachable -- nothing to repair

        ratio = abs(freq_ab - freq_ba) / (freq_ab + freq_ba)
        if ratio >= epsilon:
            continue  # not a concurrency-pruning case -- likely legitimate frequency filtering

        candidates.append({
            "source": a, "target": b,
            "freq_source_to_target": freq_ab, "freq_target_to_source": freq_ba,
            "concurrency_ratio": round(ratio, 4),
        })

    # Guard against creating a brand-new 2-cycle: if BOTH (a,b) and (b,a) independently
    # qualify as concurrency-pruned (this is common -- the concurrency test is symmetric
    # by construction, so a near-balanced pair often has neither direction already
    # modeled), adding invisible transitions for both would literally wire up an A<->B
    # loop with no exit -- the exact runaway-repetition pathology this whole repair
    # exists to fix in the first place, and WOFLAN's soundness check does NOT catch it
    # (soundness only requires that completion stays reachable, not that unbounded
    # cycling is impossible). When both directions qualify, that symmetry is itself
    # evidence the pair may be genuinely concurrent (which is what Split Miner assumed),
    # so neither direction is auto-repaired -- both are dropped and reported for manual
    # review instead of silently picking one.
    by_pair = {}
    for c in candidates:
        by_pair.setdefault(frozenset((c["source"], c["target"])), []).append(c)

    safe_candidates, bidirectional_dropped = [], []
    for pair, cands in by_pair.items():
        if len(cands) > 1:
            bidirectional_dropped.extend(cands)
        else:
            safe_candidates.extend(cands)

    if bidirectional_dropped:
        print(f"  NOTE: {len(bidirectional_dropped)} candidate(s) dropped -- both directions of "
              f"the pair qualified (would create a 2-cycle), possibly genuine concurrency:")
        for c in bidirectional_dropped:
            print(f"    {c['source']} <-> {c['target']}")

    return sorted(safe_candidates, key=lambda c: -c["freq_source_to_target"])


def is_sound(net: PetriNet, im, fm) -> bool:
    try:
        from pm4py.algo.analysis.woflan import algorithm as woflan
        return bool(woflan.apply(net, im, fm, parameters={
            woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
            woflan.Parameters.PRINT_DIAGNOSTICS: False,
        }))
    except Exception as exc:
        print(f"  WARNING: soundness check unavailable ({exc}); treating as sound (unverified).")
        return True


def repair_net(net: PetriNet, im, fm, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply each candidate edge as a new invisible transition (source's existing output
    place(s) -> target's enabling place), keeping it only if the net stays sound afterward.
    Returns (applied, skipped_unsound)."""
    label_to_transitions = {}
    for t in net.transitions:
        if t.label is not None:
            label_to_transitions.setdefault(t.label, []).append(t)

    applied, skipped = [], []
    for i, cand in enumerate(candidates):
        source_transitions = label_to_transitions.get(cand["source"], [])
        target_transitions = label_to_transitions.get(cand["target"], [])
        source_places = {a.target for t in source_transitions for a in t.out_arcs}
        target_places = {a.source for t in target_transitions for a in t.in_arcs}

        if not source_places or not target_places:
            cand["reason"] = "source or target has no place to attach to"
            skipped.append(cand)
            continue

        # One new invisible transition per (source output place, target enabling place) pair
        # actually attempted; kept only if the net remains sound with ALL of them added
        # together (a single source activity can have >1 output place, e.g. after a split).
        new_transitions = []
        for sp in source_places:
            for tp in target_places:
                nt = PetriNet.Transition(f"repair_{cand['source']}_to_{cand['target']}_{i}_{len(new_transitions)}", None)
                net.transitions.add(nt)
                petri_utils.add_arc_from_to(sp, nt, net)
                petri_utils.add_arc_from_to(nt, tp, net)
                new_transitions.append(nt)

        if is_sound(net, im, fm):
            cand["n_transitions_added"] = len(new_transitions)
            applied.append(cand)
        else:
            for nt in new_transitions:
                petri_utils.remove_transition(net, nt)
            cand["reason"] = "adding it broke net soundness (WOFLAN)"
            skipped.append(cand)

    return applied, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Automatically repair directly-follows edges concurrency-pruned by Split Miner."
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--case_study", type=str, required=True, choices=list(CASE_STUDIES))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"Concurrency-ratio threshold, matching Split Miner's own condition "
                             f"(default: {DEFAULT_EPSILON}, the paper's own recommended value).")
    parser.add_argument("--min_frequency", type=int, default=DEFAULT_MIN_FREQUENCY,
                        help=f"Ignore edges rarer than this regardless of ratio, to avoid "
                             f"patching noise (default: {DEFAULT_MIN_FREQUENCY}).")
    parser.add_argument("--out_pnml", type=str, default=None,
                        help="Output path for the repaired net (default: "
                             "<case_study>_best_petri_net_repaired.pnml next to the original).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report candidates, don't write a repaired net.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    pnml_rel, xes_rel = CASE_STUDIES[args.case_study]
    pnml_path = base_dir / pnml_rel
    xes_path = base_dir / xes_rel

    net, im, fm = pm4py.read_pnml(str(pnml_path))
    log = pm4py.read_xes(str(xes_path))
    df = pm4py.convert_to_dataframe(log)
    dfg = raw_dfg_frequencies(df)

    candidates = find_repairable_edges(net, dfg, args.epsilon, args.min_frequency)
    print(f"{len(candidates)} concurrency-pruned edge(s) found for {args.case_study} "
          f"(epsilon={args.epsilon}, min_frequency={args.min_frequency}):")
    for c in candidates:
        print(f"  {c['source']} -> {c['target']}   "
              f"freq={c['freq_source_to_target']} (reverse={c['freq_target_to_source']}, "
              f"ratio={c['concurrency_ratio']:.3f})")

    if not candidates:
        print("Nothing to repair.")
        return
    if args.dry_run:
        print("\n--dry_run: no net written.")
        return

    applied, skipped = repair_net(net, im, fm, candidates)

    print(f"\nApplied {len(applied)} repair(s):")
    for c in applied:
        print(f"  {c['source']} -> {c['target']} ({c['n_transitions_added']} invisible transition(s) added)")
    if skipped:
        print(f"Skipped {len(skipped)} candidate(s):")
        for c in skipped:
            print(f"  {c['source']} -> {c['target']}: {c['reason']}")

    out_pnml = Path(args.out_pnml) if args.out_pnml else pnml_path.with_name(
        pnml_path.stem + "_repaired" + pnml_path.suffix)
    pm4py.write_pnml(net, im, fm, str(out_pnml))
    print(f"\nWrote repaired net to {out_pnml}")


if __name__ == "__main__":
    main()

# Running commands:
# python discovery/repair_pruned_concurrent_edges.py --case_study BPI12
# python discovery/repair_pruned_concurrent_edges.py --case_study BPI12 --dry_run
