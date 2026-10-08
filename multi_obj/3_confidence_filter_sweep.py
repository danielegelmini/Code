#!/usr/bin/env python3
"""
3_confidence_filter_sweep.py

Effect of the confidence filter of 3_run_experiment.py (exhaustive method) for
several values of its threshold gamma (symmetric: gamma_cls = gamma_reg = gamma).

For every test case it repeats exactly what compute_recommendations_top_k does
-- feasible (activity, resource) candidates from the transition system, minus the
forbidden activities; their predictions; P(candidate beats the no-recommendation
baseline) on the outcome and on the time -- but computes the two probabilities
ONCE and then applies every gamma to them. For each gamma it counts how many
candidates pass the filter, how large the 2-D Pareto front of the survivors is,
and so how many recommendations the case receives (min(k, front size): the
p-dispersion selection keeps the whole front when it has k points or fewer).

Writes, in this folder:
  - 3_confidence_filter_sweep.txt: per case study, one table with a row per
    gamma (candidates, share passing the filter, front size, share of the test
    cases with a recommendation at rank 1..k) and one table with the
    distribution of the number of recommendations per case;
  - 3_confidence_filter_sweep.csv: the same numbers, one row per case study and gamma.

When the recommendation files of step 3 exist for the same k, the number of
recommendations per case computed here for the gamma they were produced with is
checked against them (the gamma is not stored in those files: pass it with
--check_gamma, default 0.5).

Usage:
    python 3_confidence_filter_sweep.py
    python 3_confidence_filter_sweep.py --case_studies "BAC,BPI12" --gammas "0,0.5,0.55,0.6" --k 5
    python 3_confidence_filter_sweep.py --case_studies BAC --max_cases 50     (quick test)
"""

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from paretoset import paretoset

from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.get_features import load_case_study, get_case_study_features
from utils.setup_cache import get_transition_graph
from utils.recommendation_functions import (
    act_with_res_func,
    next_possible_activities,
    build_query_instances,
    build_no_recommendation_baseline_instances,
    _to_row_df,
    _build_valid_pairs,
    _evaluate_candidates,
    _compute_confidence_probabilities,
    _front_objective_matrix,
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent  # outputs are written here (multi_obj/), next to the script
CASE_ID_NAME = "case:concept:name"
DEFAULT_CASE_STUDIES = "BAC,BPI12,bpi17_before,bpi17_after"
DEFAULT_GAMMAS = "0.5,0.6,0.7,0.9"
BPI12_CASE_STUDIES = {"BPI12"}

# same forbidden activities as 3_run_experiment.py (_default_forbidden_map)
FORBIDDEN = {
    "bpi17_before": ["O_Accepted"],
    "bpi17_after": ["O_Accepted"],
    "BPI12_not_reordered": ["O_ACCEPTED"],
    "BPI12_not_reordered_sim": ["O_ACCEPTED"],
    "BPI12": ["O_ACCEPTED"],
    "BPI12_sim": ["O_ACCEPTED"],
    "BAC": ["Network Adjustment Requested", "Back-Office Adjustment Requested"],
}


def front_size(objs: np.ndarray) -> int:
    """Size of the 2-D Pareto front (max outcome, max 1 - time) of the given candidates, built as
    select_top_k_pareto_actions builds it (same objective matrix, same paretoset call)."""
    if len(objs) == 0:
        return 0
    # 7-tuples as exhaustive_pareto_search returns them; only outcome and time are read
    items = [(None, None, o, t, None, None, None) for o, t in objs[:, :2]]
    raw_vals, sense = _front_objective_matrix(items)
    return int(paretoset(raw_vals, sense=sense).sum())


def sweep_case_study(case_study: str, gammas: list, k: int, window_size: int, max_cases,
                     min_next_share: float = 0.01) -> pd.DataFrame:
    """One row per test case and gamma: candidates, survivors (both KPIs / outcome only / time
    only), Pareto front size and number of recommendations."""
    train_data, test_data, test_log = load_case_study(case_study)
    if case_study in BPI12_CASE_STUDIES:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data = convert_dtypes_bpi12(test_data, "experiment")
        test_log = convert_dtypes_bpi12(test_log, "experiment")

    (outcome_model, time_model, case_id_name, activity_column_name, resource_column_name,
     _, _, _) = get_case_study_features(case_study)
    transition_graph = get_transition_graph(case_study, train_data, case_id_name=case_id_name,
                                            activity_column_name=activity_column_name,
                                            window_size=window_size, rebuild=False, min_next_share=min_next_share)
    act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)
    forbidden = set(FORBIDDEN.get(case_study, []))
    query_instances = build_query_instances(test_data, case_id_name)
    baselines = build_no_recommendation_baseline_instances(query_instances)

    case_ids = pd.unique(test_data[case_id_name])
    if max_cases:
        case_ids = case_ids[:max_cases]

    rows = []
    for cid in tqdm.tqdm(case_ids, desc=case_study):
        trace_history = test_log.loc[test_log[case_id_name] == cid, activity_column_name].tolist()
        poss_all = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss_all if a not in forbidden]
        pairs = _build_valid_pairs(poss, act_with_res) if poss else []
        if not pairs:
            # why the case has no candidate at all (the same for every gamma)
            reason = ("no_next_activity" if not poss_all else "only_forbidden" if not poss else "no_resource")
            for g in gammas:
                rows.append({"case_id": str(cid), "gamma": g, "n_candidates": 0, "n_pass_outcome": 0,
                             "n_pass_time": 0, "n_pass": 0, "front_size": 0, "n_rec": 0,
                             "no_candidate_reason": reason})
            continue
        query_instance = _to_row_df(query_instances[cid])
        objs = _evaluate_candidates(pairs, query_instance, outcome_model, time_model)
        p_out, p_time = _compute_confidence_probabilities(pairs, baselines[cid], query_instance,
                                                          outcome_model, time_model)
        p_out, p_time = np.asarray(p_out), np.asarray(p_time)
        for g in gammas:
            keep_out, keep_time = p_out >= g, p_time >= g
            keep = keep_out & keep_time
            n_front = front_size(objs[keep])
            rows.append({"case_id": str(cid), "gamma": g, "n_candidates": len(pairs),
                         "n_pass_outcome": int(keep_out.sum()), "n_pass_time": int(keep_time.sum()),
                         "n_pass": int(keep.sum()), "front_size": n_front, "n_rec": min(k, n_front),
                         "no_candidate_reason": ""})
    return pd.DataFrame(rows)


def summarize(per_case: pd.DataFrame, case_study: str, k: int) -> pd.DataFrame:
    """One row per gamma. Shares of cases are over ALL the test cases; candidates and front size
    are averaged over the cases with at least one feasible candidate; the shares of candidates
    passing the filter are pooled (passing candidates / candidates, over all the cases)."""
    out = []
    for g, d in per_case.groupby("gamma", sort=True):
        with_cand = d[d["n_candidates"] > 0]
        n_cand = d["n_candidates"].sum()
        row = {
            "case_study": case_study,
            "gamma": g,
            "test_cases": len(d),
            "cases_with_candidates": len(with_cand),
            "mean_candidates": with_cand["n_candidates"].mean(),
            "mean_passing": with_cand["n_pass"].mean(),
            "pct_pass_outcome": 100 * d["n_pass_outcome"].sum() / n_cand if n_cand else np.nan,
            "pct_pass_time": 100 * d["n_pass_time"].sum() / n_cand if n_cand else np.nan,
            "pct_pass_both": 100 * d["n_pass"].sum() / n_cand if n_cand else np.nan,
            "mean_front_size": with_cand["front_size"].mean(),
            # cases with no recommendation at all, by cause: no candidate (transition system / forbidden
            # activities / resources), or candidates but none passing the filter
            "n_no_next_activity": int((d["no_candidate_reason"] == "no_next_activity").sum()),
            "n_only_forbidden": int((d["no_candidate_reason"] == "only_forbidden").sum()),
            "n_no_resource": int((d["no_candidate_reason"] == "no_resource").sum()),
            "n_none_passing": int(((d["n_candidates"] > 0) & (d["n_pass"] == 0)).sum()),
        }
        for r in range(1, k + 1):
            row[f"pct_cases_rank{r}"] = 100 * (d["n_rec"] >= r).mean()
        for n in range(0, k + 1):
            row[f"pct_cases_{n}_recs"] = 100 * (d["n_rec"] == n).mean()
        out.append(row)
    return pd.DataFrame(out)


def check_against_step3(per_case: pd.DataFrame, case_study: str, k: int, gamma: float) -> str:
    """Compare the number of recommendations per case at `gamma` with the step-3 recommendation files."""
    rec_dir = Path("case_studies") / case_study / "recommendations"
    counts = None
    for rank in range(1, k + 1):
        path = rec_dir / f"recommendations_{case_study}_exhaustive_top{rank}of{k}.csv"
        if not path.exists():
            return f"check skipped: {path} not found"
        rec = pd.read_csv(path, dtype=str)
        has = (rec["Next_activity"].notna() & rec["Next_resource"].notna()).astype(int)
        has.index = rec[CASE_ID_NAME].astype(str)
        counts = has if counts is None else counts.add(has, fill_value=0)
    mine = per_case.loc[np.isclose(per_case["gamma"], gamma)].set_index("case_id")["n_rec"]
    if mine.empty:
        return f"check skipped: gamma {gamma} not in the sweep"
    common = mine.index.intersection(counts.index)
    agree = (mine.loc[common] == counts.loc[common]).mean() if len(common) else np.nan
    return (f"check against the step-3 recommendation files (gamma {gamma}): number of recommendations equal "
            f"for {agree:.1%} of {len(common)} cases")


def format_tables(summary: pd.DataFrame, k: int, check: str, elapsed: float) -> list:
    cs = summary["case_study"].iloc[0]
    first = summary.iloc[0]
    lines = [f"{cs.upper()}", "-" * 100,
             f"Test cases: {int(first['test_cases'])} | with at least one feasible candidate: "
             f"{int(first['cases_with_candidates'])} | candidates per case (mean): {first['mean_candidates']:.1f}",
             f"{check} | computed in {elapsed / 60:.1f} min", ""]

    t1 = pd.DataFrame({
        "gamma": summary["gamma"].map("{:.2f}".format),
        "cand/case": summary["mean_candidates"].map("{:.1f}".format),
        "pass/case": summary["mean_passing"].map("{:.1f}".format),
        "% pass out": summary["pct_pass_outcome"].map("{:.1f}".format),
        "% pass time": summary["pct_pass_time"].map("{:.1f}".format),
        "% pass both": summary["pct_pass_both"].map("{:.1f}".format),
        "front/case": summary["mean_front_size"].map("{:.2f}".format),
        **{f"% rank {r}": summary[f"pct_cases_rank{r}"].map("{:.1f}".format) for r in range(1, k + 1)},
    })
    lines += ["(a) Candidates passing the filter and share of test cases with a recommendation at each rank",
              t1.to_string(index=False), ""]

    t2 = pd.DataFrame({
        "gamma": summary["gamma"].map("{:.2f}".format),
        **{f"{n} recs": summary[f"pct_cases_{n}_recs"].map("{:.1f}".format) for n in range(k, -1, -1)},
    })
    lines += ["(b) Distribution of the number of recommendations per test case (% of the test cases)",
              t2.to_string(index=False), ""]

    n_test = summary["test_cases"]
    no_rec = summary["n_no_next_activity"] + summary["n_only_forbidden"] + summary["n_no_resource"] + summary["n_none_passing"]
    t3 = pd.DataFrame({
        "gamma": summary["gamma"].map("{:.2f}".format),
        "no next act. (TS)": summary["n_no_next_activity"],
        "only forbidden": summary["n_only_forbidden"],
        "no resource": summary["n_no_resource"],
        "none passes filter": summary["n_none_passing"],
        "total no rec.": no_rec,
        "% of test": (100 * no_rec / n_test).map("{:.1f}".format),
    })
    lines += ["(c) Test cases with NO recommendation, by cause (number of cases)",
              "    no next act. (TS): the transition system knows no next activity, not even matching only the last",
              "    activity of the prefix; only forbidden: every next activity is forbidden; no resource: no resource",
              "    known for the next activities; none passes filter: candidates exist but none reaches gamma on both KPIs.",
              t3.to_string(index=False), "", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Effect of the confidence-filter threshold gamma on the recommendations.")
    parser.add_argument("--case_studies", type=str, default=DEFAULT_CASE_STUDIES,
                        help=f"Comma-separated case studies (default: {DEFAULT_CASE_STUDIES}).")
    parser.add_argument("--gammas", type=str, default=DEFAULT_GAMMAS,
                        help=f"Comma-separated thresholds, used for both KPIs (default: {DEFAULT_GAMMAS}). "
                             "0 = no filter.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--min_next_share", type=float, default=0.01,
                        help="Same as 3_run_experiment.py: rarer next activities of a window are not candidates.")
    parser.add_argument("--check_gamma", type=float, default=0.5,
                        help="gamma the step-3 recommendation files were produced with, for the consistency check.")
    parser.add_argument("--max_cases", type=int, default=None, help="Only the first N test cases (quick test).")
    parser.add_argument("--out_txt", type=str, default=str(HERE / "3_confidence_filter_sweep.txt"))
    parser.add_argument("--out_csv", type=str, default=str(HERE / "3_confidence_filter_sweep.csv"))
    args = parser.parse_args()

    gammas = sorted({float(g) for g in args.gammas.split(",") if g.strip()})
    case_studies = [c.strip() for c in args.case_studies.split(",") if c.strip()]
    lines = ["=" * 100, "CONFIDENCE FILTER SWEEP (exhaustive method, gamma_cls = gamma_reg = gamma)",
             f"Written by 3_confidence_filter_sweep.py - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"k = {args.k} | window size = {args.window_size} | min next-activity share = {args.min_next_share:g} | "
             f"gammas = {', '.join(f'{g:g}' for g in gammas)}"
             + (f" | ONLY THE FIRST {args.max_cases} TEST CASES" if args.max_cases else ""),
             "=" * 100,
             "cand/case   : feasible (activity, resource) candidates per case (transition system, forbidden activities removed),",
             "              averaged over the cases with at least one candidate; gamma = 0 means no filter.",
             "pass/case   : candidates per case passing the filter on both KPIs.",
             "% pass ...  : share of all the candidates with P(beats the no-recommendation baseline) >= gamma on the",
             "              outcome only, on the time only, and on both (the filter).",
             "front/case  : size of the 2-D Pareto front (outcome, 1 - time) of the surviving candidates.",
             "% rank r    : share of ALL the test cases with a recommendation at rank r, i.e. with at least r",
             "              recommendations (min(k, front size)): the cases simulated at rank r in step 4.",
             "", ""]
    summaries = []
    for cs in case_studies:
        t0 = time.time()
        per_case = sweep_case_study(cs, gammas, args.k, args.window_size, args.max_cases, args.min_next_share)
        summary = summarize(per_case, cs, args.k)
        check = check_against_step3(per_case, cs, args.k, args.check_gamma)
        print(check)
        lines += format_tables(summary, args.k, check, time.time() - t0)
        summaries.append(summary)
        # written after every case study, so a long run keeps what is already computed
        Path(args.out_txt).write_text("\n".join(lines), encoding="utf-8")
        pd.concat(summaries, ignore_index=True).to_csv(args.out_csv, index=False)

    print("\n".join(lines))
    print(f"Saved {args.out_txt} and {args.out_csv}")


if __name__ == "__main__":
    main()
