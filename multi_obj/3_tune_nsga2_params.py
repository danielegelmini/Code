"""
Benchmarks NSGA-II against the exhaustive search across a grid of
(pop_size, n_generations) values, on a sample of cases, to help pick the
smallest NSGA-II configuration that still gives a Pareto front close to the
true one (found by the exhaustive method). NSGA-II calls the predictive
models once per generation, so for the small discrete search spaces typical
here (few dozen to a few hundred valid (activity, resource) pairs per case)
it can end up slower than just evaluating everything exhaustively, without
necessarily being more accurate.

For every sampled case it:
  1. computes the true Pareto front with exhaustive_pareto_search (used as
     ground truth and as the time baseline);
  2. for every (pop_size, n_generations) combination in the grid, runs
     nsga2_pareto_search and measures elapsed time plus two standard
     multi-objective quality indicators against the true front:
       - hypervolume ratio (HV(nsga2) / HV(exhaustive)): the area enclosed
         by each front relative to a shared reference point -- 1.0 means
         NSGA-II covers exactly as much of the objective space as the true
         front.
       - IGD (Inverted Generational Distance): average distance from each
         point of the true front to its closest NSGA-II point -- lower is
         better; it penalizes NSGA-II missing regions of the true front
         even when hypervolume looks fine.

Results are saved as one row per (case, pop_size, n_generations) to a CSV,
plus an aggregated summary (mean time/HV ratio/IGD per grid point,
compared against the exhaustive baseline) printed to stdout and saved
alongside it.
"""

import argparse
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from paretoset import paretoset
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from utils.get_features import load_case_study, get_case_study_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.recommendation_functions import (
    _build_valid_pairs,
    act_with_res_func,
    build_query_instances,
    exhaustive_pareto_search,
    next_possible_activities,
    nsga2_pareto_search,
)
from utils.transition_system import transition_system

warnings.filterwarnings("ignore")


def _default_forbidden_map():
    bpi17_forbidden = ["O_Accepted"]
    bac_forbidden = ["Network Adjustment Requested", "Back-Office Adjustment Requested"]
    return {
        "bpi17_before": bpi17_forbidden,
        "bpi17_after": bpi17_forbidden,
        "BPI12": ["O_ACCEPTED"],
        "BAC": bac_forbidden,
    }


def _front_xy(pareto_tuples):
    """(act, res, outcome, time, uncertainty) tuples -> (outcome, 1 - time,
    -uncertainty) points, all three maximized (so the same paretoset/HV/IGD
    machinery keeps working unchanged in 3-D)."""
    x = np.array([t[2] for t in pareto_tuples], dtype=float)
    y = 1.0 - np.array([t[3] for t in pareto_tuples], dtype=float)
    z = -np.array([t[4] for t in pareto_tuples], dtype=float)
    return np.column_stack([x, y, z])


def _true_front_points(vals):
    if len(vals) == 0:
        return vals
    mask = paretoset(vals, sense=["max", "max", "max"])
    return vals[mask]


def compute_quality(exhaustive_set, nsga2_set):
    """
    Returns (hv_ratio, igd, n_exhaustive_front, n_nsga2_front) comparing the
    NSGA-II front to the true (exhaustive) front for one case, or None if
    either search returned nothing usable.
    """
    if not exhaustive_set or not nsga2_set:
        return None

    ex_front = _true_front_points(_front_xy(exhaustive_set))
    ns_front = _true_front_points(_front_xy(nsga2_set))
    if len(ex_front) == 0 or len(ns_front) == 0:
        return None

    # Shared reference point (in the original maximize/maximize space),
    # slightly worse than the worst point of either front, so both
    # hypervolumes are computed against the same baseline and stay
    # comparable to each other.
    combined = np.vstack([ex_front, ns_front])
    ref_point_max_space = combined.min(axis=0) - 1e-6

    # pymoo's indicators assume minimization -> negate everything.
    F_ex = -ex_front
    F_ns = -ns_front
    ref_point = -ref_point_max_space

    hv_indicator = HV(ref_point=ref_point)
    hv_ex = hv_indicator(F_ex)
    hv_ns = hv_indicator(F_ns)
    hv_ratio = float(hv_ns / hv_ex) if hv_ex > 0 else float("nan")

    igd_val = float(IGD(F_ex)(F_ns))

    return hv_ratio, igd_val, len(ex_front), len(ns_front)


def sample_cases(case_sizes, n_cases, min_valid_pairs):
    """
    Picks up to n_cases case ids, spread across the distribution of search
    space sizes (number of valid (activity, resource) pairs), so the
    benchmark isn't dominated by only-small or only-large cases. Cases with
    fewer than min_valid_pairs valid pairs are dropped (their Pareto front
    is trivial and not informative for this comparison).
    """
    eligible = [(cid, n) for cid, n in case_sizes.items() if n >= min_valid_pairs]
    if not eligible:
        return []

    eligible.sort(key=lambda pair: pair[1])
    if len(eligible) <= n_cases:
        return [cid for cid, _ in eligible]

    # Evenly spaced indices over the size-sorted list -> spread across
    # small/medium/large search spaces instead of a plain random sample.
    idx = np.linspace(0, len(eligible) - 1, n_cases).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [eligible[i][0] for i in idx]


def run_benchmark(
    case_study: str,
    n_cases: int = 30,
    pop_sizes=(10, 20, 30, 50),
    n_generations_list=(3, 5, 7, 10),
    window_size: int = 5,
    min_valid_pairs: int = 5,
    random_state: int = 1234,
    output_dir: str = None,
):
    np.random.seed(random_state)
    random.seed(random_state)

    print(f"Loading data for {case_study}...")
    train_data, test_data, test_log = load_case_study(case_study)
    if case_study in {"BPI12"}:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data = convert_dtypes_bpi12(test_data, "experiment")
        test_log = convert_dtypes_bpi12(test_log, "experiment")

    (
        predictive_outcome_model,
        predictive_time_model,
        case_id_name,
        activity_column_name,
        resource_column_name,
        continuous_features,
        categorical_features,
        columns_to_remove,
    ) = get_case_study_features(case_study)

    print("Building transition system and maps...")
    transition_graph, _ = transition_system(
        train_data, case_id_name=case_id_name, activity_column_name=activity_column_name, window_size=window_size
    )
    act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)
    forbidden = set(_default_forbidden_map().get(case_study, []))
    query_instances_by_case = build_query_instances(test_data, case_id_name)
    unique_cases = pd.unique(test_data[case_id_name])

    print(f"Sizing search space for {len(unique_cases)} cases...")
    case_sizes = {}
    case_poss = {}
    for cid in tqdm.tqdm(unique_cases, desc="Sizing cases"):
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()
        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            continue
        valid_pairs = _build_valid_pairs(poss, act_with_res)
        if valid_pairs:
            case_sizes[cid] = len(valid_pairs)
            case_poss[cid] = poss

    sampled_cases = sample_cases(case_sizes, n_cases, min_valid_pairs)
    print(
        f"Sampled {len(sampled_cases)} cases out of {len(case_sizes)} eligible "
        f"(>= {min_valid_pairs} valid pairs), search space sizes: "
        f"min={min(case_sizes[c] for c in sampled_cases)}, "
        f"median={int(np.median([case_sizes[c] for c in sampled_cases]))}, "
        f"max={max(case_sizes[c] for c in sampled_cases)}"
    )

    rows = []
    for cid in tqdm.tqdm(sampled_cases, desc="Benchmarking cases"):
        query_instance = query_instances_by_case[cid]
        poss = case_poss[cid]
        n_valid_pairs = case_sizes[cid]

        t0 = time.time()
        exhaustive_set = exhaustive_pareto_search(
            query_instance, poss, predictive_outcome_model, predictive_time_model, act_with_res
        )
        exhaustive_time = time.time() - t0

        rows.append({
            "case_id": cid,
            "n_valid_pairs": n_valid_pairs,
            "method": "exhaustive",
            "pop_size": None,
            "n_generations": None,
            "time_s": exhaustive_time,
            "hv_ratio": 1.0,
            "igd": 0.0,
            "n_front_points": len(_true_front_points(_front_xy(exhaustive_set))) if exhaustive_set else 0,
        })

        for pop_size in pop_sizes:
            for n_generations in n_generations_list:
                t0 = time.time()
                nsga2_set = nsga2_pareto_search(
                    query_instance=query_instance,
                    possible_actions=poss,
                    act_with_res=act_with_res,
                    predictive_outcome_model=predictive_outcome_model,
                    predictive_time_model=predictive_time_model,
                    pop_size=pop_size,
                    n_generations=n_generations,
                    random_state=random_state,
                )
                nsga2_time = time.time() - t0

                quality = compute_quality(exhaustive_set, nsga2_set)
                hv_ratio, igd_val, _, n_front_points = quality if quality else (float("nan"), float("nan"), 0, 0)

                rows.append({
                    "case_id": cid,
                    "n_valid_pairs": n_valid_pairs,
                    "method": "nsga2",
                    "pop_size": pop_size,
                    "n_generations": n_generations,
                    "time_s": nsga2_time,
                    "hv_ratio": hv_ratio,
                    "igd": igd_val,
                    "n_front_points": n_front_points,
                })

    results_df = pd.DataFrame(rows)

    out_dir = Path(output_dir) if output_dir else Path(f"./case_studies/{case_study}")
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"nsga2_tuning_{case_study}_detail.csv"
    results_df.to_csv(detail_path, index=False)

    exhaustive_mean_time = results_df.loc[results_df["method"] == "exhaustive", "time_s"].mean()

    summary_df = (
        results_df[results_df["method"] == "nsga2"]
        .groupby(["pop_size", "n_generations"], as_index=False)
        .agg(
            mean_time_s=("time_s", "mean"),
            mean_hv_ratio=("hv_ratio", "mean"),
            mean_igd=("igd", "mean"),
            mean_front_points=("n_front_points", "mean"),
        )
        .sort_values(["mean_time_s"])
    )
    summary_df["speedup_vs_exhaustive"] = exhaustive_mean_time / summary_df["mean_time_s"]

    summary_path = out_dir / f"nsga2_tuning_{case_study}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY -- {case_study}")
    print("=" * 70)
    print(f"Exhaustive baseline: mean time = {exhaustive_mean_time:.4f}s over {len(sampled_cases)} cases")
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\nDetail saved to:  {detail_path}")
    print(f"Summary saved to: {summary_path}")

    return results_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark NSGA-II (pop_size x n_generations grid) against the exhaustive "
                     "Pareto search, to find the smallest/fastest NSGA-II configuration that still "
                     "reproduces a Pareto front close to the true one."
    )
    parser.add_argument("--case_study", type=str, required=True, help="Case study (e.g. 'BAC').")
    parser.add_argument("--n_cases", type=int, default=30, help="Number of cases to sample (default: 30).")
    parser.add_argument(
        "--pop_sizes", type=str, default="10,20,30,50",
        help="Comma-separated list of pop_size values to try (default: 10,20,30,50).",
    )
    parser.add_argument(
        "--n_generations", type=str, default="3,5,7,10",
        help="Comma-separated list of n_generations values to try (default: 3,5,7,10).",
    )
    parser.add_argument("--window_size", type=int, default=5, help="Transition system window size (default: 5).")
    parser.add_argument(
        "--min_valid_pairs", type=int, default=5,
        help="Skip cases with fewer than this many valid (activity, resource) pairs (default: 5).",
    )
    parser.add_argument("--random_state", type=int, default=1234, help="Random seed (default: 1234).")

    args = parser.parse_args()

    run_benchmark(
        case_study=args.case_study,
        n_cases=args.n_cases,
        pop_sizes=tuple(int(x) for x in args.pop_sizes.split(",")),
        n_generations_list=tuple(int(x) for x in args.n_generations.split(",")),
        window_size=args.window_size,
        min_valid_pairs=args.min_valid_pairs,
        random_state=args.random_state,
    )

# Example usage:
# python 3_tune_nsga2_params.py --case_study "BAC" --n_cases 30 --pop_sizes "70,85,100" --n_generations "1,2,3,4,5"
