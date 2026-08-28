"""
Reports the distribution of the search space size (number of valid
(activity, resource) pairs, i.e. what exhaustive_pareto_search/
nsga2_pareto_search actually evaluate) across ALL cases of one or more case
studies -- not just the sample used by 3_tune_nsga2_params.py's benchmark.
Confirms whether the small search spaces seen there hold for the whole
dataset, not just the 30 sampled cases.

Only does the sizing step (next_possible_activities + _build_valid_pairs),
no model calls -- fast even over thousands of cases.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import tqdm

from utils.get_features import load_case_study, get_case_study_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.recommendation_functions import _build_valid_pairs, act_with_res_func, next_possible_activities
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


def compute_search_space_sizes(case_study: str, window_size: int = 5):
    """Returns (sizes, n_cases_no_options, n_total_cases) over every case in the test set."""
    train_data, test_data, test_log = load_case_study(case_study)
    if case_study in {"BPI12"}:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data = convert_dtypes_bpi12(test_data, "experiment")
        test_log = convert_dtypes_bpi12(test_log, "experiment")

    (_, _, case_id_name, activity_column_name, resource_column_name, _, _, _) = get_case_study_features(case_study)

    transition_graph, _ = transition_system(
        train_data, case_id_name=case_id_name, activity_column_name=activity_column_name, window_size=window_size
    )
    act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)
    forbidden = set(_default_forbidden_map().get(case_study, []))
    unique_cases = pd.unique(test_data[case_id_name])

    sizes = []
    n_no_options = 0
    for cid in tqdm.tqdm(unique_cases, desc=f"Sizing {case_study}"):
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()
        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            n_no_options += 1
            continue
        sizes.append(len(_build_valid_pairs(poss, act_with_res)))

    return np.array(sizes), n_no_options, len(unique_cases)


def report(case_study: str, window_size: int = 5):
    sizes, n_no_options, n_total = compute_search_space_sizes(case_study, window_size)
    if len(sizes) == 0:
        print(f"{case_study}: no case has a valid search space (all {n_total} cases have no next activity).")
        return None

    return {
        "case_study": case_study,
        "n_total_cases": n_total,
        "n_cases_with_options": len(sizes),
        "n_cases_no_options": n_no_options,
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "std": float(np.std(sizes)),
        "min": int(np.min(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "p99": float(np.percentile(sizes, 99)),
        "max": int(np.max(sizes)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report the full distribution of search space size (valid (activity, resource) "
                     "pairs per case) across every case in the test set, for one or more case studies."
    )
    parser.add_argument(
        "--case_study", type=str, nargs="+",
        default=["BAC", "BPI12", "bpi17_before", "bpi17_after"],
        help="One or more case studies (default: all four).",
    )
    parser.add_argument("--window_size", type=int, default=5, help="Transition system window size (default: 5).")
    args = parser.parse_args()

    all_stats = []
    for cs in args.case_study:
        print(f"\n{'=' * 70}\n{cs}\n{'=' * 70}")
        stats = report(cs, args.window_size)
        if stats:
            all_stats.append(stats)

    summary_df = pd.DataFrame(all_stats)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 70)
    print("SEARCH SPACE SIZE SUMMARY (valid (activity, resource) pairs per case, ALL cases)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    out_path = "search_space_stats.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

# Example usage:
# python 3_search_space_stats.py --case_study "BAC" "BPI12" "bpi17_before" "bpi17_after"
# python 3_search_space_stats.py --case_study "BAC"
