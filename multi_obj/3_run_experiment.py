import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
import time
from typing import Dict, Any, Tuple, List, Optional

# Local imports
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.recommendation_functions import (
    compute_recommendations_top_k,
    act_with_res_func,
    next_possible_activities,
    build_query_instances,
    build_no_recommendation_baseline_instances)

from utils.get_features import load_case_study, get_case_study_features
from utils.setup_cache import get_transition_graph

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

sys.path.append("./src")

case_id_name = "case:concept:name"
activity_column_name = "concept:name"
end_date_name = "time:timestamp"
start_date_name = "start:timestamp"
resource_column_name = "org:resource"
outcome_name = "outcome"


def _default_forbidden_map() -> Dict[str, list[str]]:
    bpi17_forbidden = ["O_Accepted"]
    bac_forbidden = ["Network Adjustment Requested", "Back-Office Adjustment Requested"]

    return {
        "bpi17_before": bpi17_forbidden,
        "bpi17_after": bpi17_forbidden,
        "BPI12_not_reordered": ["O_ACCEPTED"],
        "BPI12_not_reordered_sim": ["O_ACCEPTED"],
        "BPI12": ["O_ACCEPTED"],
        "BPI12_sim": ["O_ACCEPTED"],
        "BAC": bac_forbidden,
    }

def run_experiment_top_k(
    case_study: str,
    method: Optional[str] = None,
    window_size: int = 5,
    pop_size: int = 20,
    n_generations: int = 15,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    random_state: Optional[int] = 1234,
    k: int = 5,
    rebuild_cache: bool = False,
    gamma_cls: float = 0.5,
    gamma_reg: float = 0.5,
) -> Dict[str, List[Dict[Any, Any]]]:
    """
    Same as run_experiment, but instead of a single best (activity, resource)
    recommendation per case, it selects up to k diverse Pareto-optimal pairs
    per case (compute_recommendations_top_k, max-min / p-dispersion
    selection) and saves k separate recommendation files instead of one.

    The expensive setup (loading data, transition system, activity->resources
    map, query instances) depends only on case_study, not on the method --
    so it's done ONCE here and reused. By default (method left as None) only
    the exhaustive search runs; NSGA-II is parked "until further notice" (it is
    consistently slower on these small candidate sets). Pass method="nsga2" to
    still run it.

    Parameters
    ----------
    (same as run_experiment)

    method : str or None
        "exhaustive" or "nsga2" to run only that method, or None (default)
        to run both, one after another.

    k : int
        Number of diverse recommendations to select per case. Defaults to 5.

    Returns
    -------
    dict of str to list of dict
        {method_name: recommendations_list}, one entry per method actually
        run. Each recommendations_list has length k; its j-th dict is the
        recommendations dictionary {case_id: (next_activity, next_resource)}
        for the j-th selected pair.
    """

    np.random.seed(random_state if random_state is not None else 1234)

    if method is None:
        methods_to_run = ["exhaustive"]
    else:
        method = method.lower()
        if method not in {"exhaustive", "nsga2"}:
            raise ValueError("method must be either 'nsga2' or 'exhaustive'.")
        methods_to_run = [method]

    t0 = time.time()

    # -------------------------
    # Load and preprocess data (once, shared across methods)
    # -------------------------
    print("Loading data...")
    train_data, test_data, test_log = load_case_study(case_study)

    if case_study in {"BPI12_not_reordered", "BPI12_not_reordered_sim", "BPI12", "BPI12_sim"}:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data  = convert_dtypes_bpi12(test_data, "experiment")
        test_log  = convert_dtypes_bpi12(test_log, "experiment")

    # -------------------------
    # Features and models
    # -------------------------
    print("Getting features...")
    (
        predictive_outcome_model,
        predictive_time_model,
        case_id_name_local,
        activity_column_name_local,
        resource_column_name_local,
        continuous_features,
        categorical_features,
        columns_to_remove,
    ) = get_case_study_features(case_study)

    # Keep local overrides in case they differ per case study
    global case_id_name, activity_column_name, resource_column_name
    case_id_name = case_id_name_local
    activity_column_name = activity_column_name_local
    resource_column_name = resource_column_name_local

    # -------------------------
    # Transition system (cached on disk per case study + window size)
    # -------------------------
    transition_graph = get_transition_graph(
        case_study,
        train_data,
        case_id_name=case_id_name,
        activity_column_name=activity_column_name,
        window_size=window_size,
        rebuild=rebuild_cache,
    )

    # -------------------------
    # Activity-resource map
    # -------------------------
    print("Building activity -> resources map...")
    act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)

    # -------------------------
    # Forbidden activities map
    # -------------------------
    forbidden_map = _default_forbidden_map()

    # -------------------------
    # Prepare query instances + confidence-as-KPI baseline instances
    # -------------------------
    print("Preparing query instances...")
    query_instances_by_case = build_query_instances(
            test_data, case_id_name
        )  # Using test data with last row only

    # The "no recommendation" baseline for the confidence-as-KPI filter (only
    # used for method="exhaustive"): the SAME query instance the candidates are
    # evaluated on, with NEXT_ACTIVITY/NEXT_RESOURCE set to NO_NEXT_TOKEN (see
    # build_no_recommendation_baseline_instances). It exists for every case,
    # including traces with a single event.
    baseline_instances_by_case = build_no_recommendation_baseline_instances(query_instances_by_case)

    print(f"Setup done in {time.time() - t0:.2f}s. Running: {', '.join(methods_to_run)}\n")

    # -------------------------
    # Generate top-k recommendations, one method after another
    # -------------------------
    save_path = f"./case_studies/{case_study}/recommendations"
    os.makedirs(save_path, exist_ok=True)

    results_by_method: Dict[str, List[Dict[Any, Any]]] = {}
    for current_method in methods_to_run:
        method_t0 = time.time()
        print(
            f"Running top-{k} {current_method} approach | "
            f"case study: {case_study} | "
            f"WINDOW_SIZE: {window_size}"
            + (f" | pop_size: {pop_size} | n_generations: {n_generations}" if current_method == "nsga2" else "")
        )
        print(f"Generating top-{k} recommendations...")
        recommendations_list, _, _ = compute_recommendations_top_k(
            test_log=test_log,
            test_data=test_data,
            case_study=case_study,
            case_id_name=case_id_name,
            activity_column_name=activity_column_name,
            transition_graph=transition_graph,
            window_size=window_size,
            forbidden_map=forbidden_map,
            predictive_outcome_model=predictive_outcome_model,
            predictive_time_model=predictive_time_model,
            act_with_res=act_with_res,
            query_instances_by_case=query_instances_by_case,
            method=current_method,
            pop_size=pop_size,
            n_generations=n_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            random_state=random_state,
            k=k,
            baseline_instances_by_case=baseline_instances_by_case if current_method == "exhaustive" else None,
            gamma_cls=gamma_cls,
            gamma_reg=gamma_reg,
        )

        for rank, recommendations in enumerate(recommendations_list, start=1):
            filename = os.path.join(
                save_path, f"recommendations_{case_study}_{current_method}_top{rank}of{k}.csv"
            )
            rec_df = pd.DataFrame.from_dict(
                recommendations, orient="index", columns=["Next_activity", "Next_resource"]
            ).reset_index().rename(columns={"index": "case:concept:name"})
            rec_df.to_csv(filename, index=False)
            print(f"Saved rank {rank}/{k} results to {filename}")

        results_by_method[current_method] = recommendations_list
        print(f"{current_method} done in {time.time() - method_t0:.2f}s\n")

    print(f"All methods done in {time.time() - t0:.2f}s total")

    return results_by_method


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an experiment (genetic/nsga2 or exhaustive) with specified parameters."
    )
    parser.add_argument(
        "--case_study",
        type=str,
        required=True,
        help="Specify the case study (e.g. 'BAC').",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["nsga2", "exhaustive"],
        help="Specify the method: 'nsga2' (NSGA-II, pymoo) or 'exhaustive'. "
             "If omitted, only 'exhaustive' runs.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Window size for the transition system (default: 5).",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=50,
        help="NSGA-II population size (only used for method='nsga2', default: 50).",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=10,
        help="NSGA-II number of generations (only used for method='nsga2', default: 10).",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.9,
        help="NSGA-II crossover probability (only used for method='nsga2', default: 0.9).",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.3,
        help="NSGA-II mutation probability (only used for method='nsga2', default: 0.3).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top-k recommendations to select (only used for run_experiment_top_k, default: 5).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=1234,
        help="Random seed for numpy and NSGA-II (default: 1234).",
    )
    parser.add_argument(
        "--rebuild-cache",
        dest="rebuild_cache",
        action="store_true",
        help="Force recomputing the (cached) transition system instead of loading it.",
    )
    parser.add_argument(
        "--gamma_cls",
        type=float,
        default=0.5,
        help="Confidence-as-KPI threshold (method='exhaustive' only): minimum "
             "required P(candidate outcome beats the no-recommendation baseline) "
             "for a candidate to survive the pre-Pareto filter (default: 0.5).",
    )
    parser.add_argument(
        "--gamma_reg",
        type=float,
        default=0.5,
        help="Confidence-as-KPI threshold (method='exhaustive' only): minimum "
             "required P(candidate remaining time beats the no-recommendation "
             "baseline) for a candidate to survive the pre-Pareto filter (default: 0.5).",
    )

    args = parser.parse_args()

    run_experiment_top_k(
        case_study=args.case_study,
        method=args.method,
        window_size=args.window_size,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        random_state=args.random_state,
        k=args.k,
        rebuild_cache=args.rebuild_cache,
        gamma_cls=args.gamma_cls,
        gamma_reg=args.gamma_reg,
    )

# FOR RUNNING EXPERIMENT:
# case_study: "BAC", "BPI12", "bpi17_before", "bpi17_after"

# default: only "exhaustive" (NSGA-II parked until further notice)
# python 3_run_experiment.py --case_study "BAC" --window_size 5 --k 5

# example method: only "nsga2"
# python 3_run_experiment.py --case_study "BAC" --method "nsga2" --window_size 5 --pop_size 50 --n_generations 10 --k 5

# example method: only "exhaustive"
# python 3_run_experiment.py --case_study "BPI12" --method "exhaustive" --window_size 5 --k 5