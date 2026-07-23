import pandas as pd
import numpy as np
import argparse
import joblib
import pickle
import os
import sys
import time
from typing import Dict, Any, Tuple, List, Optional

# Local imports
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.recommendation_functions import (
    compute_recommendations,
    act_with_res_func,
    next_possible_activities,
    build_query_instances)

from utils.get_features import load_case_study, get_case_study_features
from utils.transition_system import transition_system

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

NSGA2_METHOD_ALIASES = {"genetic", "nsga2"}


def _default_forbidden_map() -> Dict[str, list[str]]:
    bpi17_forbidden = ["O_Accepted"]
    bac_forbidden = ["Network Adjustment Requested", "Back-Office Adjustment Requested"]

    return {
        "bpi17_before": bpi17_forbidden,
        "bpi17_after": bpi17_forbidden,
        "bpi12": ["O_ACCEPTED"],
        "BAC": bac_forbidden,
    }


def run_experiment(
    case_study: str,
    method: str,
    window_size: int = 5,
    reduced_threshold: float = 0.05,
    pop_size: int = 20,
    n_generations: int = 15,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    random_state: Optional[int] = 1234,
) -> Dict[Any, Any]:
    """
    Run an experiment for a given case study and method.

    Parameters
    ----------
    case_study : str
        Name of the case study (e.g., "BAC", "bpi12", "bpi17_before", "bpi17_after").

    method : str
        "genetic" / "nsga2" -> NSGA-II based approach (pymoo, multi-objective
            Pareto search over NEXT_ACTIVITY/NEXT_RESOURCE).
        "exhaustive" -> Exhaustive approach (prediction-based recommendation).

    window_size : int
        Window size for the transition system.

    reduced_threshold : float
        Reduced percentage for outcome prediction (only used to report the
        experiment configuration; the NSGA-II objectives already target
        max outcome / min time directly).

    pop_size : int
        NSGA-II population size (only used for 'genetic'/'nsga2').

    n_generations : int
        Number of NSGA-II generations (only used for 'genetic'/'nsga2').

    crossover_rate : float
        NSGA-II crossover probability (only used for 'genetic'/'nsga2').

    mutation_rate : float
        NSGA-II mutation probability (only used for 'genetic'/'nsga2').

    random_state : int | None
        Random seed used both for numpy and for the NSGA-II run (reproducibility).

    Returns
    -------
    dict
        Recommendations dictionary {case_id: (next_activity, next_resource)}.
    """

    np.random.seed(random_state if random_state is not None else 1234)
    method = method.lower()
    reduced_percentage = 1 - reduced_threshold

    if method not in {"exhaustive", *NSGA2_METHOD_ALIASES}:
        raise ValueError("method must be either 'genetic' (alias 'nsga2') or 'exhaustive'.")

    is_nsga2 = method in NSGA2_METHOD_ALIASES

    print(
        f"Running {'nsga2' if is_nsga2 else method} approach | "
        f"case study: {case_study} | "
        f"WINDOW_SIZE: {window_size} | "
        f"Reduced threshold: {reduced_threshold}"
        + (f" | pop_size: {pop_size} | n_generations: {n_generations}" if is_nsga2 else "")
    )

    t0 = time.time()

    # -------------------------
    # Load and preprocess data
    # -------------------------
    print("Loading data...")
    train_data, test_data, test_log = load_case_study(case_study)

    if case_study.lower() in {"bpi12"}:
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
    # Transition system
    # -------------------------
    print("Building transition system...")
    transition_graph, ts_with_freq = transition_system(
        train_data,
        case_id_name=case_id_name,
        activity_column_name=activity_column_name,
        window_size=window_size)

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
    # Prepare query instances
    # -------------------------
    print("Preparing query instances...")
    query_instances_by_case = build_query_instances(
            test_data, case_id_name
        )  # Using test data with last row only

    # -------------------------
    # Generate recommendations
    # -------------------------
    print("Generating recommendations...")
    recommendations = compute_recommendations(
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
        method=("nsga2" if is_nsga2 else "exhaustive"),
        pop_size=pop_size,
        n_generations=n_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        random_state=random_state,
    )

    # -------------------------
    # Save results
    # -------------------------
    save_path = f"./case_studies/{case_study}"
    os.makedirs(save_path, exist_ok=True)
    method_tag = "nsga2" if is_nsga2 else method
    filename = os.path.join(
        save_path, f"recommendations_{case_study}_{method_tag}.pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(recommendations, f)

    elapsed = time.time() - t0
    print(f"Saved results to {filename}")
    print(f"Done in {elapsed:.2f}s")

    return recommendations


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
        required=True,
        choices=["genetic", "nsga2", "exhaustive"],
        help="Specify the method: 'genetic'/'nsga2' (NSGA-II, pymoo) or 'exhaustive'.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Window size for the transition system (default: 5).",
    )
    parser.add_argument(
        "--reduced_threshold",
        type=float,
        default=0.05,
        help="Reduced threshold for predicted outcome (default: 0.05).",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=20,
        help="NSGA-II population size (only used for method='genetic'/'nsga2', default: 20).",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=15,
        help="NSGA-II number of generations (only used for method='genetic'/'nsga2', default: 15).",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.9,
        help="NSGA-II crossover probability (only used for method='genetic'/'nsga2', default: 0.9).",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.3,
        help="NSGA-II mutation probability (only used for method='genetic'/'nsga2', default: 0.3).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=1234,
        help="Random seed for numpy and NSGA-II (default: 1234).",
    )

    args = parser.parse_args()

    run_experiment(
        case_study=args.case_study,
        method=args.method,
        window_size=args.window_size,
        reduced_threshold=args.reduced_threshold,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        random_state=args.random_state,
    )

# FOR RUNNING EXPERIMENT:
# case_study: "BAC", "BPI12", "bpi17_before", "bpi17_after"

# example method: "genetic"
# python run_experiment.py --case_study "BAC" --method "nsga2" --window_size 5 --reduced_threshold 0.05 --pop_size 50 --n_generations 10
# example method: "exhaustive"
# python run_experiment.py --case_study "BPI12" --method "exhaustive" --window_size 5 --reduced_threshold 0.05