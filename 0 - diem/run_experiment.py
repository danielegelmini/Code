import pandas as pd
import numpy as np
import argparse
import joblib
import pickle
import os
import sys
import time
from typing import Dict, Any

# Local imports
from utils.recommendation_functions import (
    exhaustive_recommendations,
    genetic_recommendations,
    act_with_res_func,
    next_possible_activities,
    build_query_instances,
    create_DiCE_model)

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
outcome_name = "outcome_sigmoid_mm"
cols_to_vary = ["NEXT_ACTIVITY", "NEXT_RESOURCE"]


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
    num_cfes: int | None = None,
    window_size: int = 5,
    reduced_threshold: float = 0.05,
) -> Dict[Any, Any]:
    """
    Run an experiment for a given case study and method.

    Parameters
    ----------
    case_study : str
        Name of the case study (e.g., "BAC", "bpi12", "bpi17_before", "bpi17_after").

    method : str
        "genetic"  -> DiCE-based genetic approach (counterfactual-based recommendation).
        "exhaustive" -> Exhaustive approach (prediction-based recommendation).

    num_cfes : int | None
        Number of counterfactual explanations (only used for 'genetic').

    window_size : int
        Window size for the transition system.

    seed : int
        Random seed.
        
    reduced_threshold : float
        Reduced percentage for outcome prediction.

    Returns
    -------
    dict
        Recommendations dictionary (structure depends on the called method).
    """

    np.random.seed(1234)
    method = method.lower()
    reduced_percentage = 1 - reduced_threshold

    if method not in {"genetic", "exhaustive"}:
        raise ValueError("method must be either 'genetic' or 'exhaustive'.")

    if method == "genetic" and num_cfes is None:
        raise ValueError("num_cfes must be provided when method == 'genetic'.")

    print(
        f"Running {method} approach | "
        f"case study: {case_study} | "
        f"WINDOW_SIZE: {window_size}"
        f"Reduced threshold: {reduced_threshold}"
        + (f" | NUM_CFEs: {num_cfes}" if method == "genetic" else "")
    )

    t0 = time.time()

    # -------------------------
    # Load and preprocess data
    # -------------------------
    print("Loading data...")
    train_data, test_data, test_log = load_case_study(case_study)

    # -------------------------
    # Features and models
    # -------------------------
    print("Getting features...")
    (
        predictive_model,
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
    query_instances_by_case = build_query_instances_dice(
            test_data, case_id_name
        )  # Using test data with last row only

    # -------------------------
    # Generate recommendations
    # -------------------------
    if method == "genetic":
        print("Building explainer...")
        explainer = create_DiCE_model(
            train_data,
            continuous_features,
            categorical_features,
            outcome_name,
            method,  # 'genetic'
            predictive_model,
            columns_to_remove=columns_to_remove,
        )

        print("Generating recommendations with DiCE (genetic)...")
        recommendations = genetic_recommendations(
            case_study,
            test_log,
            test_data,
            predictive_model,
            method,  # dice_method: 'genetic'
            explainer,
            transition_graph,
            reduced_threshold,
            num_cfes,
            window_size,
            cols_to_vary,
            case_id_name,
            activity_column_name,
            outcome_name,
            forbidden_map,
            act_with_res,
            query_instances_by_case,
        )

    else:  # exhaustive baseline
        print("Generating recommendations with exhaustive baseline...")
        recommendations = exhaustive_recommendations(
            test_log,
            case_id_name,
            activity_column_name,
            case_study,
            transition_graph,
            window_size,
            forbidden_map,
            predictive_model,
            act_with_res,
            query_instances_by_case,
        )

    # -------------------------
    # Save results
    # -------------------------
    save_path = f"./case_studies/{case_study}"
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(
        save_path, f"recommendations_{case_study}_{method}.pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(recommendations, f)

    elapsed = time.time() - t0
    print(f"Saved results to {filename}")
    print(f"Done in {elapsed:.2f}s")

    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an experiment (genetic or exhaustive) with specified parameters."
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
        choices=["genetic", "exhaustive"],
        help="Specify the method: 'genetic' or 'exhaustive'.",
    )
    parser.add_argument(
        "--num_cfes",
        type=int,
        default=None,
        help="Number of counterfactual explanations (required for method='genetic').",
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

    args = parser.parse_args()

    run_experiment(
        case_study=args.case_study,
        method=args.method,
        num_cfes=args.num_cfes,
        window_size=args.window_size,
        reduced_threshold=args.reduced_threshold
    )

# FOR RUNNING EXPERIMENT:
# python run_experiment.py --case_study "BAC" --method "genetic" --num_cfes 5 --window_size 5 --reduced_threshold 0.05
