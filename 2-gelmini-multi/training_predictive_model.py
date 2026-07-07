# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Adjust these imports based on where convert_dtypes_bpi12 actually lives
from utils.get_features import get_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.predictive_models_functions import train_ml_model

end_date_name = 'time:timestamp'
start_date_name = 'start:timestamp'

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train a Machine Learning Model for a specific Case Study.")
    
    # Required arguments
    parser.add_argument("--case_study", type=str, required=True, 
                        help="Name of the case study folder/file (e.g., 'BPI12')")
    
    # Optional hyperparameters with your default values
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                        help="Learning rate for the model (default: 0.1)")
    parser.add_argument("--depth", type=int, default=9, 
                        help="Depth of the trees (default: 9)")
    parser.add_argument("--n_iterations", type=int, default=2000, 
                        help="Number of iterations/estimators (default: 2000)")
    parser.add_argument("--learning_rate_min", type=float, default=None,
                        help="Minimum learning rate for random search range")
    parser.add_argument("--learning_rate_max", type=float, default=None,
                        help="Maximum learning rate for random search range")
    parser.add_argument("--depth_min", type=int, default=None,
                        help="Minimum depth for random search range")
    parser.add_argument("--depth_max", type=int, default=None,
                        help="Maximum depth for random search range")
    parser.add_argument("--n_iterations_min", type=int, default=None,
                        help="Minimum number of iterations for random search range")
    parser.add_argument("--n_iterations_max", type=int, default=None,
                        help="Maximum number of iterations for random search range")
    parser.add_argument("--random_search_trials", type=int, default=10,
                        help="Number of random hyperparameter combinations to evaluate during search")

    
    args = parser.parse_args()
    case_study = args.case_study
    learning_rate = args.learning_rate
    depth = args.depth
    n_iterations = args.n_iterations
    learning_rate_min = args.learning_rate_min
    learning_rate_max = args.learning_rate_max
    depth_min = args.depth_min
    depth_max = args.depth_max
    n_iterations_min = args.n_iterations_min
    n_iterations_max = args.n_iterations_max
    random_search_trials = args.random_search_trials

    print("\n" + "="*50)
    print(f" >>> LOADING DATASETS FOR {case_study.upper()} <<< ")
    print("="*50)
    
    data_dir = Path(f"./case_studies/{case_study}")

    # Dynamic Feature Fetching
    case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove = get_features(case_study)

    # Loading datasets safely
    train_data = pd.read_csv(data_dir / "train_data.csv", parse_dates=[end_date_name, start_date_name])
    test_data = pd.read_csv(data_dir / "test_data.csv", parse_dates=[end_date_name, start_date_name])
    test_log = pd.read_csv(data_dir / "test_log.csv", parse_dates=[end_date_name, start_date_name])
    test_log_last = pd.read_csv(data_dir / "test_log_with_last_act.csv", parse_dates=[end_date_name, start_date_name])

    # Condition-based data type conversion
    if case_study == "BPI12":
        print("\nApplying BPI12 specific data type conversions...")
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data  = convert_dtypes_bpi12(test_data, "experiment")
        test_log  = convert_dtypes_bpi12(test_log, "experiment")
        test_log_last  = convert_dtypes_bpi12(test_log_last, "experiment")

    # Create record payload
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_study": case_study,
        "learning_rate": learning_rate,
        "depth": depth,
        "n_iterations": n_iterations,
    }
    df_record = pd.DataFrame([record])

    output_dir = Path(f"./case_studies/{case_study}/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "params.csv"
    df_record.to_csv(results_file, mode='w', header=True, index=False)

    print(train_data.columns.tolist())

    print("\n" + "="*50)
    print(f" >>> TRAINING ML MODEL <<< ")
    print(f" Parameters: \n - Learning Rate: {learning_rate}\n - Depth: {depth}\n - Iterations: {n_iterations}")
    if learning_rate_min is not None and learning_rate_max is not None:
        print(f" - Learning rate range: [{learning_rate_min}, {learning_rate_max}]")
    if depth_min is not None and depth_max is not None:
        print(f" - Depth range: [{depth_min}, {depth_max}]")
    if n_iterations_min is not None and n_iterations_max is not None:
        print(f" - Iterations range: [{n_iterations_min}, {n_iterations_max}]")
    print(f" - Random search trials: {random_search_trials}")
    print("="*50)
    
    # Run training pipeline
    train_ml_model(
        train_data=train_data, 
        test_data=test_data, 
        case_id_name=case_id_name, 
        outcome_name='outcome', 
        columns_to_remove=columns_to_remove, 
        continuous_features=continuous_features, 
        categorical_features=categorical_features, 
        learning_rate=learning_rate, 
        depth=depth, 
        n_iterations=n_iterations, 
        case_study=case_study,
        learning_rate_min=learning_rate_min,
        learning_rate_max=learning_rate_max,
        depth_min=depth_min,
        depth_max=depth_max,
        n_iterations_min=n_iterations_min,
        n_iterations_max=n_iterations_max,
        random_search_trials=random_search_trials,
    )
    
    print("\n" + "*"*50)
    print(" TRAINING PROCESS COMPLETE! ")
    print("*"*50 + "\n")

if __name__ == "__main__":
    main()

# Running command:
# python training_predictive_model.py --case_study "bpi12" --learning_rate 0.5 --depth 9 --n_iterations 2000

# python training_predictive_model.py --case_study "BPI12" --learning_rate_min 0.01 --learning_rate_max 0.3 --depth_min 5 --depth_max 10 --n_iterations_min 1000 --n_iterations_max 3000 --random_search_trials 20