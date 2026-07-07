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
    
    args = parser.parse_args()
    case_study = args.case_study
    learning_rate = args.learning_rate
    depth = args.depth
    n_iterations = args.n_iterations

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

    print("\n" + "="*50)
    print(f" >>> TRAINING ML MODEL <<< ")
    print(f" Parameters: \n - Learning Rate: {learning_rate}\n - Depth: {depth}\n - Iterations: {n_iterations}")
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
        case_study=case_study
    )
    
    print("\n" + "*"*50)
    print(" TRAINING PROCESS COMPLETE! ")
    print("*"*50 + "\n")

if __name__ == "__main__":
    main()

# Running command:
# python training_predictive_model.py --case_study "BAC" --learning_rate 0.5 --depth 14 --n_iterations 4000