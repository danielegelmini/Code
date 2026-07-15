import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.get_features import get_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.predictive_models_functions import train_ml_model

end_date_name = 'time:timestamp'
start_date_name = 'start:timestamp'

params = {
    "case_study": "bpi17_after",
    "optuna_trials": 80,  
    "early_stopping_rounds": 50, 
    "search_spaces": {
        "learning_rate": {"type": "float", "min": 0.01, "max": 0.15, "log": True}, #0.01-0.3 for bpi12 and bac, 0.01-0.15 for bpic2017 
        "depth": {"type": "int", "min": 3, "max": 6}, #4-8 for bpi12 and bac, 3-6 for bpic2017                            
        "l2_leaf_reg": {"type": "float", "min": 15.0, "max": 60.0}, #1-30 for bpi12 and bac, 15-60 for bpic2017               
        "colsample_bylevel": {"type": "float", "min": 0.6, "max": 1.0},
        "bootstrap_type": {"type": "categorical", "choices": ["Bayesian", "Bernoulli", "MVS"]}
    }
    
}

def main():
    runtime_params = params.copy()
    case_study = runtime_params["case_study"]

    print("\n" + "="*50)
    print(f" >>> LOADING DATASETS FOR {case_study.upper()} <<< ")
    print("="*50)
    
    data_dir = Path(f"./case_studies/{case_study}")
    case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove = get_features(case_study)

    
    train_data = pd.read_csv(data_dir / "train_data.csv", parse_dates=[end_date_name, start_date_name])
    test_data = pd.read_csv(data_dir / "test_data.csv", parse_dates=[end_date_name, start_date_name])

    if case_study == "BPI12":
        print("\nApplying BPI12 specific data type conversions...")
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data  = convert_dtypes_bpi12(test_data, "experiment")

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_study": case_study,
        "optuna_trials": runtime_params["optuna_trials"],
        "early_stopping_rounds": runtime_params["early_stopping_rounds"],
        **{f"search_{k}": str(v) for k, v in runtime_params["search_spaces"].items()}
    }
    df_record = pd.DataFrame([record])
    output_dir = Path(f"./case_studies/{case_study}/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_record.to_csv(output_dir / "params.csv", mode='w', header=True, index=False)

    print("\n" + "="*50)
    print(f" >>> TRAINING ML MODEL WITH OPTUNA <<< ")
    print("="*50)
    
    train_ml_model(
        train_data=train_data, 
        test_data=test_data, 
        case_id_name=case_id_name, 
        outcome_name='outcome', 
        columns_to_remove=columns_to_remove, 
        continuous_features=continuous_features, 
        categorical_features=categorical_features, 
        case_study=case_study,
        params=runtime_params,
    )
    
    print("\n" + "*"*50)
    print(" TRAINING PROCESS COMPLETE! ")
    print("*"*50 + "\n")

if __name__ == "__main__":
    main()