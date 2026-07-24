# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import numpy as np
import pm4py
from utils.data_normalization import normalize_remaining_time
from utils.pre_processing_functions import (
    prepare_data_and_add_features, 
    add_next_act_res, 
    preprocessing_activity_frequency, 
    getting_total_time, 
    data_labelling,
    data_pre_processing,
    linear_combination
)
from dateutil import parser
from datetime import datetime
from pathlib import Path

from utils.train_test_split import train_test_split

date_format = "%Y-%m-%d %H:%M:%S.%f"
case_id_name = "case:concept:name"
start_date_name = "start:timestamp"
end_date_name = "time:timestamp"
activity_column_name = "concept:name"
resource_column_name = "org:resource"


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process process-mining data for a case study.")
    parser.add_argument("--case_study", type=str, required=True, help="Name of the case study folder/file")
    parser.add_argument("--split_time", type=str, required=True, help="Timestamp to split train and test data (e.g., '2026-06-30 14:00:00')")
    parser.add_argument("--lambda_value", type=float, default=0.5, help="Weight for linear combination of normalized remaining time and case outcome (default: 0.5)")
    
    args = parser.parse_args()
    case_study = args.case_study
    split_time = args.split_time
    lambda_value = args.lambda_value

    # ==========================================
    # [ STEP 1: PREPROCESSING ]
    # ==========================================
    print("\n" + "="*50)
    print(f" >>> DATA PREPROCESSING FOR {case_study} <<< ")
    print("="*50)
    
    # Primary Pre-processing step (Dummy positions passed initially, handled dynamically inside function)
    df = data_pre_processing(case_study, 0, 0, date_format, 0, case_id_name, 
                             start_date_name, end_date_name, activity_column_name, resource_column_name)

    # ==========================================
    # [ STEP 2: DATA SPLITTING ]
    # ==========================================
    print("\n" + "="*50)
    print(" >>> SPLITTING TRAIN & TEST DATA <<< ")
    print("="*50)
    train_data, test_data = train_test_split(df, case_study, split_time, "case:concept:name")

    # ==========================================
    # [ STEP 3: NORMALIZATION ]
    # ==========================================
    print("\n" + "="*50)
    print(" >>> NORMALIZING REMAINING TIME <<< ")
    print("="*50)
    train_data, test_data = normalize_remaining_time(case_study, train_data, test_data, save_plot=True)

    # ==========================================
    # [ STEP 4: FILTERING TEST LOGS ]
    # ==========================================
    print("\n" + "="*50)
    print(" >>> FILTERING AND GENERATING TEST LOGS <<< ")
    print("="*50)
    
    # Test log filtering based on split time
    test_log = test_data[test_data["time:timestamp"] <= split_time].reset_index(drop=True)
    test_log['time:timestamp'] = pd.to_datetime(test_log['time:timestamp'], format='mixed')
    
    # Get the last activity of each trace in the test log
    test_log_with_last_act = test_log.loc[test_log.groupby('case:concept:name')['time:timestamp'].idxmax()].reset_index(drop=True)

    # Ensuring splitting is done correctly
    assert len(test_data['case:concept:name'].unique()) == len(test_log['case:concept:name'].unique()), \
        "The number of traces in the test set and test log should be the same!"
    
    print("Test log filtering completed successfully. Number of traces in test log:", len(test_log['case:concept:name'].unique()))
    # ==========================================
    # [ STEP 5: LINEAR COMBINATION ]
    # ==========================================
    print("\n" + "="*50)
    print(f" >>> COMPUTING LINEAR COMBINATIONS (λ = {lambda_value}) <<< ")
    print("="*50)
    train_data = linear_combination(train_data, lambda_weight=lambda_value)
    test_data = linear_combination(test_data, lambda_weight=lambda_value)
    test_log = linear_combination(test_log, lambda_weight=lambda_value)
    test_log_with_last_act = linear_combination(test_log_with_last_act, lambda_weight=lambda_value)

    # ==========================================
    # [ STEP 6: EXPORT DATA ]
    # ==========================================
    output_dir = Path(f"./case_studies/{case_study}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)
    test_log.to_csv(output_dir / "test_log.csv", index=False)
    test_log_with_last_act.to_csv(output_dir / "test_log_with_last_act.csv", index=False)
    
    print("\n" + "*"*50)
    print(" ALL TASKS COMPLETED SUCCESSFULLY! ")
    print("*"*50 + "\n")
    print("All data files have been saved in the following directory:", output_dir)

if __name__ == "__main__":
    main()

# Running commands:
# python data_preprocessing.py --case_study "BPI12" --split_time "2012-02-16 00:00:00" --lambda_value 0.5