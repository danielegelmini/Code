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
    linear_combination,
)
from datetime import datetime
from pathlib import Path

from utils.train_test_split import train_test_split, compute_split_time

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
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="The split time t_split is the earliest time at which this share of the traces is "
                             "completed (default: 0.8). Traces completed by t_split -> train; traces running at "
                             "t_split -> test; traces starting after t_split -> excluded.")
    parser.add_argument("--lambda_value", type=float, default=0.5, help="Weight for linear combination of normalized remaining time and case outcome (default: 0.5)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to every output CSV file name (e.g. '_new'). "
                             "Use it to avoid overwriting existing files while validating.")

    args = parser.parse_args()
    case_study = args.case_study
    lambda_value = args.lambda_value
    output_suffix = args.output_suffix

    # ==========================================
    # [ STEP 1: PREPROCESSING ]
    # ==========================================
    print("\n" + "="*50)
    print(f" >>> DATA PREPROCESSING FOR {case_study} <<< ")
    print("="*50)

    # Primary Pre-processing step (Dummy positions passed initially, handled dynamically inside function)
    df = data_pre_processing(case_study, 0, 0, date_format, 0, case_id_name,
                             start_date_name, end_date_name, activity_column_name, resource_column_name,
                             output_suffix=output_suffix)

    # ==========================================
    # [ STEP 2: DATA SPLITTING ]
    # ==========================================
    print("\n" + "="*50)
    print(" >>> SPLITTING TRAIN & TEST DATA <<< ")
    print("="*50)
    split_time = compute_split_time(df, case_id_name, args.train_ratio)
    print(f"Split time (earliest time at which {args.train_ratio:.0%} of the traces are completed): {split_time}")
    train_data, test_data = train_test_split(df, case_study, split_time, "case:concept:name",
                                             output_suffix=output_suffix)

    # ==========================================
    # [ STEP 3: NORMALIZATION ]
    # ==========================================
    print("\n" + "="*50)
    print(" >>> NORMALIZING REMAINING TIME <<< ")
    print("="*50)
    train_data, test_data = normalize_remaining_time(case_study, train_data, test_data, save_plot=False)

    # ==========================================
    # [ STEP 4: FILTERING TEST LOGS ]
    # ==========================================
    print("\n" + "="*50)
    print(" >>> FILTERING AND GENERATING TEST LOGS <<< ")
    print("="*50)

    # 1. Convert the timestamp column to datetime FIRST
    test_data['time:timestamp'] = pd.to_datetime(test_data['time:timestamp'], format='mixed')

    # 2. Convert the string split_time to a datetime object
    split_time_dt = pd.to_datetime(split_time)

    # 3. Handle Timezones (if the dataset has +00:00, align the split time to match)
    if test_data['time:timestamp'].dt.tz is not None and split_time_dt.tzinfo is None:
        split_time_dt = split_time_dt.tz_localize(test_data['time:timestamp'].dt.tz)

    # 4. Now filter safely
    test_log = test_data[test_data["time:timestamp"] <= split_time_dt].reset_index(drop=True)

    # Get the last activity of each trace in the test log
    test_log_with_last_act = test_log.loc[test_log.groupby('case:concept:name')['time:timestamp'].idxmax()].reset_index(drop=True)

    # Sanity check on the split. A test trace can legitimately have zero events
    # at or before the split time (its first activity is still running across the
    # split), in which case it drops out of the test log; warn instead of failing.
    n_test = len(test_data['case:concept:name'].unique())
    n_test_log = len(test_log['case:concept:name'].unique())
    if n_test != n_test_log:
        print(f"WARNING: {n_test - n_test_log} of {n_test} test traces have no event at/before "
              f"the split time and are absent from the test log.")

    print("Test log filtering completed. Number of traces in test log:", n_test_log)

    # ==========================================
    # [ STEP 5: LINEAR COMBINATION ]
    # ==========================================
    print("\n" + "="*50)
    print(f" >>> COMPUTING LINEAR COMBINATIONS (lambda = {lambda_value}) <<< ")
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

    train_data.to_csv(output_dir / f"train_data{output_suffix}.csv", index=False)
    test_data.to_csv(output_dir / f"test_data{output_suffix}.csv", index=False)
    test_log.to_csv(output_dir / f"test_log{output_suffix}.csv", index=False)
    test_log_with_last_act.to_csv(output_dir / f"test_log_with_last_act{output_suffix}.csv", index=False)

    print("\n" + "*"*50)
    print(" ALL TASKS COMPLETED SUCCESSFULLY! ")
    print("*"*50 + "\n")
    print("All data files have been saved in the following directory:", output_dir)

if __name__ == "__main__":
    main()

# Running commands:
# python 1_data_preprocessing.py --case_study "BPI12"        --lambda_value 0.5
# python 1_data_preprocessing.py --case_study "BAC"          --lambda_value 0.5
# python 1_data_preprocessing.py --case_study "bpi17_before" --lambda_value 0.5
# python 1_data_preprocessing.py --case_study "bpi17_after"  --lambda_value 0.5
