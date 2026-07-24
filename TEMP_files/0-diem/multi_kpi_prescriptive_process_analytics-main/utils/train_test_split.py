import numpy as np
import pandas as pd
from pathlib import Path

def getting_traces_status(dataframe, case_id_name):
    df = dataframe.copy()
    list_unique_id = df[case_id_name].unique()
    df['trace_status'] = ""
    for case_id in list_unique_id:
        sub_df = df.loc[df[case_id_name] == case_id] # Creating a dataframe with all activities refered to the same case_id
        indexes = sub_df.index.values.tolist()
        start_event_idx = indexes[0]
        last_event_idx = indexes[-1]
        for i in indexes:
            if i == last_event_idx: # Indicating last activity
                df['trace_status'][i] = 'completed'
            elif i == start_event_idx:
                 df['trace_status'][i] = 'start'
            else:
                df['trace_status'][i] = 'active'
    return df    

def extract_data_after_tsplit(df, data_with_trace_status, t_split, case_id_name):
    start_traces_df = data_with_trace_status[(data_with_trace_status['trace_status'] == 'start')]
    completed_traces_df = data_with_trace_status[data_with_trace_status['trace_status'] == 'completed']
    train_id = completed_traces_df[completed_traces_df["time:timestamp"] <= t_split][case_id_name].unique() # Traces that ended at or before split time go to train set
    future_id = start_traces_df[start_traces_df["start:timestamp"] >= t_split][case_id_name].unique() # Traces that started after split time (Remove these traces from the test set - only consider traces are running at split time)
    train_data = df.loc[df[case_id_name].isin(train_id)].reset_index(drop=True)
    return train_data, train_id, future_id

def train_test_split(df, case_study, t_split, case_id_name):
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])
    temp_df = df.copy()
    # Flag starting and completing event of traces
    new_temp_test = getting_traces_status(temp_df, case_id_name)

    # Split data based on the split time (All traces with completed time before the split time will be in train set) 
    train_data, train_id, future_id = extract_data_after_tsplit(df, new_temp_test, t_split, case_id_name) 

    # Create test set by removing traces that are in train and future sets
    ids = np.concatenate([train_id, future_id], axis=0)
    test_data = df.loc[~df[case_id_name].isin(ids)].reset_index(drop=True) # Representing running traces at split time
    
    output_dir = Path(f"./case_studies/{case_study}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)

    print("Summary:")
    print("Total number of traces in the dataset:", len(df['case:concept:name'].unique()))
    print(f"Number of traces in train: {len(train_id)} ({len(train_id)/len(df['case:concept:name'].unique())*100:.2f}%)")
    print(f"Number of traces in future (exclude from train and test sets): {len(future_id)} ({len(future_id)/len(df['case:concept:name'].unique())*100:.2f}%)")
    print(f"Number of traces in test: {len(test_data['case:concept:name'].unique())} ({len(test_data['case:concept:name'].unique())/len(df['case:concept:name'].unique())*100:.2f}%)")

    return train_data, test_data