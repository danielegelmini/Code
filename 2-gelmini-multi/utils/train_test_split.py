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
                df.loc[i, 'trace_status'] = 'completed'
            elif i == start_event_idx:
                df.loc[i, 'trace_status'] = 'start'
            else:
                df.loc[i, 'trace_status'] = 'active'
    return df

# def extract_data_after_tsplit(df, data_with_trace_status, t_split, case_id_name):
#     # Normalize split timestamp for safe datetime comparisons with tz-aware columns
#     t_split = pd.to_datetime(t_split)
#     if pd.api.types.is_datetime64tz_dtype(data_with_trace_status['time:timestamp'].dtype) and t_split.tzinfo is None:
#         t_split = t_split.tz_localize(data_with_trace_status['time:timestamp'].dt.tz)
#     if pd.api.types.is_datetime64tz_dtype(data_with_trace_status['start:timestamp'].dtype) and t_split.tzinfo is None:
#         t_split = t_split.tz_localize(data_with_trace_status['start:timestamp'].dt.tz)

#     start_traces_df = data_with_trace_status[(data_with_trace_status['trace_status'] == 'start')]
#     completed_traces_df = data_with_trace_status[data_with_trace_status['trace_status'] == 'completed']
#     train_id = completed_traces_df[completed_traces_df["time:timestamp"] <= t_split][case_id_name].unique() # Traces that ended at or before split time go to train set
#     future_id = start_traces_df[start_traces_df["start:timestamp"] >= t_split][case_id_name].unique() # Traces that started after split time (Remove these traces from the test set - only consider traces are running at split time)
#     train_data = df.loc[df[case_id_name].isin(train_id)].reset_index(drop=True)
#     return train_data, train_id, future_id

def extract_data_after_tsplit(df, data_with_trace_status, t_split, case_id_name):
    # Normalize split timestamp for safe datetime comparisons with tz-aware columns
    t_split = pd.to_datetime(t_split)
    if pd.api.types.is_datetime64tz_dtype(data_with_trace_status['time:timestamp'].dtype) and t_split.tzinfo is None:
        t_split = t_split.tz_localize(data_with_trace_status['time:timestamp'].dt.tz)
        
    # Get completed traces
    completed_traces = data_with_trace_status[data_with_trace_status['trace_status'] == 'completed']
    train_id = completed_traces[completed_traces["time:timestamp"] <= t_split][case_id_name].unique()
    
    # ==========================================
    # NEW CODE: FORCE EXACTLY 80% LIMIT
    # ==========================================
    total_traces = df[case_id_name].nunique()
    target_train_count = round(total_traces * 0.80)
    
    sliced_off_ids = []
    if len(train_id) > target_train_count:
        # Sort the training cases by their exact end time to keep chronological order
        trace_ends = df[df[case_id_name].isin(train_id)].groupby(case_id_name)['time:timestamp'].max().sort_values()
        
        # Keep exactly the number needed for 80%
        train_id_limited = trace_ends.iloc[:target_train_count].index.values
        
        # Identify the traces we cut off due to tied timestamps
        sliced_off_ids = list(set(train_id) - set(train_id_limited))
        train_id = train_id_limited
    # ==========================================

    # Get future traces
    future_traces = data_with_trace_status[data_with_trace_status['trace_status'] == 'start']
    future_id = future_traces[future_traces["time:timestamp"] > t_split][case_id_name].unique()
    
    # Add the sliced-off IDs to future_id so they are excluded from the running test set
    if sliced_off_ids:
        future_id = np.concatenate([future_id, sliced_off_ids])
        
    train_data = df[df[case_id_name].isin(train_id)].reset_index(drop=True)
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

def extract_internal_running_validation(X_trans, y_train, train_data, case_id_name, train_ratio=0.85):
    """
    Simula lo split temporale del predecessore all'interno di Optuna.
    Prende le tracce di train originarie, trova un t_split interno basato sulla 
    proporzione fornita (es. 85%) e isola le tracce in corso (running) da usare come validazione.
    """
    import pandas as pd
    import numpy as np
    
    # Ricostruiamo la mappatura dei Case ID e dei relativi Timestamp
    df_mini = pd.DataFrame({
        'case_id': train_data.loc[y_train.index, case_id_name],
        'timestamp': train_data.loc[y_train.index, 'time:timestamp']
    })
    
    # Ordiniamo le tracce in base al loro istante di completamento
    trace_ends = df_mini.groupby('case_id')['timestamp'].max().sort_values()
    
    # Identifichiamo il punto di split interno (es. all'85%)
    target_train_count = int(len(trace_ends) * train_ratio)
    if target_train_count == 0 or target_train_count >= len(trace_ends):
        # Fallback di sicurezza se il dataset è troppo piccolo per essere campionato cronologicamente
        split_pos = int(len(X_trans) * train_ratio)
        return X_trans[:split_pos], X_trans[split_pos:], y_train.iloc[:split_pos], y_train.iloc[split_pos:]
        
    t_split_internal = trace_ends.iloc[target_train_count]
    
    # ID delle tracce assegnate al sotto-addestramento (completate prima del t_split)
    sub_train_ids = trace_ends.iloc[:target_train_count].index.values
    
    # Le tracce di validazione sono quelle "in corso": iniziate prima o a t_split, ma non ancora completate
    trace_starts = df_mini.groupby('case_id')['timestamp'].min()
    started_before_split = trace_starts[trace_starts <= t_split_internal].index
    
    sub_val_ids = df_mini[
        (df_mini['case_id'].isin(started_before_split)) & 
        (~df_mini['case_id'].isin(sub_train_ids))
    ]['case_id'].unique()
    
    # Estraiamo gli indici di riga del dataframe originale
    tr_indices = df_mini[df_mini['case_id'].isin(sub_train_ids)].index
    val_indices = df_mini[df_mini['case_id'].isin(sub_val_ids)].index
    
    # Fallback nel caso in cui non ci siano tracce attive in quel preciso istante
    if len(val_indices) == 0 or len(tr_indices) == 0:
        split_pos = int(len(X_trans) * train_ratio)
        return X_trans[:split_pos], X_trans[split_pos:], y_train.iloc[:split_pos], y_train.iloc[split_pos:]
        
    # Mappiamo gli indici originali sulle posizioni correnti all'interno delle matrici
    pos_map = {idx: pos for pos, idx in enumerate(y_train.index)}
    tr_pos = [pos_map[i] for i in tr_indices if i in pos_map]
    val_pos = [pos_map[i] for i in val_indices if i in pos_map]
    
    return X_trans[tr_pos], X_trans[val_pos], y_train.iloc[tr_pos], y_train.iloc[val_pos]