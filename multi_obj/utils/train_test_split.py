import numpy as np
import pandas as pd
from pathlib import Path

def getting_traces_status(dataframe, case_id_name):
    # Flag, for every event, whether it is the first ('start'), the last
    # ('completed') or an intermediate ('active') event of its trace. The order
    # of the rows as they arrive is preserved (the caller sorts by case/time
    # beforehand). Vectorised equivalent of the original per-event loop: for a
    # single-event trace the event is 'completed' (the "last" check wins).
    df = dataframe.copy()
    grp = df.groupby(case_id_name, sort=False)
    position = grp.cumcount()
    trace_len = grp[case_id_name].transform('size')
    df['trace_status'] = np.where(
        position == trace_len - 1, 'completed',
        np.where(position == 0, 'start', 'active')
    )
    return df

def extract_data_after_tsplit(df, data_with_trace_status, t_split, case_id_name):
    start_traces_df = data_with_trace_status[(data_with_trace_status['trace_status'] == 'start')]
    completed_traces_df = data_with_trace_status[data_with_trace_status['trace_status'] == 'completed']
    train_id = completed_traces_df[completed_traces_df["time:timestamp"] <= t_split][case_id_name].unique() # Traces that ended at or before split time go to train set
    future_id = start_traces_df[start_traces_df["start:timestamp"] >= t_split][case_id_name].unique() # Traces that started after split time (Remove these traces from the test set - only consider traces are running at split time)
    train_data = df.loc[df[case_id_name].isin(train_id)].reset_index(drop=True)
    return train_data, train_id, future_id

def train_test_split(df, case_study, t_split, case_id_name, output_suffix=""):
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

    # Normalise the split timestamp so it can be compared against the (possibly
    # timezone-aware) 'time:timestamp' / 'start:timestamp' columns.
    t_split = pd.to_datetime(t_split)
    ts_col = pd.to_datetime(df['time:timestamp'])
    ts_tz = getattr(ts_col.dt, 'tz', None)
    if ts_tz is not None and t_split.tzinfo is None:
        t_split = t_split.tz_localize(ts_tz)
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
    train_data.to_csv(output_dir / f"train_data{output_suffix}.csv", index=False)
    test_data.to_csv(output_dir / f"test_data{output_suffix}.csv", index=False)

    print("Summary:")
    print("Total number of traces in the dataset:", len(df['case:concept:name'].unique()))
    print(f"Number of traces in train: {len(train_id)} ({len(train_id)/len(df['case:concept:name'].unique())*100:.2f}%)")
    print(f"Number of traces in future (exclude from train and test sets): {len(future_id)} ({len(future_id)/len(df['case:concept:name'].unique())*100:.2f}%)")
    print(f"Number of traces in test: {len(test_data['case:concept:name'].unique())} ({len(test_data['case:concept:name'].unique())/len(df['case:concept:name'].unique())*100:.2f}%)")

    return train_data, test_data

def extract_internal_running_validation(X_trans, y_train, train_data, case_id_name, train_ratio=0.8):
    """
    Split temporale per la validazione interna usata da Optuna.

    Le tracce di training vengono ordinate in base al loro istante di
    completamento (timestamp massimo). Le prime `train_ratio` (es. 80%)
    diventano il sotto-training set; TUTTE le restanti tracce (il restante
    20%, indipendentemente da quando sono iniziate) diventano il set di
    validazione. Nessuna traccia viene scartata.
    """

    # Ricostruiamo la mappatura Case ID -> Timestamp per le righe correnti
    df_mini = pd.DataFrame({
        'case_id': train_data.loc[y_train.index, case_id_name],
        'timestamp': train_data.loc[y_train.index, 'time:timestamp']
    })

    # Ordiniamo le tracce in base al loro istante di completamento
    trace_ends = df_mini.groupby('case_id')['timestamp'].max().sort_values()

    # Punto di split: le prime `target_train_count` tracce (per data di
    # completamento) vanno in train, tutte le altre in validation
    target_train_count = int(len(trace_ends) * train_ratio)

    sub_train_ids = trace_ends.iloc[:target_train_count].index.values
    sub_val_ids = trace_ends.iloc[target_train_count:].index.values
    tr_indices = df_mini[df_mini['case_id'].isin(sub_train_ids)].index
    val_indices = df_mini[df_mini['case_id'].isin(sub_val_ids)].index
    
    pos_map = {idx: pos for pos, idx in enumerate(y_train.index)}
    tr_pos = [pos_map[i] for i in tr_indices if i in pos_map]
    val_pos = [pos_map[i] for i in val_indices if i in pos_map]

    return X_trans[tr_pos], X_trans[val_pos], y_train.iloc[tr_pos], y_train.iloc[val_pos]