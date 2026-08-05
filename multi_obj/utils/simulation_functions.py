import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils.pre_processing_functions import convert_dtypes_bpi12

import sys
import warnings
sys.path.append("./src")
warnings.filterwarnings("ignore")

case_id_name = "case:concept:name"
activity_column_name = "concept:name"
end_date_name = "time:timestamp"
start_date_name = "start:timestamp"
resource_column_name = "org:resource"
outcome_name = "outcome_sigmoid"

def to_be_named(case_study, method, n_sim, folder_path, encoded_activity=None):
    
    sim_path = folder_path + f"case_studies/{case_study}/simulation_results/{method}/" # Folder to save simulation results
    test_log = pd.read_csv(folder_path + f"case_studies/{case_study}/test_log.csv") # Test log path
    result_df = pd.read_csv(folder_path + f"case_studies/{case_study}/recommendations_{case_study}_{method}.csv")
    dataframes = []


    for i in range(n_sim):
        sim = pd.read_csv(sim_path + "sim_{}.csv".format(i+1))
        if case_study in {"BPI12", "bpi12_time", "bpi12_status", "bpi12_025", "bpi12_075", "consulta", "bpi12_039"}:
            sim = convert_dtypes_bpi12(sim, 'simulation')
        sim = sim[[case_id_name, start_date_name, end_date_name, activity_column_name, resource_column_name]] 
        sim = getting_remaining_time(sim, "case:concept:name", "time:timestamp")
        sim = status_encoding(sim, case_study, encoded_activity)

        for j in range(len(sim)):
            sim.loc[j, case_id_name] = str(sim[case_id_name][j]) + "_" + str(i+1)
                
        dataframes.append(sim)

    test_data_simulation = pd.concat(dataframes, ignore_index=True).reset_index(drop=True) 

    rec_df = preparing_data_for_simulation(result_df, test_log, case_id_name, end_date_name, case_study)

    res, res_status = compute_res_and_status(case_study, rec_df, test_data_simulation, n_sim)

    return res, res_status

def build_recommender_df(prev_log: pd.DataFrame, recommendations: dict) -> pd.DataFrame:

    prev_log = prev_log.copy()
    prev_log["recommendation:act"] = None
    prev_log["recommendation:res"] = None

    for case_id, recs in recommendations.items():
        act = recs.get("act", None)
        res = recs.get("res", None)
        case_rows = prev_log[prev_log["case:concept:name"] == case_id]
        if case_rows.empty:
            continue

        if "time:timestamp" in case_rows.columns:
            case_rows = case_rows.sort_values("time:timestamp")

        last_index = case_rows.index[-1]
        prev_log.at[last_index, "recommendation:act"] = act
        prev_log.at[last_index, "recommendation:res"] = res

    return prev_log


def _remove_prefix_rows(full_seq: pd.DataFrame, prefix_seq: pd.DataFrame) -> pd.DataFrame:
    if full_seq.empty or prefix_seq.empty:
        return full_seq.copy()

    key_cols = [activity_column_name, resource_column_name, start_date_name, end_date_name]

    full_keyed = full_seq.copy()
    prefix_keyed = prefix_seq.copy()

    full_keyed["__dup_idx"] = full_keyed.groupby(key_cols).cumcount()
    prefix_keyed["__dup_idx"] = prefix_keyed.groupby(key_cols).cumcount()
    prefix_keyed["__is_prefix"] = True

    merged = full_keyed.merge(
        prefix_keyed[key_cols + ["__dup_idx", "__is_prefix"]],
        on=key_cols + ["__dup_idx"],
        how="left",
    )

    out = merged[merged["__is_prefix"].isna()].drop(columns=["__dup_idx", "__is_prefix"]).copy()
    return out.sort_values(by=[start_date_name, end_date_name]).reset_index(drop=True)


def check_recommendation_following(
    prefix_log: pd.DataFrame,
    recommendations: dict,
    simulation_logs: list[pd.DataFrame],
) -> pd.DataFrame:
    """Check recommendation adherence per case/simulation on the first post-prefix start-time batch.

    Returns one row per (case_id, simulation_index) with:
      - recommended activity/resource
      - first post-prefix start-time batch
      - whether recommended pair appears in that batch
    """

    prefix = prefix_log.copy()
    prefix[case_id_name] = prefix[case_id_name].astype(str)
    prefix[start_date_name] = pd.to_datetime(prefix[start_date_name], format="mixed", utc=True, errors="coerce")
    prefix[end_date_name] = pd.to_datetime(prefix[end_date_name], format="mixed", utc=True, errors="coerce")

    normalized_recs = {str(k): v for k, v in recommendations.items()}
    rows = []

    for sim_idx, sim_df in enumerate(simulation_logs, start=1):
        sim = sim_df.copy()
        sim[case_id_name] = sim[case_id_name].astype(str)
        sim[start_date_name] = pd.to_datetime(sim[start_date_name], format="mixed", utc=True, errors="coerce")
        sim[end_date_name] = pd.to_datetime(sim[end_date_name], format="mixed", utc=True, errors="coerce")

        for cid, recs in normalized_recs.items():
            rec_act = recs.get("act")
            rec_res = recs.get("res")
            if rec_act is None:
                continue

            prefix_seq = prefix[prefix[case_id_name] == cid].sort_values(by=[start_date_name, end_date_name]).reset_index(drop=True)
            if prefix_seq.empty:
                continue

            sim_seq = sim[sim[case_id_name] == cid].sort_values(by=[start_date_name, end_date_name]).reset_index(drop=True)
            if sim_seq.empty:
                rows.append({
                    case_id_name: cid,
                    "simulation_index": sim_idx,
                    "recommended_activity": rec_act,
                    "recommended_resource": rec_res,
                    "first_batch": "",
                    "match_recommendation": False,
                    "status": "missing_case",
                })
                continue

            boundary = prefix_seq[end_date_name].max()
            generated = _remove_prefix_rows(sim_seq, prefix_seq)
            continuation = generated[generated[end_date_name] >= boundary].sort_values(by=[start_date_name, end_date_name]).reset_index(drop=True)

            if continuation.empty:
                rows.append({
                    case_id_name: cid,
                    "simulation_index": sim_idx,
                    "recommended_activity": rec_act,
                    "recommended_resource": rec_res,
                    "first_batch": "",
                    "match_recommendation": False,
                    "status": "no_continuation",
                })
                continue

            first_start = continuation[start_date_name].min()
            first_batch = continuation[continuation[start_date_name] == first_start]

            match = False
            if pd.isna(rec_res) or rec_res is None:
                match = (first_batch[activity_column_name] == rec_act).any()
            else:
                match = ((first_batch[activity_column_name] == rec_act) & (first_batch[resource_column_name] == rec_res)).any()

            batch_str = " | ".join(
                f"{r[activity_column_name]} / {r[resource_column_name]}"
                for _, r in first_batch.iterrows()
            )

            rows.append({
                case_id_name: cid,
                "simulation_index": sim_idx,
                "recommended_activity": rec_act,
                "recommended_resource": rec_res,
                "first_batch": batch_str,
                "match_recommendation": bool(match),
                "status": "ok",
            })

    return pd.DataFrame(rows)

def preparing_data_for_simulation(result_df, test_log, case_id_name, end_date_name, case_study):
    # Generating dataframe with repl_id, act_1, res_1, starting_time
    simu_df = pd.DataFrame(columns=["case:concept:name", "repl_id", "act_1", "res_1", "starting_time"])
    
    if case_study in {"BPI12", "bpi12_time", "bpi12_status", "bpi12_025", "bpi12_075", "consulta", "bpi12_0", "bpi12_039"}:
        result_df = convert_dtypes_bpi12(result_df, 'simulation_prep')
        test_log = convert_dtypes_bpi12(test_log, 'experiment')
        simu_df = convert_dtypes_bpi12(simu_df, 'simulation_')
    
    test_log_ids = result_df[case_id_name].unique()
    simu_df["case:concept:name"] = test_log_ids
        
    for i, idx in enumerate(test_log_ids):
        trace_df = test_log[test_log[case_id_name] == idx]

        trace_ids = trace_df[case_id_name].tolist() # List of prefix history
        simu_df.loc[i, 'repl_id'] = len(trace_ids) - 1
        simu_df.loc[i, "act_1"] = result_df["Next_activity"][i]
        simu_df.loc[i, "res_1"] = result_df["Next_resource"][i]
        simu_df.loc[i, "starting_time"] = trace_df[-1:][end_date_name].values[0]
    
    simu_df = simu_df.sort_values(by=['starting_time']).reset_index(drop=True)

    return simu_df   
 
def convert_time(time_to_convert):
    dt = datetime.strptime(time_to_convert.split('+')[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
    return dt

def getting_remaining_time(dataframe, case_id_name, end_date_name):
    dataframe["remaining_time"] = np.nan
    dataframe['time:timestamp'] = dataframe['time:timestamp'].apply(convert_time)
    dataframe['start:timestamp'] = dataframe['start:timestamp'].apply(convert_time)

    list_trace_id = set(dataframe[case_id_name].unique()) # Getting list of trace ID
    for trace_id in list_trace_id:
        trace_df = dataframe[dataframe[case_id_name] == trace_id] # Getting sub df for each trace
        
        sub_trace_df_sorted = trace_df.sort_values(by=[end_date_name]) # Sorting by time
            
        indices = sub_trace_df_sorted.index.values.tolist()
        last_event_idx = indices[-1]

        for idx in indices:
            dataframe.loc[idx, 'remaining_time'] = (sub_trace_df_sorted[end_date_name][last_event_idx] - sub_trace_df_sorted[end_date_name][idx]).total_seconds()

    dataframe = dataframe.sort_values(by=[case_id_name, end_date_name]).reset_index(drop=True)  
    return dataframe

def status_encoding(df: pd.DataFrame, case_study: str, encoded_activity: str | None = None) -> pd.DataFrame:
    
    if case_study in {"BAC", "BAC_synthetic", "BAC_time", "BAC_status", "BAC_025", "BAC_075", "BAC_0", "BAC_041"}:
        # Status=0 if any forbidden activity appears in the case, else 1
        forbidden = {"Back-Office Adjustment Requested", "Network Adjustment Requested"}
        case_status = (
            df.groupby("case:concept:name")["concept:name"]
              .apply(lambda acts: 0 if any(a in forbidden for a in acts) else 1)
        )
    else:
        if not encoded_activity:
            raise ValueError("encoded_activity must be provided for non-BAC case studies")
        case_status = (
            df.groupby("case:concept:name")["concept:name"]
              .apply(lambda acts: 1 if encoded_activity in acts.values else 0)
        )

    # Map back to df
    df["status"] = df["case:concept:name"].map(case_status)

    return df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

def compute_res_and_status(case_study, rec_df, test_simu, n_sim):
    """
    Compute average remaining_time and status per trace_id.

    Notes:
    - The original code looped over n_sim but did not filter by simulation,
      so it averaged identical values. This version preserves that behavior.
    - If you truly have per-simulation rows, see the comment at the bottom.
    # """
    if case_study in {"BPI12", "BPI12_time", "BPI12_status", "BPI12_025", "BPI12_075", "consulta", "BPI12_0", "BPI12_039"}:
        test_simu = convert_dtypes_bpi12(test_simu, 'simulation')
        rec_df = convert_dtypes_bpi12(rec_df, 'simulation_')
    res = {}
    res_status = {}
    trace_ids = rec_df[case_id_name].unique()

    for trace_id in trace_ids:
        rec_index = rec_df[rec_df[case_id_name] == trace_id]['repl_id'].values[0] + 1
        list_remaning_time = []
        list_status = []
        for i in range(n_sim):
            idx_sim = str(trace_id) + "_" + str(i+1)
            trace_df = test_simu[test_simu[case_id_name] == idx_sim]
            if not trace_df.empty:
                start_index = trace_df.index.values.tolist()[0]
                #change 1
                if start_index + rec_index  >= start_index + len(trace_df):
                    remaining_time = 0
                else:
                    #chage 2
                    remaining_time = trace_df.loc[start_index + rec_index]['remaining_time'] 
                status = trace_df['status'].unique()[0]
                if remaining_time < 0:
                    print(idx_sim)
                list_remaning_time.append(remaining_time)
                list_status.append(status)
        
        if len(list_remaning_time) > 0 and len(list_status) > 0:
            res[trace_id] = np.mean(list_remaning_time)
            res_status[trace_id] = np.mean(list_status)
        else:
            res[trace_id] = -1
            res_status[trace_id] = 0

    return res, res_status