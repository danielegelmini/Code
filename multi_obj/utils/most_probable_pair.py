"""
For each case in a query-instance dataframe (one row per
case, named test_log_with_last_act.csv), find the most frequent next activity
given its trace history (via the transition system built on train_data), and
the most frequent resource for that activity (via a resource-frequency map
built on train_data).
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import tqdm

from utils.transition_system import transition_system
from utils.recommendation_functions import next_possible_activities


def act_with_res_freq_func(
    df: pd.DataFrame, activity_column_name: str, resource_column_name: str
) -> Dict[str, Dict[str, int]]:
    """
    For each activity in df, build a mapping of resource -> frequency count,
    based on historical (training) data. "NotDef" resources are dropped.
    """
    list_activities = df[activity_column_name].unique()
    result = {}
    for act in list_activities:
        res = df[df[activity_column_name] == act][resource_column_name].value_counts().to_dict()
        res.pop("NotDef", None)
        result[act] = res
    return result


def best_pair_initialization(
    trace_history: List[str],
    transition_graph: Dict,
    list_act_with_res: Dict[str, Dict[str, int]],
    window_size: int = 5,
) -> Optional[Tuple[str, str]]:
    """
    Given a trace history, return the (most_probable_activity, most_probable_resource)
    pair, or None if no possible next activity is found in the transition graph.
    """
    pos_acts = next_possible_activities(trace_history, transition_graph, window_size)

    if not pos_acts:  # No possible activities found
        return None

    best_act = [k for k, v in pos_acts.items() if v == max(pos_acts.values())][0]

    act_res_freq = list_act_with_res.get(best_act)
    if not act_res_freq:  # No known resource history for this activity
        return None

    best_res = [k for k, v in act_res_freq.items() if v == max(act_res_freq.values())][0]

    return (best_act, best_res)


def add_most_probable_pairs(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    test_log: pd.DataFrame,
    case_id_name: str,
    activity_column_name: str,
    resource_column_name: str,
    next_activity_column: str = "NEXT_ACTIVITY",
    next_resource_column: str = "NEXT_RESOURCE",
    window_size: int = 5,
) -> pd.DataFrame:
    
    test_data = test_data.copy()

    print("Building transition system from train_data...")
    _, ts_with_freq = transition_system(
        train_data,
        case_id_name=case_id_name,
        activity_column_name=activity_column_name,
        window_size=window_size,
    )

    print("Building activity -> resource frequency map...")
    list_act_with_res = act_with_res_freq_func(train_data, activity_column_name, resource_column_name)

    n_total = len(test_data)
    n_missing = 0

    case_ids = test_data[case_id_name].tolist()
    for i, case_id in enumerate(tqdm.tqdm(case_ids, desc="Processing cases...")):
        trace_df = test_log[test_log[case_id_name] == case_id]
        trace_history = trace_df[activity_column_name].tolist()

        (best_act, best_res) = best_pair_initialization(
            trace_history, ts_with_freq, list_act_with_res, window_size=window_size
        )
        if (best_act, best_res) is None:
            n_missing += 1
            continue

        test_data.at[test_data.index[i], next_activity_column] = best_act
        test_data.at[test_data.index[i], next_resource_column] = best_res

    print(f"Cases with no possible next action found: {n_missing}/{len(case_ids)}")

    return test_data
