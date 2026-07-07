import dice_ml
from dice_ml import Dice
import tqdm
import pandas as pd
from typing import Dict, Tuple, Any, List
import numpy as np

case_id_name = "case:concept:name"
activity_column_name = "concept:name"
end_date_name = "time:timestamp"
start_date_name = "start:timestamp"
resource_column_name = "org:resource"
outcome_name = "outcome_sigmoid_mm"


def build_query_instances(test_df, case_id_name):
    """
    Create a dictionary with case_id: query_instance in dictionary
    
    Parameters:
        test_data (df): Dataframe contains only query instance (Last event before prescription)
        case_id_name (str): The name of the column to be removed that contains case IDs.
        
    Returns:
        dict: A dictionary containing two elements:
            - case_id (str): Case IDs.
            - attribute (dict): {"feature_name": "value",..}
    """
    drop_cols = {case_id_name, start_date_name, end_date_name, "total_time", "remaining_time", "status", outcome_name}
    feature_columns = [c for c in test_df.columns if c not in drop_cols]
    query_instances_by_case = {
        row[case_id_name]: row[feature_columns].to_dict()
        for _, row in test_df.iterrows()
    }
    return query_instances_by_case

def create_DiCE_model(dataframe, continuous_features, categorical_features, outcome_name, dice_method, predictive_model, columns_to_remove):
    """
    Creating DiCE model
    """
    data_for_dice = dataframe.drop(columns_to_remove, axis = 1)
    data_model = dice_ml.Data(dataframe=data_for_dice,
                              continuous_features=continuous_features,
                              categorical_features=categorical_features,
                              outcome_name=outcome_name)
    
    ml_backend = dice_ml.Model(model=predictive_model, backend="sklearn", model_type='regressor')
    method = dice_method  
    explainer = Dice(data_model, ml_backend, method=method) 
    return explainer


def next_possible_activities(trace_history, transition_graph, WINDOW_SIZE):
    """
    Returns the list of possible next activities based on the transition graph and the trace history.
    """
    n = len(trace_history)
    pos_acts = []
    if  n <= WINDOW_SIZE: # trace history is smaller than the window size
        trace_to_compare = trace_history
        trace_to_str =  "".join(trace_to_compare)
        if trace_to_str in transition_graph.keys():
            pos_acts = transition_graph[trace_to_str]
        else:
            for ts in transition_graph.keys():
                ts_to_list = ts.split(", ")
                if ts_to_list == trace_to_compare:
                    pos_acts = transition_graph[ts]
    else:
        trace_to_compare = trace_history[-WINDOW_SIZE:] 
        for ts in transition_graph.keys():
            ts_to_list = ts.split(", ")
            if ts_to_list == trace_to_compare:
                pos_acts = transition_graph[ts]

    return list(pos_acts)
 

def CFE_for_a_single_query(explainer, query_instance, y_predicted, reduced_percentage, total_CFs, cols_to_vary, next_possible_activity, act_with_res, dice_method):
    """
    Generates a list of Counterfactual examples (CFEs) for a query instance
    If no recommendations has found returns -1
    
    Parameters:
        explainer: DiCE explainer object already initialized in create_DiCE_model function.
        query_instance: The input instance (running trace prefix).
        y_predicted: Predicted outcome from the Predictive model.
        reduced_percentage: Target reduction factor for outcome.
        total_CFs: Number of CFEs to generate.
        cols_to_vary: Features allowed to change (Next_activity and Next_resource).
        next_possible_activity: Allowed next activities based on Transition system.
        act_with_res: Activity–resource mapping (Resource Filter).
        dice_method: 'genetic' or 'random' method.        
    Returns:
        List of Counterfactual examples or -1 if no valid counterfactual could be computed.
    """
    total_time_predicted = y_predicted
    if y_predicted < 0:
        total_time_upper_bound = 1
    else:
        total_time_upper_bound = total_time_predicted * reduced_percentage
    
    if dice_method == 'genetic':
        try:
            cfe = explainer.generate_counterfactuals(query_instance,
                                            total_CFs = total_CFs,
                                            features_to_vary = cols_to_vary,
                                            desired_range = [0, total_time_upper_bound],
                                            permitted_range = {"NEXT_ACTIVITY": next_possible_activity},
                                            act_res = act_with_res,
                                            proximity_weight=0.2,
                                            sparsity_weight=0,
                                            maxiterations=10)
        except:
            cfe = -1 # DiCE cannot generate recommendations

    else:
        try:
            cfe = explainer.generate_counterfactuals(query_instance,
                                            total_CFs = total_CFs,
                                            features_to_vary = cols_to_vary,
                                            desired_range = [0, total_time_upper_bound],
                                            permitted_range = {"NEXT_ACTIVITY": next_possible_activity})
        except:
            cfe = -1 # DiCE cannot generate recommendations
    return cfe

def act_with_res_func(df, activity_column_name, resource_column_name):
    """
    Generates a dictionary mapping each unique activity to a list of unique resources associated with it,
    excluding 'missing' and 'NotDef'.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    activity_column_name : str
        The name of the column containing activity names.
    resource_column_name : str
        The name of the column containing resource names.
        
    Returns
    -------
    dict
        {activity: [unique resources]} mapping
    """
    # Group once, drop duplicates efficiently
    grouped = df.groupby(activity_column_name)[resource_column_name].unique()

    # Filter forbidden tokens
    forbidden = {"missing", "NotDef"}
    return {
        act: [res for res in resources if res not in forbidden]
        for act, resources in grouped.items()
    }


def get_best_pairs(cfe, outcome_name):
    cfe_single_query_df = cfe.cf_examples_list[0].final_cfs_df
    n_valid_cfes = len(cfe_single_query_df)
    smallest_outcome = cfe_single_query_df[outcome_name].min()
    best_act, best_res = cfe_single_query_df[cfe_single_query_df[outcome_name] == smallest_outcome][['NEXT_ACTIVITY', 'NEXT_RESOURCE']].values[0]
    return n_valid_cfes, best_act, best_res

def _to_row_df(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[[0]] if len(x) > 1 else x
    if isinstance(x, pd.Series):
        return x.to_frame().T
    if isinstance(x, dict):
        return pd.DataFrame([x])
    raise TypeError(f"Unsupported query_instance type: {type(x)}")

def genetic_recommendations(
    case_study: str,
    test_log: pd.DataFrame,
    test_data: pd.DataFrame, # Contain most probable pairs
    predictive_model: Any,
    dice_method: str,
    explainer,
    transition_graph,
    reduced_percentage: float,
    TOTAL_CFS: int,
    window_size: int,
    cols_to_vary: List[str],
    case_id_name: str,
    activity_column_name: str,
    outcome_name: str,
    forbidden_map: Dict[str, List[str]],
    act_with_res,
    query_instances_by_case: Dict[Any, pd.DataFrame],
    ) -> Dict[Any, Tuple[str, str]]:
    
    """
    Generate DiCE recommendations as a dict {case_id: (next_activity, next_resource)}.
    Uses pre-built query_instances_by_case instead of dropping columns inside loop.
    """

    # --- Forbidden activities ---
    forbidden = set(forbidden_map.get(case_study, []))

    # --- Results dict ---
    rec: Dict[Any, Tuple[str, str]] = {}

    # --- Iterate cases ---
    for cid in pd.unique(test_data[case_id_name]):
        trace_df = test_log[test_log[case_id_name] == cid]
        # Take query instance from pre-built dict
        query_instance = _to_row_df(query_instances_by_case[cid])
        # Candidate next activities
        trace_history = trace_df[activity_column_name].tolist()
        # possible next activities (filter forbidden)
        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            rec[cid] = [] # No recommendation!
            continue

        # Predict baseline with NEXT_* blank 
        copy_qi = query_instance.copy()
        for col in ("NEXT_ACTIVITY", "NEXT_RESOURCE"):
            if col not in copy_qi.columns:
                copy_qi[col] = ""
            else:
                copy_qi.loc[:, col] = ""
        try:
            pred = predictive_model.predict(copy_qi)
            predicted_outcome = float(np.ravel(pred)[0])
            if not np.isfinite(predicted_time):
                predicted_outcome = 1.0
        except Exception:
            predicted_outcome = 1.0
            
        # Generate CFEs
        try:
            cfe_single_query = CFE_for_a_single_query(
                    explainer=explainer,
                    query_instance=query_instance,
                    y_predicted=predicted_time,
                    reduced_percentage=reduced_percentage,
                    total_CFs=TOTAL_CFS,
                    cols_to_vary=cols_to_vary,
                    next_possible_activity=poss,
                    act_with_res=act_with_res,
                    dice_method=dice_method,
                )
        except (Exception):
            cfe_single_query = -1

        # Decide recommendation
        if cfe_single_query == -1:
            rec[cid] = (query_instance['NEXT_ACTIVITY'].values[0], query_instance['NEXT_RESOURCE'].values[0]) # Getting Most Probable pair when No recommendation found!
            continue

        if dice_method == "genetic":
            try:
                _, best_act, best_res = get_best_pairs(cfe_single_query, outcome_name)
                rec[cid] = (best_act, best_res)
            except Exception:
                rec[cid] = []
    return rec


def kpi_computation_exhaustive(pos_acts, query_instance, model, act_with_res):
    result = {}
    for next_act in pos_acts:
        list_res = act_with_res[next_act]
        next_res = random.choice(list_res) #Randomly assign a resource
        temp_query_instance = query_instance.copy()
        temp_query_instance['NEXT_ACTIVITY'] = next_act
        temp_query_instance['NEXT_RESOURCE'] = ""
        predicted_total_time = model.predict(temp_query_instance)
        result[(next_act, next_res)] = predicted_total_time[0]
    return  min(result.items(), key=lambda x: x[1]) # (act, remaining_time)
    
def exhaustive_recommendations(train_data, test_log, test_data, predictive_model, transition_graph,
                               WINDOW_SIZE, case_id_name, activity_column_name, resource_column_name,
                               outcome_name, start_date_name, end_date_name):
    """
    This function generates the exhaustive recommendations by choosing the pair with smallest predicted outcome.
    """
    rec = {}
    
    forbidden = set(forbidden_map.get(case_study, []))

    def _to_row(qi):
        if isinstance(qi, pd.DataFrame):
            return qi.iloc[[0]].copy()
        elif isinstance(qi, pd.Series):
            return qi.to_frame().T
        elif isinstance(qi, dict):
            return pd.DataFrame([qi])
        else:
            raise TypeError("query_instance must be DataFrame, Series, or dict.")
    
    test_log_ids = test_df[case_id_name].unique()

    for i, cid in enumerate(tqdm.tqdm(test_log_ids)):
        # history of this case
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()
        current_execution = test_data[test_data[case_id_name] == idx]
        
        query_instance = current_execution.drop([case_id_name, start_date_name, end_date_name, "total_time", "remaining_time", "status", outcome_name], axis=1)

        # possible next activities (filter forbidden)
        poss = next_possible_activities(trace_history, transition_graph, WINDOW_SIZE)
        poss = [a for a in poss if a not in forbidden]
        
       
        (best_act, best_res), best_time = kpi_computation_exhaustive(next_possible_activity,                                                    query_instance, predictive_model, act_with_res)
        
        rec[cid] = (best_act, best_res)

    return rec
