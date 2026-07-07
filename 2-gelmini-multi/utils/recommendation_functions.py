
import tqdm
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
from paretoset import paretoset

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from utils.pre_processing_functions import convert_dtypes_bpi12

case_id_name = "case:concept:name"
activity_column_name = "concept:name"
end_date_name = "time:timestamp"
start_date_name = "start:timestamp"
resource_column_name = "org:resource"
outcome_name = "outcome"

NON_FEATURE_COLUMNS = {
    case_id_name,
    start_date_name,
    end_date_name,
    "total_time",
    "remaining_time",
    "label",
    "sigmoid_mm",
    "time_from_midnight",
    outcome_name,
}

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
    drop_cols = {case_id_name, start_date_name, end_date_name, "total_time", "remaining_time", "label", "sigmoid_mm", 'time_from_midnight', outcome_name}
    feature_columns = [c for c in test_df.columns if c not in drop_cols]
    query_instances_by_case = {
        row[case_id_name]: row[feature_columns].to_dict()
        for _, row in test_df.iterrows()
    }
    return query_instances_by_case

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

def _to_row_df(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[[0]] if len(x) > 1 else x
    if isinstance(x, pd.Series):
        return x.to_frame().T
    if isinstance(x, dict):
        return pd.DataFrame([x])
    raise TypeError(f"Unsupported query_instance type: {type(x)}")

def align_query_instance_with_model(query_instance, model):
    query_df = _to_row_df(query_instance).copy()
    if not hasattr(model, "named_steps") or "transformation" not in model.named_steps:
        return query_df

    transformation = model.named_steps["transformation"]
    numeric_cols = []
    categorical_cols = []
    for name, _, cols in transformation.transformers:
        if name == "num":
            if isinstance(cols, (list, tuple)):
                numeric_cols.extend(cols)
            else:
                numeric_cols.append(cols)
        elif name == "cat":
            if isinstance(cols, (list, tuple)):
                categorical_cols.extend(cols)
            else:
                categorical_cols.append(cols)

    required_cols = list(dict.fromkeys(numeric_cols + categorical_cols))
    missing_cols = [c for c in required_cols if c not in query_df.columns]
    for col in missing_cols:
        query_df[col] = 0 if col in numeric_cols else ""

    return query_df

def generate_pareto_set(
    query_instance,
    possible_actions,
    predictive_outcome_model,
    predictive_time_model,
    act_with_res,
):
    """Build the exhaustive Pareto set for each candidate action/resource pair."""
    pareto_set = []
    rows = []
    for next_act in possible_actions:
        for next_res in act_with_res.get(next_act, []):
            temp_query_instance = query_instance.copy()
            temp_query_instance['NEXT_ACTIVITY'] = next_act
            temp_query_instance['NEXT_RESOURCE'] = next_res
            rows.append(_to_row_df(temp_query_instance).iloc[0].to_dict())

    if not rows:
        return pareto_set

    temp_df = pd.DataFrame(rows)
    temp_df["predicted_outcome"] = predictive_outcome_model.predict(temp_df)
    temp_df["predicted_total_time"] = predictive_time_model.predict(temp_df)
    for _, row in temp_df.iterrows():
        pareto_set.append(
            (
                row['NEXT_ACTIVITY'],
                row['NEXT_RESOURCE'],
                float(row['predicted_outcome']),
                float(row['predicted_total_time']),
            )
        )
    return pareto_set

def select_best_pareto_action(pareto_set):
    """Select the best action/resource pair from the Pareto set."""
    if not pareto_set:
        return None

    pareto_vals = np.array([[item[2], item[3]] for item in pareto_set], dtype=float)
    is_pareto = paretoset(pareto_vals, sense=["max", "min"])
    pareto_front = [item for item, keep in zip(pareto_set, is_pareto) if keep]
    pareto_front_vals = np.array([[item[2], item[3]] for item in pareto_front], dtype=float)
    distances = np.linalg.norm(pareto_front_vals - np.array([1.0, 0.0]), axis=1)
    best_index = np.argmin(distances)
    best_act, best_res = pareto_front[best_index][:2]
    return (best_act, best_res)
    
def exhaustive_recommendations(test_log, # History of traces
                               test_data,
                               case_id_name,
                               activity_column_name,
                               case_study,
                               transition_graph,
                               window_size,
                               forbidden_map,
                               predictive_outcome_model,
                               predictive_time_model,
                               act_with_res):
    """
    This function generates the exhaustive recommendations by choosing the pair with smallest predicted remaining time and highest predicted outcome using a Pareto front approach.
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
    
    test_log_ids = test_log[case_id_name].unique()

    for i, cid in enumerate(tqdm.tqdm(test_log_ids)):
        # history of this case
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()
        current_execution = test_data[test_data[case_id_name] == cid]
        
        query_instance = current_execution.drop(
            columns=[c for c in NON_FEATURE_COLUMNS if c in current_execution.columns]
        )
        query_instance = align_query_instance_with_model(query_instance, predictive_time_model)


        # possible next activities (filter forbidden)
        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]

        pareto_set = generate_pareto_set(
            query_instance,
            poss,
            predictive_outcome_model,
            predictive_time_model,
            act_with_res,
        )
        best_pair = select_best_pareto_action(pareto_set)
        rec[cid] = best_pair if best_pair is not None else (None, None)

    return rec

### Genetic part
def _build_valid_pairs(possible_actions: List[str], act_with_res: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Enumera tutte le coppie (attivita', risorsa) valide per il caso corrente.
    Questa lista diventa il dominio discreto su cui pymoo ottimizza tramite
    un singolo indice intero: ogni indice in [0, N-1] e' per costruzione
    una coppia valida, quindi non servono crossover/mutazioni custom."""
    pairs: List[Tuple[str, str]] = []
    for act in possible_actions:
        for res in act_with_res.get(act, []):
            pairs.append((act, res))
    return pairs
 
 
def _evaluate_candidates(
    candidate_pairs: List[Tuple[str, str]],
    query_instance,
    predictive_outcome_model,
    predictive_time_model,
) -> np.ndarray:
    """Valuta in batch una lista di candidati (stesso pattern di generate_pareto_set).
 
    Returns
    -------
    np.ndarray shape (n_candidati, 2) colonne = [predicted_outcome, predicted_total_time]
    """
    rows = []
    for next_act, next_res in candidate_pairs:
        temp_query_instance = query_instance.copy()
        temp_query_instance['NEXT_ACTIVITY'] = next_act
        temp_query_instance['NEXT_RESOURCE'] = next_res
        rows.append(_to_row_df(temp_query_instance).iloc[0].to_dict())
 
    temp_df = pd.DataFrame(rows)
    predicted_outcome = predictive_outcome_model.predict(temp_df)
    predicted_total_time = predictive_time_model.predict(temp_df)
    return np.column_stack([predicted_outcome, predicted_total_time])
 
 
class _ActivityResourceProblem(Problem):
    """Problema pymoo a variabile intera: x in {0, ..., len(valid_pairs)-1}.
    f1 = -predicted_outcome (pymoo minimizza -> massimizza outcome)
    f2 =  predicted_total_time (minimizza tempo)"""
 
    def __init__(self, valid_pairs, query_instance, predictive_outcome_model, predictive_time_model):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=0, xu=max(len(valid_pairs) - 1, 0), vtype=int)
        self.valid_pairs = valid_pairs
        self.query_instance = query_instance
        self.predictive_outcome_model = predictive_outcome_model
        self.predictive_time_model = predictive_time_model
 
    def _evaluate(self, X, out, *args, **kwargs):
        idx = np.clip(np.round(X[:, 0]).astype(int), 0, len(self.valid_pairs) - 1)
        candidates = [self.valid_pairs[i] for i in idx]
        objs = _evaluate_candidates(candidates, self.query_instance, self.predictive_outcome_model, self.predictive_time_model)
        out["F"] = np.column_stack([-objs[:, 0], objs[:, 1]])
 
 
def nsga2_pareto_search(
    query_instance,
    possible_actions: List[str],
    act_with_res: Dict[str, List[str]],
    predictive_outcome_model,
    predictive_time_model,
    pop_size: int = 20,
    n_generations: int = 15,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    random_state: Optional[int] = None,
) -> List[Tuple[str, str, float, float]]:
    """
    Esegue NSGA-II (pymoo) sullo spazio discreto (NEXT_ACTIVITY, NEXT_RESOURCE)
    e restituisce il fronte di Pareto finale nello stesso formato prodotto da
    `generate_pareto_set`, cosi' da poter riusare senza modifiche
    `select_best_pareto_action`.
    """
    valid_pairs = _build_valid_pairs(possible_actions, act_with_res)
    if not valid_pairs:
        return []
 
    if len(valid_pairs) == 1:
        objs = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
        act, res = valid_pairs[0]
        return [(act, res, float(objs[0, 0]), float(objs[0, 1]))]
 
    problem = _ActivityResourceProblem(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
 
    algorithm = NSGA2(
        pop_size=min(pop_size, len(valid_pairs)),
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_rate, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=mutation_rate, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
 
    res = minimize(problem, algorithm, get_termination("n_gen", n_generations), seed=random_state, verbose=False)
    if res.X is None:
        return []
 
    X, F = np.atleast_2d(res.X), np.atleast_2d(res.F)
    pareto_set, seen = [], set()
    for i in range(X.shape[0]):
        xi = int(np.clip(round(X[i, 0]), 0, len(valid_pairs) - 1))
        if xi in seen:
            continue
        seen.add(xi)
        act, resource = valid_pairs[xi]
        pareto_set.append((act, resource, float(-F[i, 0]), float(F[i, 1])))
    return pareto_set
 
 
def nsga2_recommendations(
    test_log: pd.DataFrame,
    test_data: pd.DataFrame,
    case_study: str,
    case_id_name: str,
    activity_column_name: str,
    transition_graph,
    window_size: int,
    forbidden_map: Dict[str, List[str]],
    predictive_outcome_model,
    predictive_time_model,
    act_with_res: Dict[str, List[str]],
    query_instances_by_case: Dict[Any, Any],
    pop_size: int = 20,
    n_generations: int = 15,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    random_state: Optional[int] = None,
) -> Dict[Any, Tuple[Optional[str], Optional[str]]]:
    """
    Sostituisce `genetic_recommendations` (DiCE): stessa struttura di
    `exhaustive_recommendations`, ma il fronte di Pareto per ogni caso viene
    generato con NSGA-II (pymoo) invece dell'enumerazione esaustiva. La
    scelta finale usa la STESSA regola gia' presente nel progetto
    (`select_best_pareto_action`).
    """
    forbidden = set(forbidden_map.get(case_study, []))
    rec: Dict[Any, Tuple[Optional[str], Optional[str]]] = {}
 
    for cid in tqdm.tqdm(pd.unique(test_data[case_id_name])):
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()
 
        query_instance = _to_row_df(query_instances_by_case[cid])
        query_instance = align_query_instance_with_model(query_instance, predictive_time_model)
 
        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            rec[cid] = (None, None)
            continue
 
        pareto_front = nsga2_pareto_search(
            query_instance=query_instance,
            possible_actions=poss,
            act_with_res=act_with_res,
            predictive_outcome_model=predictive_outcome_model,
            predictive_time_model=predictive_time_model,
            pop_size=pop_size,
            n_generations=n_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            random_state=random_state,
        )
 
        if not pareto_front:
            rec[cid] = (None, None)
            continue
 
        best_pair = select_best_pareto_action(pareto_front)
        rec[cid] = best_pair if best_pair is not None else (None, None)
 
    return rec