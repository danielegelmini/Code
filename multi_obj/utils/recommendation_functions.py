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

# ---------------------------------------------------------------------------
# Utils for run_experiment.py
# ---------------------------------------------------------------------------
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
    grouped = df.groupby(activity_column_name)[resource_column_name].unique()
    forbidden = {"missing", "NotDef"}
    return {
        act: [res for res in resources if res not in forbidden]
        for act, resources in grouped.items()
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

# ---------------------------------------------------------------------------
# Utils for recommendation functions
# ---------------------------------------------------------------------------

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

def _to_row_df(x):
    """
    Converts the input query instance into a single-row pandas DataFrame.
    
    Parameters:
        x (pd.DataFrame, pd.Series, or dict): The query instance data to format.
        
    Returns:
        pd.DataFrame: A DataFrame containing exactly one row representing the query instance.
        
    Raises:
        TypeError: If the input is not a DataFrame, Series, or dictionary.
    """
    if isinstance(x, pd.DataFrame):
        return x.iloc[[0]] if len(x) > 1 else x
    if isinstance(x, pd.Series):
        return x.to_frame().T
    if isinstance(x, dict):
        return pd.DataFrame([x])
    raise TypeError(f"Unsupported query_instance type: {type(x)}")

def align_query_instance_with_model(query_instance, model):
    """
    Ensures the query instance has all the necessary columns expected by the predictive 
    model's transformation steps. Fills missing numerical columns with 0 and 
    categorical columns with an empty string.

    Parameters:
        query_instance (pd.DataFrame, pd.Series, or dict): The raw feature data for a specific case.
        model: The trained predictive pipeline, expected to have a "transformation" step.

    Returns:
        pd.DataFrame: A single-row DataFrame perfectly aligned with the model's required input schema.
    """
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

# ---------------------------------------------------------------------------
# Utils for Pareto search
# ---------------------------------------------------------------------------
def _build_valid_pairs(possible_actions: List[str], act_with_res: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """
    Generates all valid combinations of next activities and their corresponding allowed resources.

    Parameters:
        possible_actions (List[str]): A list of activities that can logically occur next.
        act_with_res (Dict[str, List[str]]): A mapping of activities to their allowed resources.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing valid (activity, resource) combinations.
    """
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
    """
    Evaluates a list of candidate (activity, resource) pairs by passing them through 
    the predictive models to estimate both the outcome and the required time.

    Parameters:
        candidate_pairs (List[Tuple[str, str]]): The combinations of (activity, resource) to evaluate.
        query_instance (pd.DataFrame, pd.Series, or dict): The current state features of the case.
        predictive_outcome_model: The trained model used to predict the target outcome.
        predictive_time_model: The trained model used to predict the total or remaining time.

    Returns:
        np.ndarray: A 2D numpy array where each row corresponds to a candidate pair, 
                    formatted as [predicted_outcome, predicted_total_time].
    """
    base_outcome_row = align_query_instance_with_model(query_instance, predictive_outcome_model).iloc[0].to_dict()
    base_time_row = align_query_instance_with_model(query_instance, predictive_time_model).iloc[0].to_dict()

    outcome_rows, time_rows = [], []
    for next_act, next_res in candidate_pairs:
        o_row = dict(base_outcome_row)
        o_row['NEXT_ACTIVITY'] = next_act
        o_row['NEXT_RESOURCE'] = next_res
        outcome_rows.append(o_row)

        t_row = dict(base_time_row)
        t_row['NEXT_ACTIVITY'] = next_act
        t_row['NEXT_RESOURCE'] = next_res
        time_rows.append(t_row)

    predicted_outcome = predictive_outcome_model.predict(pd.DataFrame(outcome_rows))
    predicted_total_time = predictive_time_model.predict(pd.DataFrame(time_rows))
    return np.column_stack([predicted_outcome, predicted_total_time])

# ---------------------------------------------------------------------------
# Exhaustive research
# ---------------------------------------------------------------------------
def exhaustive_pareto_search(
    query_instance,
    possible_actions,
    predictive_outcome_model,
    predictive_time_model,
    act_with_res,
):
    """
    Computes predictions for all valid combinations of possible next activities and resources 
    to build the complete search space for evaluation.

    Parameters:
        query_instance: The current state features of the case.
        possible_actions (List[str]): Allowed next activities based on the transition graph.
        predictive_outcome_model: The predictive model for the primary outcome.
        predictive_time_model: The predictive model for total/remaining time.
        act_with_res (Dict[str, List[str]]): Mapping of valid resources for each activity.

    Returns:
        List[Tuple[str, str, float, float]]: A list of tuples containing 
        (activity, resource, predicted_outcome, predicted_time) for all evaluated valid pairs.
    """
    valid_pairs = _build_valid_pairs(possible_actions, act_with_res)
    if not valid_pairs:
        return []

    objs = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
    return [
        (act, res, float(outcome), float(total_time))
        for (act, res), (outcome, total_time) in zip(valid_pairs, objs)
    ]


# ---------------------------------------------------------------------------
# NSGA-II (pymoo)
# ---------------------------------------------------------------------------
class _ActivityResourceProblem(Problem):
    """Problema pymoo a variabile intera: x in {0, ..., len(valid_pairs)-1}.
    f1 = -predicted_outcome (pymoo minimizza -> massimizza outcome)
    f2 =  predicted_total_time (minimizza tempo)

    Questa classe e' richiesta dall'API di pymoo (NSGA2 si aspetta una
    sottoclasse di Problem con un metodo _evaluate) — non e' un'incoerenza
    stilistica rispetto al metodo esaustivo: e' un thin wrapper attorno alla
    STESSA `_evaluate_candidates` usata da `generate_exhaustive_pareto_set`.
    """

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
    pop_size: int = 50,
    n_generations: int = 10,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    random_state: Optional[int] = None,
) -> List[Tuple[str, str, float, float]]:
    """
    Approximates the Pareto front of best (activity, resource) pairs using the NSGA-II 
    genetic algorithm. It attempts to simultaneously maximize outcome and minimize time.

    Parameters:
        query_instance: The current state features of the case.
        possible_actions (List[str]): Allowed next activities based on the transition graph.
        act_with_res (Dict[str, List[str]]): Mapping of valid resources for each activity.
        predictive_outcome_model: The predictive model for the primary outcome.
        predictive_time_model: The predictive model for total/remaining time.
        pop_size (int, optional): The population size for the genetic algorithm. Defaults to 50.
        n_generations (int, optional): The number of generations to evolve. Defaults to 10.
        crossover_rate (float, optional): The probability of crossover. Defaults to 0.9.
        mutation_rate (float, optional): The probability of mutation. Defaults to 0.3.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        List[Tuple[str, str, float, float]]: A list of tuples containing the Pareto-optimal 
        (activity, resource, predicted_outcome, predicted_time) pairs discovered by the algorithm.
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
        pop_size=pop_size, #initialize population
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_rate, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=mutation_rate, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    # Running the generations
    res = minimize(problem, algorithm, get_termination("n_gen", n_generations), seed=random_state, verbose=False)

    if res.X is None:
        return []
    # Winner extraction: convert the continuous solution to discrete indices and retrieve the corresponding (activity, resource) pairs
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

# ---------------------------------------------------------------------------
# Selection rules for the best action/resource pair from the Pareto set
# ---------------------------------------------------------------------------
def select_best_pareto_action(pareto_set):
    """Select the best action/resource pair from the Pareto set."""
    if not pareto_set:
        return None

    ideal_point = np.array([1.0, 0.0])  # Ideal point for max outcome and min time

    pareto_vals = np.array([[item[2], item[3]] for item in pareto_set], dtype=float)
    is_pareto = paretoset(pareto_vals, sense=["max", "min"])
    pareto_front = [item for item, keep in zip(pareto_set, is_pareto) if keep]
    pareto_front_vals = np.array([[item[2], item[3]] for item in pareto_front], dtype=float)

    distances = np.linalg.norm(pareto_front_vals - ideal_point, axis=1)
    best_index = np.argmin(distances)
    best_act, best_res = pareto_front[best_index][:2]
    return (best_act, best_res)

# ---------------------------------------------------------------------------
# Recommendation function for 'exhaustive' and 'nsga2'/'genetic' methods
# ---------------------------------------------------------------------------
def compute_recommendations(
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
    method: str = "exhaustive",
    pop_size: int = 50,
    n_generations: int = 10,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    random_state: Optional[int] = None,
) -> Dict[Any, Tuple[Optional[str], Optional[str]]]:
    """
    Unified recommendation function that supports both 'exhaustive' and
    'nsga2'/'genetic' methods. 
    """
    method = method.lower()
    forbidden = set(forbidden_map.get(case_study, []))
    rec: Dict[Any, Tuple[Optional[str], Optional[str]]] = {}

    for cid in tqdm.tqdm(pd.unique(test_data[case_id_name])):
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()

        query_instance = _to_row_df(query_instances_by_case[cid])

        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            rec[cid] = (None, None)
            continue

        if method in {"nsga2", "genetic"}:
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
        elif method == "exhaustive":
            pareto_front = exhaustive_pareto_search(
                query_instance,
                poss,
                predictive_outcome_model,
                predictive_time_model,
                act_with_res,
            )
        else:
            raise ValueError("Unknown method for recommendations: %s" % method)

        if not pareto_front:
            rec[cid] = (None, None)
            continue

        best_pair = select_best_pareto_action(pareto_front)
        rec[cid] = best_pair if best_pair is not None else (None, None)

    return rec