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

import pulp
from spopt.locate import PDispersion

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
    Generates a dictionary mapping each unique activity to a list of its associated unique resources.
    This mapping excludes 'missing' and 'NotDef' resources.

    Args:
        df (pandas.DataFrame): The DataFrame containing the event log data.
        activity_column_name (str): The name of the column containing activity names.
        resource_column_name (str): The name of the column containing resource names.

    Returns:
        dict: A dictionary in the format {activity: [unique resources]}.
    """
    grouped = df.groupby(activity_column_name)[resource_column_name].unique()
    forbidden = {"missing", "NotDef"}
    return {
        act: [res for res in resources if res not in forbidden]
        for act, resources in grouped.items()
    }

def build_query_instances(test_df, case_id_name):
    """
    Creates a dictionary mapping case IDs to their respective query instances.
    A query instance represents the last event before a prescription, excluding 
    specific columns like case ID, timestamps, labels, and outcome.

    Args:
        test_df (pandas.DataFrame): Dataframe containing only the query instances.
        case_id_name (str): The name of the column containing case IDs to be removed.

    Returns:
        dict: A dictionary where the keys are case IDs (str) and the values are dictionaries representing the instance features ({"feature_name": "value", ...}).
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
    Determines the list of possible next activities based on a transition graph and trace history.

    Compares the trace history (or its last WINDOW_SIZE activities, whichever
    is shorter) against the transition graph to find valid subsequent
    activities. If that exact-length prefix was never observed in training,
    falls back to progressively shorter suffixes of it (window-1, window-2,
    ..., down to just the single last activity), returning the first
    non-empty match. The empty prefix (transition_graph[""], i.e. "what
    typically starts a trace") is deliberately never used as a fallback --
    it answers a different question (how traces begin) than "what can follow
    this case's history", so it wouldn't be a meaningful recommendation here.
    A case only ends up with no possible next activity if not even its last
    activity alone was ever seen as a training prefix.

    Args:
        trace_history (list of str): The history of activities for a given case.
        transition_graph (dict): A dictionary mapping trace sequences (as strings) to possible next activities.
        WINDOW_SIZE (int): The maximum number of recent activities to consider when matching.

    Returns:
        list of str: A list of activities that can logically follow the current trace history.
    """
    window = trace_history if len(trace_history) <= WINDOW_SIZE else trace_history[-WINDOW_SIZE:]
    if not window:
        return []

    parsed_keys = [(ts, ts.split(", ")) for ts in transition_graph.keys()]

    for length in range(len(window), 0, -1):
        suffix = window[-length:]
        for ts, ts_to_list in parsed_keys:
            if ts_to_list == suffix:
                pos_acts = transition_graph[ts]
                if pos_acts:
                    return list(pos_acts)

    return []

def _to_row_df(x):
    """
    Converts the input query instance into a single-row pandas DataFrame.
    
    Args:
        x (pandas.DataFrame, pandas.Series, or dict): The query instance data to format.
        
    Returns:
        pandas.DataFrame: A DataFrame containing exactly one row representing the query instance.
        
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
    Ensures the query instance has all necessary columns expected by the predictive model's transformation steps. 
    Fills missing numerical columns with 0 and categorical columns with an empty string.

    Args:
        query_instance (pandas.DataFrame, pandas.Series, or dict): The raw feature data for a specific case.
        model (sklearn.pipeline.Pipeline): The trained predictive pipeline, expected to have a "transformation" step.

    Returns:
        pandas.DataFrame: A single-row DataFrame perfectly aligned with the model's required input schema.
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

    Args:
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

    Args:
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

    predicted_outcome = predictive_outcome_model.predict_proba(pd.DataFrame(outcome_rows))[:, 1]
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

    Args:
        query_instance (pandas.DataFrame, pandas.Series, or dict): The current state features of the case.
        possible_actions (list of str): Allowed next activities based on the transition graph.
        predictive_outcome_model (estimator): The predictive model for the primary outcome.
        predictive_time_model (estimator): The predictive model for total/remaining time.
        act_with_res (dict of str to list of str): Mapping of valid resources for each activity.

    Returns:
        list of tuple: A list of tuples containing (activity, resource, predicted_outcome, predicted_time) 
        for all evaluated valid pairs.
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
    """
    Integer-variable pymoo Problem subclass for evaluating activity and resource pairs.

    It maps an integer decision variable to a candidate pair in order to minimize the negated 
    predicted outcome (thereby maximizing it) and minimize the predicted total time.
    This acts as a wrapper around the `_evaluate_candidates` function to satisfy the pymoo API.
    """
    def __init__(self, valid_pairs, query_instance, predictive_outcome_model, predictive_time_model):
        """
        Initializes the pymoo problem definition for the NSGA-II algorithm.
        
        Args:
            valid_pairs (list of tuple): All valid (activity, resource) combinations.
            query_instance (pandas.DataFrame, pandas.Series, or dict): The current state features.
            predictive_outcome_model (estimator): Model to predict the target outcome.
            predictive_time_model (estimator): Model to predict the required time.
        """
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=0, xu=max(len(valid_pairs) - 1, 0), vtype=int)
        self.valid_pairs = valid_pairs
        self.query_instance = query_instance
        self.predictive_outcome_model = predictive_outcome_model
        self.predictive_time_model = predictive_time_model

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates the given population of candidate indices.

        Args:
            X (numpy.ndarray): The population of decision variables (indices).
            out (dict): The output dictionary where objective values ("F") are stored.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
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
    Finds the Pareto front of best (activity, resource) pairs using the NSGA-II genetic algorithm.
    It attempts to simultaneously maximize the predicted outcome and minimize the predicted time.

    Args:
        query_instance (pandas.DataFrame, pandas.Series, or dict): The current state features of the case.
        possible_actions (list of str): Allowed next activities based on the transition graph.
        act_with_res (dict of str to list of str): Mapping of valid resources for each activity.
        predictive_outcome_model (estimator): The predictive model for the primary outcome.
        predictive_time_model (estimator): The predictive model for total/remaining time.
        pop_size (int, optional): The population size for the genetic algorithm. Defaults to 50.
        n_generations (int, optional): The number of generations to evolve. Defaults to 10.
        crossover_rate (float, optional): The probability of crossover. Defaults to 0.9.
        mutation_rate (float, optional): The probability of mutation. Defaults to 0.3.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        list of tuple: A list of tuples containing the Pareto-optimal 
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
    """
    Selects the single best (activity, resource) pair from a computed Pareto set.
    Time is first transformed into (1 - time), so both objectives are maximized
    and the ideal point becomes [1.0, 1.0]. The true Pareto front is then computed
    on (outcome, 1 - time), and the selected point is the one on that front closest
    to the diagonal y = x, i.e. the most balanced trade-off between outcome and time.

    Args:
        pareto_set (list of tuple): A list of evaluated candidate tuples (activity, resource, outcome, time).

    Returns:
        tuple: The optimal (activity, resource) pair. Returns None if the set is empty.
    """
    if not pareto_set:
        return None

    outcome_vals = np.array([item[2] for item in pareto_set], dtype=float)
    inv_time_vals = 1.0 - np.array([item[3] for item in pareto_set], dtype=float)

    pareto_vals = np.column_stack([outcome_vals, inv_time_vals])
    is_pareto = paretoset(pareto_vals, sense=["max", "max"])
    pareto_front = [item for item, keep in zip(pareto_set, is_pareto) if keep]
    pareto_front_vals = pareto_vals[is_pareto]

    distances_to_diagonal = np.abs(pareto_front_vals[:, 0] - pareto_front_vals[:, 1])
    best_index = np.argmin(distances_to_diagonal)
    best_act, best_res = pareto_front[best_index][:2]
    return (best_act, best_res)

def select_top_k_pareto_actions(pareto_set, k=5):
    """
    Selects the k most representative (activity, resource) pairs from a computed
    Pareto set by exactly solving the max-min dispersion (p-dispersion) problem:
    the subset of k points is chosen so that the minimum pairwise distance among
    the selected points is maximized, so they are spread out as evenly as
    possible across the front instead of clustering in one region.

    The p-dispersion problem is solved exactly as a MILP using
    spopt.locate.PDispersion (see
    https://pysal.org/spopt/notebooks/p-dispersion.html), built from the
    pairwise Euclidean distance matrix of the front points and solved with
    the CBC solver bundled with pulp.

    Args:
        pareto_set (list of tuple): A list of evaluated candidate tuples (activity, resource, outcome, time).
        k (int): Number of points to select. Defaults to 5.

    Returns:
        list of tuple: Up to k selected (activity, resource) pairs. Empty list if pareto_set is empty.
    """
    if not pareto_set:
        return []

    outcome_vals = np.array([item[2] for item in pareto_set], dtype=float)
    inv_time_vals = 1.0 - np.array([item[3] for item in pareto_set], dtype=float)
    pareto_vals = np.column_stack([outcome_vals, inv_time_vals])

    is_pareto = paretoset(pareto_vals, sense=["max", "max"])
    front = [item for item, keep in zip(pareto_set, is_pareto) if keep]
    front_vals = pareto_vals[is_pareto]

    n_front = len(front)
    if n_front <= k:
        return [(item[0], item[1]) for item in front]

    diff = front_vals[:, None, :] - front_vals[None, :, :]
    cost_matrix = np.linalg.norm(diff, axis=2)

    p_dispersion = PDispersion.from_cost_matrix(cost_matrix, k)
    p_dispersion = p_dispersion.solve(pulp.PULP_CBC_CMD(msg=False))

    selected = [i for i, dv in enumerate(p_dispersion.fac_vars) if dv.varValue]

    return [(front[i][0], front[i][1]) for i in selected]

# ---------------------------------------------------------------------------
# Recommendation function for 'exhaustive' and 'nsga2' methods
# ---------------------------------------------------------------------------
def compute_recommendations_top_k(
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
    k: int = 5,
) -> List[Dict[Any, Tuple[Optional[str], Optional[str]]]]:
    """
    Generates next-step recommendations (activity and resource) for all cases in a test dataset.
    This unified function supports both 'exhaustive' search and 'nsga2' (genetic algorithm) 
    methods to find the optimal actions that maximize outcome and minimize time.

    Args:
        test_log (pandas.DataFrame): The full event log for the test cases.
        test_data (pandas.DataFrame): The dataset containing the latest state (query instances) for the test cases.
        case_study (str): The specific case study identifier, used to look up forbidden activities.
        case_id_name (str): The name of the column containing case IDs.
        activity_column_name (str): The name of the column containing activity names.
        transition_graph (dict): A mapping defining the valid next activities.
        window_size (int): The window size used to match the trace history against the transition graph.
        forbidden_map (dict): A dictionary mapping case studies to lists of forbidden activities.
        predictive_outcome_model (estimator): The predictive model for the primary outcome.
        predictive_time_model (estimator): The predictive model for required time.
        act_with_res (dict of str to list of str): Mapping of activities to their allowed resources.
        query_instances_by_case (dict): Precomputed query instances keyed by case ID.
        method (str, optional): The search method to use ("exhaustive" or "nsga2"). Defaults to "exhaustive".
        pop_size (int, optional): The population size (if using NSGA-II). Defaults to 50.
        n_generations (int, optional): The number of generations (if using NSGA-II). Defaults to 10.
        crossover_rate (float, optional): The crossover probability (if using NSGA-II). Defaults to 0.9.
        mutation_rate (float, optional): The mutation probability (if using NSGA-II). Defaults to 0.3.
        random_state (int, optional): Seed for reproducibility. Defaults to None.
        k (int, optional): The number of top recommendations to return for each case. Defaults to 5.

    Returns:
        list of dict: A list of length k; the j-th dict is the recommendations dictionary
        {case_id: (next_activity, next_resource)} for the j-th selected pair.
    """

    method = method.lower()
    forbidden = set(forbidden_map.get(case_study, []))
    rec_list: List[Dict[Any, Tuple[Optional[str], Optional[str]]]] = [dict() for _ in range(k)]

    for cid in tqdm.tqdm(pd.unique(test_data[case_id_name])):
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()

        query_instance = _to_row_df(query_instances_by_case[cid])

        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            for rec in rec_list:
                rec[cid] = (None, None)
            continue

        if method == "nsga2":
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
            for rec in rec_list:
                rec[cid] = (None, None)
            continue

        top_k_pairs = select_top_k_pareto_actions(pareto_front, k=k)
        for j, rec in enumerate(rec_list):
            rec[cid] = top_k_pairs[j] if j < len(top_k_pairs) else (None, None)

    return rec_list