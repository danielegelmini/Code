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

# Pareto objectives -- method "exhaustive" (the only one used in production;
# "nsga2" below is untouched, see its own docstrings):
#   #1  outcome probability, maximised -- the TEMPERATURE-CALIBRATED P(y=positive)
#       (predict_outcome_proba).
#   #2  predicted remaining time, minimised (plotted/optimised as 1 - time).
# Before these two objectives are turned into a Pareto front, every candidate
# (activity, resource) pair is scored against a "no recommendation" BASELINE --
# NOT a synthetic or statistical pair, but the real transition that already
# happened one prefix step earlier for this same case (build_baseline_instances):
# if the case's current prefix is e_1,...,e_k (what "metodo" evaluates candidate
# e_{k+1} against), the baseline re-evaluates the model on prefix e_1,...,e_{k-1}
# with its own real NEXT_ACTIVITY/NEXT_RESOURCE, which is exactly e_k -- with the
# probability that the candidate beats that baseline on each objective
# (_compute_confidence_probabilities, estimated empirically from the SAME fitted
# models' virtual-ensemble members for baseline and candidate, so their
# correlation is captured for free -- no independence/normality assumption).
# Candidates whose probability falls below gamma_cls (outcome) or
# gamma_reg (time) are dropped (_filter_by_confidence) BEFORE the Pareto front
# is built, so predictive uncertainty acts as a pre-filter/confidence gate
# rather than as a third Pareto axis. The two probabilities are carried
# through as diagnostics (never written to the recommendation CSVs fed to the
# simulation, only to the separate `_objectives.csv` sidecar).
#
# NSGA2 (nsga2_pareto_search / _ActivityResourceProblem) is NOT part of this
# confidence-as-KPI change and keeps its original 3 objectives (outcome, time,
# and the regressor's TOTAL predictive std, minimised) with no confidence
# filter -- see those functions' own docstrings.

_UNCERTAINTY_WARNED = False
_CALIBRATION_WARNED = False


def _minmax_columns(mat):
    """Min-max scale each column of a 2-D array to [0, 1]; constant columns -> 0."""
    mat = np.asarray(mat, dtype=float)
    lo = mat.min(axis=0)
    span = mat.max(axis=0) - lo
    span[span == 0] = 1.0
    return (mat - lo) / span


def predict_time_and_uncertainty(predictive_time_model, rows_df):
    """
    Run the time model on rows_df and return (mean, uncertainty), both 1-D
    numpy arrays.

    `mean` is the predicted 'sigmoid_mm' remaining time -- exactly what
    predictive_time_model.predict(rows_df) returned before. `uncertainty` is the
    recalibrated TOTAL predictive standard deviation of that prediction
    (aleatoric + epistemic).

    Works whether predictive_time_model is a bare estimator or an sklearn
    Pipeline whose final "prediction" step is an UncertaintyRegressor. If the
    model has no uncertainty support (an older RMSE/MAE model), `uncertainty`
    comes back as all zeros and a one-time warning is printed, so the rest of
    the pipeline still runs with the third objective effectively disabled.
    """
    global _UNCERTAINTY_WARNED

    predictor = predictive_time_model
    predictor_input = rows_df
    if hasattr(predictive_time_model, "named_steps"):
        steps = predictive_time_model.named_steps
        predictor = steps.get("prediction", predictive_time_model)
        if "transformation" in steps:
            predictor_input = steps["transformation"].transform(rows_df)

    if hasattr(predictor, "predict_uncertainty"):
        out = predictor.predict_uncertainty(predictor_input)
        return np.asarray(out["mean"], dtype=float), np.asarray(out["std"], dtype=float)

    mean = np.asarray(predictive_time_model.predict(rows_df), dtype=float)
    if not _UNCERTAINTY_WARNED:
        print(
            "[recommendation_functions] time model exposes no uncertainty; the "
            "third Pareto objective is disabled (all zeros). Retrain the time "
            "model with loss_function='RMSEWithUncertainty'."
        )
        _UNCERTAINTY_WARNED = True
    return mean, np.zeros_like(mean)

def predict_outcome_proba(predictive_outcome_model, rows_df):
    """
    Return P(y = positive class) for rows_df as a 1-D numpy array, using the
    temperature-calibrated probability when the outcome model supports it.

    Objective #1 of the Pareto front is "maximise the outcome probability". The
    classifier is trained with posterior_sampling=True and its UncertaintyClassifier
    wrapper carries a temperature-scaling scalar T (Guo et al. 2017) fitted on a
    held-out slice: p_cal = sigmoid(logit(p) / T). predict_proba() stays raw, so
    this is the single deliberate place where the recommendation pipeline switches
    to the calibrated probability. T-scaling is monotone -> it does not reorder
    candidates by probability, but it does change the magnitudes that feed the
    min-max normalisation and the p-dispersion distances used to pick the k
    diverse pairs, so the front is built on honest (better-calibrated) probabilities.

    Falls back to the raw predict_proba()[:, 1] (with a one-time warning) for an
    older outcome model without uncertainty support.
    """
    global _CALIBRATION_WARNED

    predictor = predictive_outcome_model
    predictor_input = rows_df
    if hasattr(predictive_outcome_model, "named_steps"):
        steps = predictive_outcome_model.named_steps
        predictor = steps.get("prediction", predictive_outcome_model)
        if "transformation" in steps:
            predictor_input = steps["transformation"].transform(rows_df)

    if hasattr(predictor, "predict_uncertainty"):
        out = predictor.predict_uncertainty(predictor_input)
        return np.asarray(out["proba_calibrated"], dtype=float)

    if not _CALIBRATION_WARNED:
        print(
            "[recommendation_functions] outcome model exposes no temperature "
            "calibration; objective #1 uses the raw predict_proba. Retrain the "
            "outcome model with posterior_sampling=True to calibrate it."
        )
        _CALIBRATION_WARNED = True
    return np.asarray(predictive_outcome_model.predict_proba(rows_df), dtype=float)[:, 1]


def predict_time_members(predictive_time_model, rows_df):
    """
    Return the per-virtual-ensemble-member predicted mean time for rows_df,
    as a 2-D numpy array of shape (n_rows, ve_count) -- see
    UncertaintyRegressor.predict_members(). Used by
    _compute_confidence_probabilities() to compare a baseline row and a
    candidate row member-by-member.

    Falls back to a single-column array of the raw point prediction (a
    "1-member ensemble") for an older time model without uncertainty
    support, reusing the same one-time warning as predict_time_and_uncertainty.
    """
    global _UNCERTAINTY_WARNED

    predictor = predictive_time_model
    predictor_input = rows_df
    if hasattr(predictive_time_model, "named_steps"):
        steps = predictive_time_model.named_steps
        predictor = steps.get("prediction", predictive_time_model)
        if "transformation" in steps:
            predictor_input = steps["transformation"].transform(rows_df)

    if hasattr(predictor, "predict_members"):
        return np.asarray(predictor.predict_members(predictor_input), dtype=float)

    if not _UNCERTAINTY_WARNED:
        print(
            "[recommendation_functions] time model exposes no uncertainty; the "
            "confidence-as-KPI filter degrades to a single-member ensemble "
            "(point prediction only). Retrain the time model with "
            "loss_function='RMSEWithUncertainty'."
        )
        _UNCERTAINTY_WARNED = True
    mean = np.asarray(predictive_time_model.predict(rows_df), dtype=float)
    return mean.reshape(-1, 1)


def predict_outcome_members(predictive_outcome_model, rows_df):
    """
    Return the per-virtual-ensemble-member CALIBRATED P(y=positive) for
    rows_df, as a 2-D numpy array of shape (n_rows, ve_count) -- see
    UncertaintyClassifier.predict_members(). Used by
    _compute_confidence_probabilities() to compare a baseline row and a
    candidate row member-by-member.

    Falls back to a single-column array of the raw predict_proba point
    prediction (a "1-member ensemble") for an older outcome model without
    uncertainty support, reusing the same one-time warning as predict_outcome_proba.
    """
    global _CALIBRATION_WARNED

    predictor = predictive_outcome_model
    predictor_input = rows_df
    if hasattr(predictive_outcome_model, "named_steps"):
        steps = predictive_outcome_model.named_steps
        predictor = steps.get("prediction", predictive_outcome_model)
        if "transformation" in steps:
            predictor_input = steps["transformation"].transform(rows_df)

    if hasattr(predictor, "predict_members"):
        return np.asarray(predictor.predict_members(predictor_input), dtype=float)

    if not _CALIBRATION_WARNED:
        print(
            "[recommendation_functions] outcome model exposes no temperature "
            "calibration; the confidence-as-KPI filter degrades to a "
            "single-member ensemble (raw predict_proba only). Retrain the "
            "outcome model with posterior_sampling=True to calibrate it."
        )
        _CALIBRATION_WARNED = True
    proba = np.asarray(predictive_outcome_model.predict_proba(rows_df), dtype=float)[:, 1]
    return proba.reshape(-1, 1)

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


def build_baseline_instances(test_df, case_id_name):
    """
    Creates the confidence-as-KPI "no recommendation" baseline instance for
    each case: the query instance ONE prefix step shorter than
    build_query_instances' (the second-to-last row for that case instead of
    the last), used completely AS-IS -- including its own NEXT_ACTIVITY /
    NEXT_RESOURCE, which are exactly the activity and resource that really
    happened next (i.e. the last step of the "metodo" prefix, e_k, for a
    case whose current prefix is e_1,...,e_k).

    In other words: rather than asking the model to score a synthetic or
    statistical "no recommendation" pair, this re-evaluates the model on the
    real, already-observed transition e_{k-1} -> e_k, evaluated one prefix
    step earlier than the candidate recommendations for e_1,...,e_k -> e_{k+1}.
    This only works because test_df has one row per prefix length for every
    case (not just the final one) -- confirmed rows are in chronological
    order within each case, and that each row's own NEXT_ACTIVITY/
    NEXT_RESOURCE already equal the following row's concept:name/org:resource
    (see pre_processing_functions.add_next_act_res).

    IMPORTANT: this is NOT the same dataframe passed to build_query_instances
    (that one -- called `test_data` throughout this codebase -- is loaded
    from test_log_with_last_act.csv and has only ONE row per case, the final
    query instance). Pass `test_log` instead (load_case_study()'s third
    return value, "all prefixes in test data") -- confirmed to have one row
    per prefix length per case, the same feature columns, and its own
    last row per case matching test_data's row for that case exactly.

    Args:
        test_df (pandas.DataFrame): test_log (NOT test_data) -- the
            multi-row-per-case dataframe with one row per prefix length.
        case_id_name (str): The name of the column containing case IDs.

    Returns:
        dict: {case_id: {"feature_name": value, ...}}. Cases with fewer than
        2 rows in test_df (no earlier prefix to fall back to) are simply
        absent from the returned dict -- callers must treat a missing case
        as "no baseline available" rather than assuming every case has one.
    """
    drop_cols = {case_id_name, start_date_name, end_date_name, "total_time", "remaining_time", "label", "sigmoid_mm", 'time_from_midnight', outcome_name}
    feature_columns = [c for c in test_df.columns if c not in drop_cols]
    baseline_instances_by_case = {}
    for cid, group in test_df.groupby(case_id_name, sort=False):
        if len(group) < 2:
            continue
        baseline_instances_by_case[cid] = group.iloc[-2][feature_columns].to_dict()
    return baseline_instances_by_case

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


def _build_candidate_rows(
    candidate_pairs: List[Tuple[str, str]],
    query_instance,
    predictive_outcome_model,
    predictive_time_model,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the outcome-model and time-model input rows for a batch of
    candidate (activity, resource) pairs applied to the same query_instance
    (prefix state), one row per pair, in the same order as candidate_pairs.

    Shared by _evaluate_candidates (point predictions) and
    _compute_confidence_probabilities (per-ensemble-member predictions), so
    both score candidates on exactly the same rows.

    Args:
        candidate_pairs (List[Tuple[str, str]]): The combinations of (activity, resource) to evaluate.
        query_instance (pd.DataFrame, pd.Series, or dict): The current state features of the case.
        predictive_outcome_model: The trained model used to predict the target outcome.
        predictive_time_model: The trained model used to predict the total or remaining time.

    Returns:
        (outcome_rows_df, time_rows_df): one row per candidate pair each.
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

    return pd.DataFrame(outcome_rows), pd.DataFrame(time_rows)


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
                    formatted as [predicted_outcome, predicted_total_time,
                    predicted_uncertainty]. predicted_outcome is the
                    temperature-calibrated P(y=positive) (predict_outcome_proba);
                    predicted_uncertainty is the recalibrated total std of the
                    time prediction.
    """
    outcome_rows_df, time_rows_df = _build_candidate_rows(
        candidate_pairs, query_instance, predictive_outcome_model, predictive_time_model
    )
    predicted_outcome = predict_outcome_proba(predictive_outcome_model, outcome_rows_df)
    predicted_total_time, predicted_uncertainty = predict_time_and_uncertainty(
        predictive_time_model, time_rows_df
    )
    return np.column_stack([predicted_outcome, predicted_total_time, predicted_uncertainty])


def _compute_confidence_probabilities(
    candidate_pairs: List[Tuple[str, str]],
    baseline_row,
    query_instance,
    predictive_outcome_model,
    predictive_time_model,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every candidate (activity, resource) pair, estimate the probability
    that it beats the "no recommendation" baseline on each objective.

    The baseline is NOT a synthetic or statistical pair: it is the model's
    prediction on the real, already-observed transition that happened one
    prefix step earlier for this same case (see build_baseline_instances) --
    baseline_row already carries its own native NEXT_ACTIVITY/NEXT_RESOURCE,
    so it is evaluated completely as-is, with no pair substitution.

    Both the baseline and every candidate are scored member-by-member on the
    SAME fitted models' virtual-ensemble members (predict_outcome_members /
    predict_time_members), so the probability is the empirical fraction of
    members for which the candidate wins. Because baseline and candidates
    come from literally the same model, this captures whatever correlation
    exists between them automatically -- no independence assumption, no
    normality assumption.

    Args:
        candidate_pairs: list of (activity, resource) tuples to score.
        baseline_row: this case's baseline instance (build_baseline_instances
            output for one case) -- a dict-like row with its own
            NEXT_ACTIVITY/NEXT_RESOURCE already set to the real e_k.
        query_instance: the prefix/query instance for this case (e_1,...,e_k),
            against which candidate_pairs are evaluated as e_{k+1}.
        predictive_outcome_model, predictive_time_model: fitted pipelines.

    Returns:
        (prob_outcome_better, prob_time_better): two 1-D numpy arrays,
        aligned with candidate_pairs. prob_outcome_better[i] is the fraction
        of outcome-model members for which candidate i's calibrated outcome
        probability exceeds the baseline's; prob_time_better[i] is the
        fraction of time-model members for which candidate i's predicted
        time is BELOW the baseline's (equivalently, 1 - time is above it).
    """
    baseline_outcome_row = align_query_instance_with_model(baseline_row, predictive_outcome_model)
    baseline_time_row = align_query_instance_with_model(baseline_row, predictive_time_model)
    candidate_outcome_rows, candidate_time_rows = _build_candidate_rows(
        candidate_pairs, query_instance, predictive_outcome_model, predictive_time_model
    )
    outcome_rows_df = pd.concat([baseline_outcome_row, candidate_outcome_rows], ignore_index=True)
    time_rows_df = pd.concat([baseline_time_row, candidate_time_rows], ignore_index=True)

    outcome_members = predict_outcome_members(predictive_outcome_model, outcome_rows_df)
    time_members = predict_time_members(predictive_time_model, time_rows_df)

    baseline_outcome_members, candidate_outcome_members = outcome_members[0], outcome_members[1:]
    baseline_time_members, candidate_time_members = time_members[0], time_members[1:]

    prob_outcome_better = (candidate_outcome_members > baseline_outcome_members[None, :]).mean(axis=1)
    prob_time_better = (candidate_time_members < baseline_time_members[None, :]).mean(axis=1)

    return prob_outcome_better, prob_time_better


def _filter_by_confidence(
    prob_outcome_better: np.ndarray,
    prob_time_better: np.ndarray,
    gamma_cls: float,
    gamma_reg: float,
) -> np.ndarray:
    """
    Boolean mask selecting the candidates whose probability of beating the
    baseline meets both confidence thresholds.

    Args:
        prob_outcome_better, prob_time_better: outputs of
            _compute_confidence_probabilities(), aligned with the same
            candidate_pairs.
        gamma_cls: minimum required prob_outcome_better.
        gamma_reg: minimum required prob_time_better.

    Returns:
        np.ndarray of bool.
    """
    return (np.asarray(prob_outcome_better) >= gamma_cls) & (np.asarray(prob_time_better) >= gamma_reg)

# ---------------------------------------------------------------------------
# Exhaustive research
# ---------------------------------------------------------------------------
def exhaustive_pareto_search(
    query_instance,
    possible_actions,
    predictive_outcome_model,
    predictive_time_model,
    act_with_res,
    baseline_row=None,
    gamma_cls=0.5,
    gamma_reg=0.5,
):
    """
    Computes predictions for all valid combinations of possible next activities and resources,
    filters them by confidence against a "no recommendation" baseline, and returns the
    survivors for the caller to build a 2D Pareto front from (outcome, 1 - time).

    The confidence filter (see _compute_confidence_probabilities /
    _filter_by_confidence) is applied here, BEFORE the caller builds the
    Pareto front, so predictive uncertainty acts as a pre-filter/confidence
    gate rather than as a third Pareto objective. It is skipped entirely
    (every valid pair is kept, prob_outcome_better = prob_time_better = 1.0)
    when baseline_row is None -- lets this function still be called without
    it (e.g. multi_obj/3_tune_nsga2_params.py's NSGA2-vs-exhaustive benchmark,
    which is untouched by this change; or a case with no valid baseline, see
    build_baseline_instances).

    Args:
        query_instance (pandas.DataFrame, pandas.Series, or dict): The current state features of the case.
        possible_actions (list of str): Allowed next activities based on the transition graph.
        predictive_outcome_model (estimator): The predictive model for the primary outcome.
        predictive_time_model (estimator): The predictive model for total/remaining time.
        act_with_res (dict of str to list of str): Mapping of valid resources for each activity.
        baseline_row (dict-like, optional): this case's confidence-filter baseline instance
            (see build_baseline_instances) -- the real transition one prefix step earlier for
            this case, used as-is with its own native NEXT_ACTIVITY/NEXT_RESOURCE. None disables
            the confidence filter.
        gamma_cls (float, optional): minimum required P(candidate outcome beats baseline). Defaults to 0.5.
        gamma_reg (float, optional): minimum required P(candidate time beats baseline). Defaults to 0.5.

    Returns:
        list of tuple: (activity, resource, predicted_outcome, predicted_time,
        predicted_uncertainty, prob_outcome_better, prob_time_better) for
        every valid pair that passed the confidence filter. When
        baseline_row is None, no comparison was actually made -- every pair
        is kept (keep_mask all True) but prob_outcome_better/prob_time_better
        are NaN, not 1.0, so downstream diagnostics never read "no baseline"
        as "100% confident" (see compute_recommendations_top_k's
        "ok_no_baseline" status for the same distinction).
    """
    valid_pairs = _build_valid_pairs(possible_actions, act_with_res)
    if not valid_pairs:
        return []

    objs = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)

    if baseline_row is not None:
        prob_outcome_better, prob_time_better = _compute_confidence_probabilities(
            valid_pairs, baseline_row, query_instance,
            predictive_outcome_model, predictive_time_model,
        )
        keep_mask = _filter_by_confidence(prob_outcome_better, prob_time_better, gamma_cls, gamma_reg)
    else:
        prob_outcome_better = np.full(len(valid_pairs), np.nan, dtype=float)
        prob_time_better = np.full(len(valid_pairs), np.nan, dtype=float)
        keep_mask = np.ones(len(valid_pairs), dtype=bool)

    return [
        (act, res, float(outcome), float(total_time), float(uncertainty), float(p_out), float(p_time))
        for (act, res), (outcome, total_time, uncertainty), p_out, p_time, keep
        in zip(valid_pairs, objs, prob_outcome_better, prob_time_better, keep_mask)
        if keep
    ]


# ---------------------------------------------------------------------------
# NSGA-II (pymoo)
# ---------------------------------------------------------------------------
class _ActivityResourceProblem(Problem):
    """
    Integer-variable pymoo Problem subclass for evaluating activity and resource pairs.

    It maps an integer decision variable to a candidate pair in order to minimize the negated
    predicted outcome (thereby maximizing it), the predicted total time, and the predicted
    uncertainty of that time. This acts as a wrapper around the `_evaluate_candidates` function
    to satisfy the pymoo API.
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
        super().__init__(n_var=1, n_obj=3, n_constr=0, xl=0, xu=max(len(valid_pairs) - 1, 0), vtype=int)
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
        # minimise: -outcome, time, uncertainty
        out["F"] = np.column_stack([-objs[:, 0], objs[:, 1], objs[:, 2]])


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
) -> List[Tuple[str, str, float, float, float]]:
    """
    Finds the Pareto front of best (activity, resource) pairs using the NSGA-II genetic algorithm.
    It simultaneously maximizes the predicted outcome and minimizes both the predicted time and
    the predictive uncertainty of that time.

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
        (activity, resource, predicted_outcome, predicted_time,
        predicted_uncertainty) pairs discovered by the algorithm.
    """
    valid_pairs = _build_valid_pairs(possible_actions, act_with_res)
    if not valid_pairs:
        return []

    if len(valid_pairs) == 1:
        objs = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
        act, res = valid_pairs[0]
        return [(act, res, float(objs[0, 0]), float(objs[0, 1]), float(objs[0, 2]))]

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
        pareto_set.append((act, resource, float(-F[i, 0]), float(F[i, 1]), float(F[i, 2])))
    return pareto_set

# ---------------------------------------------------------------------------
# Selection rules for the best action/resource pair from the Pareto set
# ---------------------------------------------------------------------------
def _front_objective_matrix(pareto_set):
    """(..., outcome, time, ...) tuples -> raw objective matrix plus the paretoset sense list.

    Dispatches on tuple length: nsga2_pareto_search's 5-tuples (activity,
    resource, outcome, time, uncertainty) keep the ORIGINAL 3-objective
    [outcome, 1 - time, uncertainty] behaviour (senses max/max/min) -- NSGA2
    is untouched by the confidence-as-KPI change. exhaustive_pareto_search's
    7-tuples (..., prob_outcome_better, prob_time_better) use the new
    2-objective [outcome, 1 - time] behaviour (senses max/max), since the
    confidence filter has already been applied upstream and predictive
    uncertainty is no longer a Pareto axis for that method."""
    outcome_vals = np.array([item[2] for item in pareto_set], dtype=float)
    inv_time_vals = 1.0 - np.array([item[3] for item in pareto_set], dtype=float)

    if len(pareto_set[0]) >= 7:
        return np.column_stack([outcome_vals, inv_time_vals]), ["max", "max"]

    uncertainty_vals = np.array([item[4] for item in pareto_set], dtype=float)
    return np.column_stack([outcome_vals, inv_time_vals, uncertainty_vals]), ["max", "max", "min"]


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
        pareto_set (list of tuple): evaluated candidate tuples
            (activity, resource, outcome, time, uncertainty).
        k (int): Number of points to select. Defaults to 5.

    Returns:
        list of tuple: Up to k selected (activity, resource) pairs. Empty list if pareto_set is empty.
    """
    if not pareto_set:
        return []

    raw_vals, sense = _front_objective_matrix(pareto_set)  # [outcome, 1 - time, uncertainty]
    is_pareto = paretoset(raw_vals, sense=sense)
    front = [item for item, keep in zip(pareto_set, is_pareto) if keep]
    front_vals = raw_vals[is_pareto]

    n_front = len(front)
    if n_front <= k:
        return [(item[0], item[1]) for item in front]

    # Min-max normalize the three objectives before measuring pairwise
    # distances, so the p-dispersion spread is even across all three axes
    # rather than dominated by whichever objective has the widest raw range
    # (uncertainty is on a different scale than outcome / 1 - time).
    norm = _minmax_columns(front_vals)
    diff = norm[:, None, :] - norm[None, :, :]
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
    baseline_instances_by_case: Optional[Dict[Any, Any]] = None,
    gamma_cls: float = 0.5,
    gamma_reg: float = 0.5,
) -> Tuple[
    List[Dict[Any, Tuple[Optional[str], Optional[str]]]],
    List[Dict[Any, Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]],
    Dict[Any, str],
]:
    """
    Generates next-step recommendations (activity and resource) for all cases in a test dataset.
    This unified function supports both 'exhaustive' search and 'nsga2' (genetic algorithm)
    methods to find the optimal actions that maximize outcome and minimize time.

    For method="exhaustive", candidates are additionally filtered by
    confidence against a "no recommendation" baseline -- the real transition
    that happened one prefix step earlier for this same case, see
    build_baseline_instances -- before the Pareto front is built (see
    exhaustive_pareto_search()). method="nsga2" is untouched by this filter
    (baseline_instances_by_case/gamma_* are simply not passed to it).

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
        baseline_instances_by_case (dict, optional): build_baseline_instances() output. Only used
            for method="exhaustive"; None (or a missing case, see build_baseline_instances) disables
            the confidence filter for that case (every valid pair is kept, as before this change).
        gamma_cls (float, optional): minimum required P(candidate outcome beats baseline). Defaults to 0.5.
        gamma_reg (float, optional): minimum required P(candidate time beats baseline). Defaults to 0.5.

    Returns:
        tuple (rec_list, obj_list, status_by_case):
          - rec_list[j] is {case_id: (next_activity, next_resource)} for the
            j-th selected pair (this is what gets written to the recommendation
            CSVs and later fed to the simulation).
          - obj_list[j] is {case_id: (pred_outcome, pred_time, pred_uncertainty,
            prob_outcome_better, prob_time_better)} for that same pair --
            diagnostic only, never passed to the simulation. prob_outcome_better/
            prob_time_better are None for method="nsga2" (it does not compute
            them) and NaN (not 1.0 -- no comparison was made) for
            method="exhaustive" cases with no baseline available (status
            "ok_no_baseline" below). Missing entries are (None, None) /
            (None, None, None, None, None).
          - status_by_case is {case_id: status}, one entry per case (not per
            rank j), with status one of: "ok" (confidence filter genuinely
            applied); "ok_no_baseline" (method="exhaustive" only -- a valid
            recommendation was produced, but no baseline was available for
            this case, e.g. a trace of length 1, see build_baseline_instances,
            so the confidence filter was skipped entirely rather than
            producing a false "confident" result); "no_possible_actions" (no
            legal next activity, or no valid (activity, resource) pair for
            it); or "no_confident_recommendation" (valid pairs existed but
            none passed the confidence filter -- only possible for
            method="exhaustive" with a baseline available for that case).
    """

    method = method.lower()
    forbidden = set(forbidden_map.get(case_study, []))
    rec_list: List[Dict[Any, Tuple[Optional[str], Optional[str]]]] = [dict() for _ in range(k)]
    obj_list: List[Dict[Any, Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]] = [dict() for _ in range(k)]
    status_by_case: Dict[Any, str] = {}

    def _fill_empty(cid, status):
        for rec, obj in zip(rec_list, obj_list):
            rec[cid] = (None, None)
            obj[cid] = (None, None, None, None, None)
        status_by_case[cid] = status

    for cid in tqdm.tqdm(pd.unique(test_data[case_id_name])):
        trace_df = test_log[test_log[case_id_name] == cid]
        trace_history = trace_df[activity_column_name].tolist()

        query_instance = _to_row_df(query_instances_by_case[cid])

        poss = next_possible_activities(trace_history, transition_graph, window_size)
        poss = [a for a in poss if a not in forbidden]
        if not poss:
            _fill_empty(cid, "no_possible_actions")
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
            if not pareto_front:
                _fill_empty(cid, "no_possible_actions")
                continue
        elif method == "exhaustive":
            # Cheap precheck so an empty front below can be attributed correctly:
            # no valid (activity, resource) pair at all vs. valid pairs that all
            # failed the confidence filter.
            if not _build_valid_pairs(poss, act_with_res):
                _fill_empty(cid, "no_possible_actions")
                continue

            baseline_row = (
                baseline_instances_by_case.get(cid) if baseline_instances_by_case is not None else None
            )
            pareto_front = exhaustive_pareto_search(
                query_instance,
                poss,
                predictive_outcome_model,
                predictive_time_model,
                act_with_res,
                baseline_row=baseline_row,
                gamma_cls=gamma_cls,
                gamma_reg=gamma_reg,
            )
            if not pareto_front:
                status = "no_confident_recommendation" if baseline_row is not None else "no_possible_actions"
                _fill_empty(cid, status)
                continue
        else:
            raise ValueError("Unknown method for recommendations: %s" % method)

        if method == "exhaustive" and baseline_row is None:
            # Confidence filter was skipped entirely for this case (no
            # baseline available, e.g. a trace of length 1 -- see
            # build_baseline_instances): the recommendation is still valid,
            # but distinguish it from a genuinely confidence-filtered "ok"
            # so this isn't silently read as "the model was confident".
            status_by_case[cid] = "ok_no_baseline"
        else:
            status_by_case[cid] = "ok"
        top_k_pairs = select_top_k_pareto_actions(pareto_front, k=k)
        # (act, res) -> (outcome, time, uncertainty, prob_outcome_better, prob_time_better),
        # for the diagnostic file. NSGA2's 5-tuples have no confidence probabilities.
        if method == "exhaustive":
            obj_by_pair = {
                (t[0], t[1]): (float(t[2]), float(t[3]), float(t[4]), float(t[5]), float(t[6]))
                for t in pareto_front
            }
        else:
            obj_by_pair = {
                (t[0], t[1]): (float(t[2]), float(t[3]), float(t[4]), None, None)
                for t in pareto_front
            }
        for j in range(k):
            if j < len(top_k_pairs):
                pair = top_k_pairs[j]
                rec_list[j][cid] = pair
                obj_list[j][cid] = obj_by_pair.get(pair, (None, None, None, None, None))
            else:
                rec_list[j][cid] = (None, None)
                obj_list[j][cid] = (None, None, None, None, None)

    return rec_list, obj_list, status_by_case