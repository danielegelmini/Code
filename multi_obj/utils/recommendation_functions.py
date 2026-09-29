import tqdm
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
from scipy.stats import t as student_t
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

from utils.pre_processing_functions import convert_dtypes_bpi12, NO_NEXT_TOKEN

case_id_name = "case:concept:name"
activity_column_name = "concept:name"
end_date_name = "time:timestamp"
start_date_name = "start:timestamp"
resource_column_name = "org:resource"
outcome_name = "outcome"

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


def predict_time_members_var(predictive_time_model, rows_df):
    """
    Return each virtual-ensemble member's own (aleatoric) variance for
    rows_df, as a 2-D numpy array of shape (n_rows, ve_count) -- see
    UncertaintyRegressor.predict_members_var(). Paired with
    predict_time_members()'s per-member means so
    _paired_gaussian_prob_greater() can compare on the FULL predictive
    variance (aleatoric + epistemic) instead of only the across-member
    (epistemic) spread. These are RAW variances: the sigma-scaling
    calibration factor is applied afterwards (see time_sigma_scale).

    Falls back to zeros (no aleatoric variance to report) for an older time
    model without uncertainty support; predict_time_members(), always
    called alongside this one, already prints the one-time warning for
    that case.
    """
    predictor = predictive_time_model
    predictor_input = rows_df
    if hasattr(predictive_time_model, "named_steps"):
        steps = predictive_time_model.named_steps
        predictor = steps.get("prediction", predictive_time_model)
        if "transformation" in steps:
            predictor_input = steps["transformation"].transform(rows_df)

    if hasattr(predictor, "predict_members_var"):
        return np.asarray(predictor.predict_members_var(predictor_input), dtype=float)

    return np.zeros((len(rows_df), 1))


def time_sigma_scale(predictive_time_model):
    """The time model's post-hoc sigma-scaling calibration factor s
    (UncertaintyRegressor.sigma_scale), or 1.0 for a model without one."""
    predictor = predictive_time_model
    if hasattr(predictive_time_model, "named_steps"):
        predictor = predictive_time_model.named_steps.get("prediction", predictive_time_model)
    return float(getattr(predictor, "sigma_scale", 1.0))


def predict_outcome_members_logit(predictive_outcome_model, rows_df):
    """
    Return the per-virtual-ensemble-member CALIBRATED logit (pre-sigmoid) for
    rows_df, as a 2-D numpy array of shape (n_rows, ve_count) -- see
    UncertaintyClassifier.predict_members_logit(). Used by
    _compute_confidence_probabilities(), which models the baseline/candidate
    comparison as two correlated Gaussians: a probability is bounded to
    [0, 1] and a poor fit for that, while the logit is unbounded.

    Falls back to the raw predict_proba-derived logit (a "1-member
    ensemble") for an older outcome model without uncertainty support,
    reusing the same one-time warning as predict_outcome_proba.
    """
    global _CALIBRATION_WARNED

    predictor = predictive_outcome_model
    predictor_input = rows_df
    if hasattr(predictive_outcome_model, "named_steps"):
        steps = predictive_outcome_model.named_steps
        predictor = steps.get("prediction", predictive_outcome_model)
        if "transformation" in steps:
            predictor_input = steps["transformation"].transform(rows_df)

    if hasattr(predictor, "predict_members_logit"):
        return np.asarray(predictor.predict_members_logit(predictor_input), dtype=float)

    if not _CALIBRATION_WARNED:
        print(
            "[recommendation_functions] outcome model exposes no temperature "
            "calibration; the confidence-as-KPI filter degrades to a "
            "single-member ensemble (raw predict_proba only). Retrain the "
            "outcome model with posterior_sampling=True to calibrate it."
        )
        _CALIBRATION_WARNED = True
    proba = np.asarray(predictive_outcome_model.predict_proba(rows_df), dtype=float)[:, 1]
    p = np.clip(proba, 1e-7, 1.0 - 1e-7)
    logit = np.log(p / (1.0 - p))
    return logit.reshape(-1, 1)


def _paired_gaussian_prob_greater(a_members, b_members, a_var=None, b_var=None, std_scale=1.0):
    """
    Estimate P(A > B) for two paired ensembles of M values each (A: one or
    more cases, B: a single baseline), modelling A and B as correlated
    Gaussians whose mean, variance and covariance are ESTIMATED -- not known
    -- from those same M virtual-ensemble members.

    Why paired, not independent: a_members and b_members come from the same
    fitted model's virtual ensemble, with the same member index m giving both
    A's and B's value, so Cov(A, B) can be estimated directly from the M
    pairs instead of assumed to be zero. Treating them as independent would
    drop a real, positive correlation (both share whatever quirks member m
    has) and bias the comparison.

    Why Student's t, not the normal (z) CDF: mean_diff and var_diff are
    themselves estimates from only M samples, not the true population
    values. Plugging point estimates into the normal CDF understates how
    uncertain the comparison really is, especially for small M -- exactly
    the situation Student's t-distribution was built to correct for. This
    uses the standard posterior-predictive result for a NEW draw from a
    normal population with unknown mean/variance (e.g. Gelman et al., BDA3
    ch. 3): given M observed pairs, a new difference D_new = A_new - B_new
    follows a location-scale Student-t with M-1 degrees of freedom,
        D_new ~ t_(M-1)( mean_diff, var_diff * (1 + 1/M) ),
    so
        P(A > B) = P(D_new > 0) = T_(M-1)( mean_diff / (std_diff * sqrt(1 + 1/M)) ),
    T_(M-1) being the standard Student-t CDF. Both the (1 + 1/M) inflation
    and t's fatter-than-normal tails shrink as M grows, so this converges to
    the plain z-test as M -> infinity -- a strict generalisation of it, not a
    different method.

    Args:
        a_members (numpy.ndarray): Shape (n, M) -- one row per case to score.
        b_members (numpy.ndarray): Shape (M,) -- the single baseline, paired
            with a_members on the member axis.
        a_var, b_var (numpy.ndarray, optional): Same shapes as a_members/
            b_members -- each member's own (aleatoric) variance, e.g. from
            UncertaintyRegressor.predict_members_var(). When given, their
            per-case mean is added to var_a/var_b so the comparison uses the
            FULL predictive variance (aleatoric + epistemic, law of total
            variance) instead of only the across-member (epistemic) spread
            that a_members/b_members alone give. The covariance term is left
            untouched: aleatoric noise on the baseline row and on the
            candidate row is independent even when it comes from the same
            member, so it inflates each side's own variance but not their
            covariance. Omit both (the default) to use the epistemic-only
            comparison.
        std_scale (float, optional): Post-hoc calibration factor applied to the
            standard deviation of the difference (the whole of it, epistemic and
            aleatoric), e.g. UncertaintyRegressor.sigma_scale -- the same factor
            the sigma-scaling calibration applies to the total predictive std.
            Defaults to 1.0 (no recalibration).

    Returns:
        numpy.ndarray: Shape (n,), each in [0, 1]. Degrades to a step
        function (0, 0.5 or 1, by the sign of mean_diff) when M < 2 -- too
        few members to estimate a variance at all.
    """
    M = b_members.shape[-1]
    mean_b = b_members.mean() #baseline
    mean_a = a_members.mean(axis=1) #method
    mean_diff = mean_a - mean_b

    if M < 2:
        return np.where(mean_diff > 0, 1.0, np.where(mean_diff < 0, 0.0, 0.5))

    var_b = b_members.var(ddof=1)
    var_a = a_members.var(axis=1, ddof=1)
    if b_var is not None:
        var_b = var_b + b_var.mean()
    if a_var is not None:
        var_a = var_a + a_var.mean(axis=1)
    cov_ab = ((a_members - mean_a[:, None]) * (b_members - mean_b)[None, :]).sum(axis=1) / (M - 1)
    var_diff = np.clip(var_a + var_b - 2.0 * cov_ab, 0.0, None) * std_scale ** 2
    std_diff = np.sqrt(var_diff * (1.0 + 1.0 / M))

    degenerate = std_diff == 0
    z = np.divide(mean_diff, std_diff, out=np.zeros_like(mean_diff), where=~degenerate)
    prob = student_t.cdf(z, df=M - 1)
    #false condition: mean > 0 -> 1; mean < 0 -> 0; mean == 0 -> 0.5
    return np.where(degenerate, np.where(mean_diff > 0, 1.0, np.where(mean_diff < 0, 0.0, 0.5)), prob)

# ---------------------------------------------------------------------------
# Utils for run_experiment.py
# ---------------------------------------------------------------------------
def act_with_res_func(df, activity_column_name, resource_column_name):
    """
    Generates a dictionary mapping each unique activity to a list of its associated unique resources.
    This mapping excludes 'missing' resources; 'NotDef' is kept as a valid
    choice, like any other value of the original dataset.

    Args:
        df (pandas.DataFrame): The DataFrame containing the event log data.
        activity_column_name (str): The name of the column containing activity names.
        resource_column_name (str): The name of the column containing resource names.

    Returns:
        dict: A dictionary in the format {activity: [unique resources]}.
    """
    grouped = df.groupby(activity_column_name)[resource_column_name].unique()
    forbidden = {"missing"}
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

def build_no_recommendation_baseline_instances(query_instances_by_case):
    """
    Creates the confidence-as-KPI "no recommendation" baseline instance for each case:
    the SAME query instance the candidates are evaluated on (same prefix e_1,...,e_k, same
    features), with NEXT_ACTIVITY and NEXT_RESOURCE both set to NO_NEXT_TOKEN.

    This relies on the models having been trained with the no-recommendation copy of the
    training set (see pre_processing_functions.add_no_recommendation_copy, applied in
    2_training_predictive_model.py), so NO_NEXT_TOKEN is a value they have learned: the
    expected outcome / remaining time from this state when no next step is specified.
    Unlike build_baseline_instances, the baseline is evaluated at the same state as the
    candidates and exists for every case, including traces with a single event.

    Args:
        query_instances_by_case (dict): build_query_instances() output.

    Returns:
        dict: {case_id: {"feature_name": value, ...}}, one entry per case.
    """
    return {
        cid: {**instance, "NEXT_ACTIVITY": NO_NEXT_TOKEN, "NEXT_RESOURCE": NO_NEXT_TOKEN}
        for cid, instance in query_instances_by_case.items()
    }


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

    The baseline is the model's prediction on baseline_row, evaluated completely
    as-is, with no pair substitution. With build_no_recommendation_baseline_instances
    it is the query instance itself with NEXT_ACTIVITY/NEXT_RESOURCE set to
    NO_NEXT_TOKEN, i.e. the same state as the candidates with no next step given.

    Both the baseline and every candidate are scored member-by-member on the
    SAME fitted models' virtual-ensemble members (predict_outcome_members_logit /
    predict_time_members): each is modelled as a Gaussian whose mean,
    variance and cross-covariance with the baseline are estimated from those
    M paired members (see _paired_gaussian_prob_greater), so the correlation
    between baseline and candidate -- both come from the same fitted model --
    is captured automatically instead of assumed away. The outcome
    comparison runs on the LOGIT scale (predict_outcome_members_logit, not
    the raw [0, 1] probability) because a Gaussian fits an unbounded
    quantity, not one clipped to [0, 1] -- the standard treatment for a
    Bernoulli parameter's uncertainty. Because that mean/variance/covariance
    are themselves estimated from only M members, the comparison uses
    Student's t-distribution (M-1 degrees of freedom) rather than the normal
    -- see _paired_gaussian_prob_greater's docstring for the derivation.

    Both comparisons run on CALIBRATED quantities:
      * time: the full predictive variance -- across-member (epistemic) spread
        plus each member's own (aleatoric) variance (predict_time_members_var)
        -- with the whole std of the difference multiplied by the regressor's
        sigma-scaling factor s (time_sigma_scale), i.e. the same calibrated
        total std that predict_uncertainty() reports;
      * outcome: the temperature-calibrated member logits
        (predict_outcome_members_logit, logit / T). Only the across-member
        spread is used: a classification member's own variance is the Bernoulli
        noise of the 0/1 result, not uncertainty about the probability. Note
        that dividing every member logit by the same T scales the mean
        difference and its std alike, so p_out is invariant to T.

    Args:
        candidate_pairs: list of (activity, resource) tuples to score.
        baseline_row: this case's baseline instance
            (build_no_recommendation_baseline_instances output for one case) -- a
            dict-like row with NEXT_ACTIVITY/NEXT_RESOURCE already set.
        query_instance: the prefix/query instance for this case (e_1,...,e_k),
            against which candidate_pairs are evaluated as e_{k+1}.
        predictive_outcome_model, predictive_time_model: fitted pipelines.

    Returns:
        (prob_outcome_better, prob_time_better): two 1-D numpy arrays,
        aligned with candidate_pairs. prob_outcome_better[i] is P(candidate
        i's calibrated outcome logit > baseline's), equivalently P(candidate
        i's calibrated outcome probability > baseline's), since the logit is
        a monotonic transform of the probability; prob_time_better[i] is
        P(candidate i's remaining time < baseline's) under the calibrated
        predictive distribution.
    """
    baseline_outcome_row = align_query_instance_with_model(baseline_row, predictive_outcome_model)
    baseline_time_row = align_query_instance_with_model(baseline_row, predictive_time_model)
    candidate_outcome_rows, candidate_time_rows = _build_candidate_rows(
        candidate_pairs, query_instance, predictive_outcome_model, predictive_time_model
    )
    outcome_rows_df = pd.concat([baseline_outcome_row, candidate_outcome_rows], ignore_index=True)
    time_rows_df = pd.concat([baseline_time_row, candidate_time_rows], ignore_index=True)

    outcome_members = predict_outcome_members_logit(predictive_outcome_model, outcome_rows_df)
    time_members = predict_time_members(predictive_time_model, time_rows_df)
    time_members_var = predict_time_members_var(predictive_time_model, time_rows_df)

    baseline_outcome_members, candidate_outcome_members = outcome_members[0], outcome_members[1:]
    baseline_time_members, candidate_time_members = time_members[0], time_members[1:]
    baseline_time_var, candidate_time_var = time_members_var[0], time_members_var[1:]

    prob_outcome_better = _paired_gaussian_prob_greater(candidate_outcome_members, baseline_outcome_members)
    prob_time_better = 1.0 - _paired_gaussian_prob_greater(
        candidate_time_members, baseline_time_members,
        a_var=candidate_time_var, b_var=baseline_time_var,
        std_scale=time_sigma_scale(predictive_time_model),
    )

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
    which is untouched by this change).

    Args:
        query_instance (pandas.DataFrame, pandas.Series, or dict): The current state features of the case.
        possible_actions (list of str): Allowed next activities based on the transition graph.
        predictive_outcome_model (estimator): The predictive model for the primary outcome.
        predictive_time_model (estimator): The predictive model for total/remaining time.
        act_with_res (dict of str to list of str): Mapping of valid resources for each activity.
        baseline_row (dict-like, optional): this case's confidence-filter baseline instance
            (see build_no_recommendation_baseline_instances) -- the query instance with
            NEXT_ACTIVITY/NEXT_RESOURCE set to NO_NEXT_TOKEN, used as-is. None disables
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
        as "100% confident".
    """
    # creation of possible valid pairs from all the activity of possible actions using the transition system 
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
    confidence against a "no recommendation" baseline -- the same query instance
    with NEXT_ACTIVITY/NEXT_RESOURCE set to NO_NEXT_TOKEN, see
    build_no_recommendation_baseline_instances -- before the Pareto front is built (see
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
        baseline_instances_by_case (dict, optional): build_no_recommendation_baseline_instances() output.
            Only used for method="exhaustive"; None (or a missing case) disables the confidence
            filter for that case (every valid pair is kept, as before this change).
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
            method="exhaustive" cases with no baseline passed. Missing
            entries are (None, None) / (None, None, None, None, None).
          - status_by_case is {case_id: status}, one entry per case (not per
            rank j), with status one of: "ok" (a recommendation was
            produced); "no_possible_actions" (no legal next activity, or no
            valid (activity, resource) pair for it); or
            "no_confident_recommendation" (valid pairs existed but none
            passed the confidence filter -- only possible for
            method="exhaustive" with a baseline available for that case).
    """
    # setup
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
        trace_history = trace_df[activity_column_name].tolist() #list of activity for case_id (cid) selected

        query_instance = _to_row_df(query_instances_by_case[cid])
        # all possible activities given the trace history and transition graph
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