"""
Step 2 of 0_estimate_start_times_vinci.py: the start-time estimator of Vinci et al. (AI in Medicine 2026,
Appendix A), ported from https://github.com/Franc3sc4/StartTimestampEstimator (run.py + src/).

    start(e) = end(e) - alpha(act(e)) * (end(e) - mintime(e))                          (Eq. A.2)

One alpha in [0, 1] per activity: alpha = 1 -> the activity starts as soon as possible, alpha = 0 -> it
is instantaneous. The alphas are searched by bisection (estimate_alphas): at every iteration the log is
repaired with the current alphas, the activity durations are refitted and given to the Prosimos
simulation model, a log is simulated, and the simulated log is compared with the real one, activity by
activity, on the times between consecutive completions of a case (Eq. A.3).

`log` is always a DataFrame with one row per event and the columns CASE, ACT, RES, END (below), sorted
per case by completion time.
"""
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from prosimos.simulation_engine import run_simulation
from scipy import stats

# Column names of the XES logs
CASE = "case:concept:name"
ACT = "concept:name"
RES = "org:resource"
START = "start:timestamp"
END = "time:timestamp"

CANDIDATE_DISTRIBUTIONS = ["fix", "norm", "expon", "uniform", "lognorm", "gamma"]


# =====================================================================================================
# The algorithm (repository: run.py, run_framework)
# =====================================================================================================
def estimate_alphas(log, bpmn_path, simulation_parameters, work_dir, n_iterations, perc_head_tail, seed):
    """Returns ({activity: alpha}, history DataFrame with the alpha and the error of every activity at every
    iteration).

    Algorithm 1 of the paper: iteration 0 tries alpha = 0 for every activity, iteration 1 alpha = 1; every
    later iteration tries the midpoint of each activity's interval [left, right], and then the endpoint with
    the larger error becomes the midpoint, so the interval always halves towards the endpoint with the
    smaller error. All n_iterations are run (run.py instead moves an endpoint only if the midpoint beats it,
    so a midpoint worse than both endpoints, also by simulation noise, freezes the interval).
    The returned alpha of an activity is, among all the alphas tried for it, the one with the smallest error
    (as run.py; the paper returns the last midpoint, which coincides with it when the error is convex in
    alpha and the simulation noise is small)."""
    activities = sorted(log[ACT].unique())
    task_activity = task_activities(bpmn_path)
    n_cases = log[CASE].nunique()
    n_simulated = n_cases + int(n_cases * perc_head_tail)  # extra cases, discarded at the head and the tail
    starting_at = log[END].min().isoformat()                # the simulation starts when the log starts
    rng = np.random.default_rng(seed)

    # Fixed for the whole search: the real inter-event times (the target) and mintime (Def. 5)
    real_times = inter_event_times(log[CASE], log[ACT], log[END])
    mintime = minimum_start_times(log)

    # Search state: for each activity the interval [left, right] and the error at its two endpoints
    interval = {"left": {a: 0.0 for a in activities}, "right": {a: 1.0 for a in activities}}
    interval_error = {}
    tried, history = [], []  # tried: (alphas, errors) of every iteration
    for i in range(n_iterations):
        # 1. alphas of this iteration: 0, then 1, then the midpoints
        alphas = alphas_to_try(i, interval)

        # 2. evaluate them: repair the log, refit the durations, simulate, compare with the real log
        starts = start_times_from_alphas(log, mintime, alphas)
        durations = fit_duration_distributions(log, starts, rng)
        simulated = simulate(bpmn_path, simulation_parameters, task_activity, durations,
                             n_simulated, starting_at, work_dir, seed)
        simulated = discard_warm_up_and_tail(simulated, n_cases, perc_head_tail)
        errors = activity_errors(real_times, simulated, activities)

        tried.append((alphas, errors))
        history.append({"iteration": i, **{f"alpha::{a}": alphas[a] for a in activities},
                        **{f"error_hours::{a}": errors[a] / 3600 for a in activities}})
        print(f"    iteration {i}: mean error {np.mean(list(errors.values())) / 3600:.2f} h")

        # 3. update the intervals: iterations 0 and 1 give the errors of the endpoints, later ones halve them
        if i == 0:
            interval_error["left"] = dict(errors)
        elif i == 1:
            interval_error["right"] = dict(errors)
        else:
            shrink_intervals(interval, interval_error, alphas, errors)

    # best alpha of each activity: the one with the smallest error among those tried
    best = {a: min(tried, key=lambda t: t[1][a])[0][a] for a in activities}
    return best, pd.DataFrame(history)


def alphas_to_try(iteration, interval):
    """Iteration 0: the left endpoints (all 0); iteration 1: the right endpoints (all 1); then the midpoints."""
    if iteration == 0:
        return dict(interval["left"])
    if iteration == 1:
        return dict(interval["right"])
    # return the alphas, no changes to the intervals
    return {a: (interval["left"][a] + interval["right"][a]) / 2 for a in interval["left"]}


def shrink_intervals(interval, interval_error, alphas, errors):
    """Algorithm 1 of the paper: for each activity, the endpoint with the larger error becomes the midpoint
    just evaluated (with the midpoint's error), whatever that error is: the interval halves towards the
    endpoint with the smaller error. Ties keep the left endpoint (argmin returns the first).
    E.g. [0, 1] with errors 20 h and 5 h, midpoint 0.5 -> the left endpoint becomes 0.5: [0.5, 1]."""
    for a, e_mid in errors.items():
        worse = "right" if interval_error["left"][a] <= interval_error["right"][a] else "left"
        interval[worse][a], interval_error[worse][a] = alphas[a], e_mid


# =====================================================================================================
# Start times (repository: src/utils.py, set_start_timestamp_from_alpha)
# =====================================================================================================
def minimum_start_times(log):
    """Def. 5: mintime(e) = max(end of the previous event of the case, end of the latest event of the same
    resource that ended before e). NaT for the first event of a case, which stays instantaneous (as in the
    repository: case arrivals do not change).
    The repository takes the resource's previous event from an unsorted list that also contains e itself,
    so it is often e (mintime = end, alpha has no effect): here it is the latest one that ended strictly
    before e."""
    # end of the previous event of the same case (NaT for the first event)
    prev_case = log.groupby(CASE)[END].shift(1)

    # end of the latest event of the same resource that ended before e (NaT if there is none)
    prev_resource = pd.Series(pd.NaT, index=log.index, dtype=log[END].dtype)
    for _, events in log.groupby(RES):
        ends = np.sort(events[END].values)                                 # all completions of the resource
        pos = np.searchsorted(ends, events[END].values, side="left") - 1   # index of the latest end < own end
        found = pos >= 0
        prev_resource.loc[events.index[found]] = pd.to_datetime(ends[pos[found]], utc=True)

    mintime = pd.concat([prev_case, prev_resource], axis=1).max(axis=1)
    return mintime.where(prev_case.notna())  # first event of a case: NaT -> instantaneous


def start_times_from_alphas(log, mintime, alphas):
    """Eq. A.2: start = end - alpha * (end - mintime); the first event of a case (mintime NaT) gets start = end."""
    gap = (log[END] - mintime).fillna(pd.Timedelta(0))
    return log[END] - gap * log[ACT].map(alphas).astype(float)


# =====================================================================================================
# Durations (repository: src/temporal_utils.py and src/distribution_utils.py)
# =====================================================================================================
def fit_duration_distributions(log, starts, rng=None):
    """{activity: (distribution name, [params])} of the durations end - start in seconds (elapsed time,
    as in the repository)."""
    durations = (log[END] - starts).dt.total_seconds()
    return {a: find_best_fit_distribution(durations[log[ACT] == a].values, rng) for a in log[ACT].unique()}


def find_best_fit_distribution(values, rng=None):
    """Fits every candidate distribution and keeps the one whose sample is closest to `values` (Wasserstein
    distance). Returns (name, [params]) with the params in the order Prosimos expects (repository:
    find_best_fit_distribution). Also used for the inter-arrival times in simulation_model.py."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or values.min() == values.max():  # nothing to fit: a constant
        return "fix", [float(values.min()) if len(values) else 0.0]

    # Parameters written into the JSON, as in the repository: Prosimos takes moments + bounds, not the
    # scipy parameters (e.g. lognorm = mean, variance, min, max)
    mean, std, vmin, vmax = values.mean(), values.std(), values.min(), values.max()
    params = {"fix": [mean], "norm": [mean, std, 0, vmax], "expon": [mean, 0, vmax], "uniform": [vmin, vmax],
              "lognorm": [mean, std ** 2, vmin, vmax], "gamma": [mean, std ** 2, vmin, vmax]}

    # Score of each candidate: Wasserstein distance between the values and a sample of the fitted scipy
    # distribution of the same size
    distance = {"fix": stats.wasserstein_distance(values, np.full(len(values), mean))}
    for name in CANDIDATE_DISTRIBUTIONS[1:]:
        dist = getattr(stats, name)
        try:
            sample = dist.rvs(*dist.fit(values), size=len(values), random_state=rng)
        except Exception:  # fit failed: skip the candidate
            continue
        distance[name] = stats.wasserstein_distance(values, sample)
    best = min(distance, key=distance.get)
    return best, [float(p) for p in params[best]]


# =====================================================================================================
# Simulation (repository: src/simulation_utils.py)
# =====================================================================================================
def task_activities(bpmn_path):
    """{task id: activity} of the BPMN (the fitness-1 net has several tasks per activity)."""
    return {e.get("id"): e.get("name") for e in ET.parse(bpmn_path).getroot().iter() if e.tag.endswith("}task")}


def simulate(bpmn_path, simulation_parameters, task_activity, durations, n_cases, starting_at, work_dir, seed):
    """Writes the durations into the simulation parameters (every resource of a task gets the activity's
    distribution, as update_sim_params) and simulates n_cases cases with Prosimos from `starting_at`.
    Returns the simulated log (columns case_id, activity, enable_time, start_time, end_time, resource)."""
    # durations -> JSON: every task of an activity, for every resource that can execute it
    for task in simulation_parameters["task_resource_distribution"]:
        name, params = durations.get(task_activity[task["task_id"]], ("fix", [0.0]))
        for r in task["resources"]:
            r["distribution_name"] = name
            r["distribution_params"] = [{"value": p} for p in params]
    json_path, log_path = Path(work_dir) / "iteration_parameters.json", Path(work_dir) / "simulated_log.csv"
    json_path.write_text(json.dumps(simulation_parameters))

    # Prosimos simulation (same seed at every iteration: the alphas, not the random numbers, change)
    random.seed(seed)
    np.random.seed(seed)
    run_simulation(str(bpmn_path), str(json_path), n_cases, log_out_path=str(log_path), starting_at=starting_at)
    simulated = pd.read_csv(log_path)
    simulated["end_time"] = pd.to_datetime(simulated["end_time"], utc=True, format="mixed")
    return simulated


def discard_warm_up_and_tail(simulated, n_cases, perc_head_tail):
    """Keeps n_cases simulated cases, dropping perc_head_tail/2 of them at the head (the system starts empty)
    and the rest at the tail (the last cases have no later arrivals) -- run.py: min_case_id, max_case_id."""
    first = int((n_cases + int(n_cases * perc_head_tail)) * perc_head_tail / 2)
    return simulated[(simulated["case_id"] > first) & (simulated["case_id"] <= first + n_cases)]


# =====================================================================================================
# Error (repository: src/metric_utils.py, compute_wass_err)
# =====================================================================================================
def inter_event_times(case_ids, activities, ends):
    """Def. 3: 0 for the first event of a case, otherwise end(e_i) - end(e_(i-1)), in seconds.
    Returns a DataFrame (case, act, end, delta) with one row per event."""
    d = pd.DataFrame({"case": np.asarray(case_ids), "act": np.asarray(activities), "end": np.asarray(ends)})
    d = d.sort_values(["case", "end"], kind="stable")
    d["delta"] = d.groupby("case")["end"].diff().dt.total_seconds().fillna(0.0)
    return d


def activity_errors(real_times, simulated, activities):
    """Eq. A.3: {activity: Wasserstein distance (seconds) between its real and simulated inter-event times}.
    An activity never simulated counts as all zeros."""
    sim_times = inter_event_times(simulated["case_id"], simulated["activity"], simulated["end_time"])
    errors = {}
    for a in activities:
        real = real_times.loc[real_times["act"] == a, "delta"].values
        sim = sim_times.loc[sim_times["act"] == a, "delta"].values
        errors[a] = stats.wasserstein_distance(real, sim if len(sim) else np.zeros(len(real)))
    return errors
