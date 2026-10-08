#!/usr/bin/env python3
"""
Throwaway trial: how much would the activity start times change if, instead of the
Chapela-Campa & Dumas estimator used by 0_estimate_start_times.py (pix-framework), we used
the simulation-based estimator of

  * Fracca, de Leoni, Asnicar, Turco, "Estimating Activity Start Timestamps in the presence
    of Waiting Times via Process Simulation", CAiSE 2022                    (paper 46)
  * Vinci et al., "Improving organizational processes in healthcare through
    simulation-driven resource allocation", AI in Medicine 2026, Appendix   (paper 45)
    -- implementation: https://github.com/Franc3sc4/StartTimestampEstimator (run.py)

!! Run it from the dedicated conda env `prosimos_env` (a clone of pix_env + simod 5.2.1 +
!! prosimos 2.1.0), with the user site-packages disabled:
!!     C:\\Users\\Utente\\.conda\\envs\\prosimos_env\\python.exe -s 0b_vinci_start_times_trial.py --case_study BPI12

Method (paper 45, Appendix A.2; run.py of the repo)
---------------------------------------------------
    start(e) = end(e) - alpha(act(e)) * (end(e) - mintime(e))
    mintime(e) = max(end of the previous event of the case,
                     end of the previous event of the same resource)          (Def. 5)

alpha(a) in [0, 1] is one value per activity: alpha = 1 starts every activity at the earliest
possible moment (the "no waiting time" assumption -- which is essentially what pix does:
start = max(enabled, available)), alpha = 0 makes it instantaneous. alpha is searched by
bisection: the log is repaired with the current alphas, the activity-duration distributions are
refitted from it and plugged into a Prosimos simulation model, a log is simulated, and for each
activity the Wasserstein distance between the real and simulated inter-completion times (time
between the completion of an event and that of the previous event of the case) is measured;
the interval [l(a), r(a)] is shrunk towards the alpha with the smaller distance.

The Prosimos model (BPMN + JSON: resources, calendars, arrivals, gateway probabilities) is
discovered once with Simod -- the authors took ready-made Prosimos models; here none exists, so
Simod (same group as Prosimos) builds it from the current pix-repaired log. Its activity
durations are then overwritten at every iteration, as in the repo (update_sim_params).

Deviations from the repo, kept on purpose
-----------------------------------------
  * mintime uses the previous event of the resource that ended STRICTLY before e (Def. 5).
    utils.set_start_timestamp_from_alpha in the repo also picks e itself (its own completion
    is in the resource list, `<= 0`) and takes the last element of an unsorted list, so for
    many events the resource term collapses to end(e) and alpha has no effect.
  * Same domain configuration as 0_estimate_start_times.py: A_/O_ activities and bot resources
    are instantaneous (alpha fixed to 0); for the missing resource "0" of BPI12 the resource
    term is skipped. In the simulation model "0" is split into pseudo-resources ("lanes") with
    non-overlapping executions, otherwise Prosimos would treat it as ONE person doing 6k
    activities in sequence.
  * The first event of a case is instantaneous (repo behaviour; also ours: arrivals unchanged).

Output (multi_obj/0b_vinci_trial/<case_study>/): simod model, per-iteration errors and alphas,
comparison table vs the pix estimate.
"""
import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
from scipy import stats

warnings.filterwarnings("ignore")

CASE, ACT, RES, START, END = "case:concept:name", "concept:name", "org:resource", "start:timestamp", "time:timestamp"

BOT_RESOURCES = {"BPI12": {"112"}, "bpi17_before": {"User_1"}, "bpi17_after": {"User_1"}}
MISSING_RESOURCE = {"BPI12": "0"}
INSTANT_PREFIXES = ("A_", "O_")
LANE_PREFIX = "missing_lane_"

BASE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_study", default="BPI12")
    p.add_argument("--n_iterations", type=int, default=12, help="Max bisection iterations (repo default: 20).")
    p.add_argument("--perc_head_tail", type=float, default=0.1, help="Extra simulated cases, trimmed half at the head and half at the tail (repo: 0.1).")
    p.add_argument("--simod_iterations", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rediscover", action="store_true", help="Re-run Simod even if the model exists.")
    p.add_argument("--working_time", action="store_true",
                   help="Fit the durations in working time of the resource calendar (Prosimos consumes durations "
                        "only inside the calendar). Default: elapsed end - start, as the repo does.")
    return p.parse_args()


# --------------------------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------------------------
def read_log(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(pm4py.read_xes(str(path), show_progress_bar=False)).reset_index(drop=True)
    df = df[[CASE, ACT, RES, START, END]].copy()
    df[RES] = df[RES].astype(str)
    df[START] = pd.to_datetime(df[START], utc=True)
    df[END] = pd.to_datetime(df[END], utc=True)
    # same per-case order as 0_estimate_start_times.py (completion time, file order breaks ties)
    return df.sort_values([CASE, END], kind="stable")


def missing_resource_lanes(df: pd.DataFrame, missing: str) -> pd.Series:
    """Greedy interval partitioning of the missing-resource events into the fewest pseudo-resources
    whose executions never overlap. Intervals = [end of the previous event of the case, end], i.e.
    the widest the estimator can make them (alpha = 1), so they stay disjoint for every alpha."""
    res = df[RES].copy()
    mask = df[RES] == missing
    if not mask.any():
        return res
    prev_end = df.groupby(CASE)[END].shift(1).fillna(df[END])
    order = df.index[mask][np.argsort(prev_end[mask].values, kind="stable")]
    lane_free = []  # end time of the last interval of each lane
    for idx in order:
        s, e = prev_end[idx], df.at[idx, END]
        for k, free in enumerate(lane_free):
            if free <= s:
                lane_free[k] = e
                res.at[idx] = f"{LANE_PREFIX}{k}"
                break
        else:
            lane_free.append(e)
            res.at[idx] = f"{LANE_PREFIX}{len(lane_free) - 1}"
    print(f"  missing resource '{missing}': {mask.sum()} events split into {len(lane_free)} lanes")
    return res


def compute_mintime(df: pd.DataFrame, cs: str) -> pd.Series:
    """Def. 5 (paper 45): max(end of the previous event of the case, end of the previous event of
    the resource that ended strictly before e). NaT for the first event of a case."""
    prev_case = df.groupby(CASE)[END].shift(1)
    prev_res = pd.Series(pd.NaT, index=df.index, dtype=df[END].dtype)
    skip = {MISSING_RESOURCE.get(cs)} | {r for r in df[RES].unique() if r.startswith(LANE_PREFIX)}
    for r, g in df.groupby(RES):
        if r in skip:
            continue
        ends = np.sort(g[END].values)
        pos = np.searchsorted(ends, g[END].values, side="left") - 1
        vals = np.where(pos >= 0, ends[np.clip(pos, 0, None)], np.datetime64("NaT"))
        prev_res.loc[g.index] = pd.to_datetime(vals, utc=True)
    mintime = pd.concat([prev_case, prev_res.where(prev_case.notna())], axis=1).max(axis=1)
    return mintime.where(prev_case.notna())


def fixed_instant_mask(df: pd.DataFrame, cs: str) -> pd.Series:
    return df[ACT].str.startswith(INSTANT_PREFIXES) | df[RES].isin(BOT_RESOURCES.get(cs, set()))


def apply_alphas(df: pd.DataFrame, mintime: pd.Series, instant: pd.Series, alphas: dict) -> pd.Series:
    """Eq. (A.2): start = alpha * mintime + (1 - alpha) * end."""
    a = df[ACT].map(alphas).astype(float).where(~instant, 0.0)
    gap = (df[END] - mintime).fillna(pd.Timedelta(0))
    return df[END] - gap * a


# --------------------------------------------------------------------------------------------
# Simod: discover the Prosimos model once
# --------------------------------------------------------------------------------------------
def discover_prosimos_model(df_pix: pd.DataFrame, out_dir: Path, cs: str, n_iter: int, rediscover: bool):
    model_dir = out_dir / "simod_model"
    bpmn, js = model_dir / f"{cs}.bpmn", model_dir / f"{cs}.json"
    if bpmn.exists() and js.exists() and not rediscover:
        print(f"  using cached Prosimos model in {model_dir}")
        return bpmn, js

    from pix_framework.io.event_log import EventLogIDs
    from simod.event_log.event_log import EventLog
    from simod.settings.simod_settings import SimodSettings
    from simod.simod import Simod

    csv_path = out_dir / f"{cs}.csv"
    out = df_pix.rename(columns={CASE: "case_id", ACT: "activity", RES: "resource", START: "start_time", END: "end_time"})
    out.to_csv(csv_path, index=False)
    ids = {"case": "case_id", "activity": "activity", "resource": "resource", "start_time": "start_time", "end_time": "end_time"}
    config = {
        "version": 5.2,
        "common": {"train_log_path": str(csv_path), "log_ids": ids, "perform_final_evaluation": False,
                   "num_final_evaluations": 0, "discover_data_attributes": False, "clean_intermediate_files": True},
        "preprocessing": {"multitasking": False},
        "control_flow": {"num_iterations": n_iter, "num_evaluations_per_iteration": 1, "mining_algorithm": "sm1",
                         "gateway_probabilities": "discovery"},
        "resource_model": {"num_iterations": n_iter, "num_evaluations_per_iteration": 1,
                           "resource_profiles": {"discovery_type": "differentiated_by_resource"}},
        "case_arrival": {"num_iterations": 1, "num_evaluations_per_iteration": 1},
    }
    settings = SimodSettings.from_yaml(config)
    event_log = EventLog.from_path(train_log_path=csv_path, log_ids=EventLogIDs.from_dict(ids),
                                   preprocessing_settings=settings.preprocessing, need_test_partition=False)
    t0 = time.time()
    simod_out = out_dir / "simod_run"
    Simod(settings, event_log=event_log, output_dir=simod_out).run()
    print(f"  Simod done in {(time.time() - t0) / 60:.1f} min")
    best = simod_out / "best_result"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(next(best.glob("*.bpmn")), bpmn)
    shutil.copy(next(p for p in best.glob("*.json") if p.name not in ("canonical_model.json", "runtimes.json")), js)
    return bpmn, js


# --------------------------------------------------------------------------------------------
# Durations -> Prosimos distributions (src/distribution_utils.py of the repo)
# --------------------------------------------------------------------------------------------
def find_best_fit_distribution(observed, rng):
    observed = np.asarray(observed, dtype=float)
    n = len(observed)
    if n == 0 or observed.min() == observed.max():
        return "fix", {"value": float(observed.min()) if n else 0.0}
    params, generated = {}, {}
    params["fix"] = {"value": observed.mean()}
    generated["fix"] = np.full(n, observed.mean())
    for name, dist in [("norm", stats.norm), ("expon", stats.expon), ("uniform", stats.uniform),
                       ("lognorm", stats.lognorm), ("gamma", stats.gamma)]:
        try:
            fit = dist.fit(observed)
            generated[name] = dist.rvs(*fit, size=n, random_state=rng)
        except Exception:
            continue
        if name == "norm":
            params[name] = {"mean": observed.mean(), "std": observed.std(), "min": 0, "max": observed.max()}
        elif name == "expon":
            params[name] = {"mean": observed.mean(), "min": 0, "max": observed.max()}
        elif name == "uniform":
            params[name] = {"min": observed.min(), "max": observed.max()}
        else:
            params[name] = {"mean": observed.mean(), "var": observed.std() ** 2, "min": observed.min(), "max": observed.max()}
    wd = {k: stats.wasserstein_distance(observed, v) for k, v in generated.items() if k in params}
    best = min(wd, key=wd.get)
    return best, params[best]


def write_json_with_durations(base_json: dict, bpmn_names: dict, durations: dict, rng, path: Path):
    """update_sim_params of the repo: every resource of a task gets the activity's distribution."""
    js = json.loads(json.dumps(base_json))
    for task in js["task_resource_distribution"]:
        act = bpmn_names.get(task["task_id"])
        dist, params = find_best_fit_distribution(durations.get(act, [0.0]), rng)
        for r in task["resources"]:
            r["distribution_name"] = dist
            r["distribution_params"] = [{"value": float(v)} for v in params.values()]
    path.write_text(json.dumps(js))


DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


def calendar_cumulative_minutes(periods: list) -> np.ndarray:
    """Cumulative working minutes over a week (Monday 00:00 = minute 0) for a Prosimos calendar."""
    mask = np.zeros(7 * 1440, dtype=bool)
    for p in periods:
        d0, d1 = DAYS.index(p["from"]), DAYS.index(p["to"])
        b = [int(x) for x in p["beginTime"].split(":")[:2]]
        e = [int(x) for x in p["endTime"].split(":")[:2]]
        b_min, e_min = b[0] * 60 + b[1], e[0] * 60 + e[1]
        if e_min == 0 and p["endTime"].startswith("00:00"):
            e_min = 1440
        for d in range(d0, d1 + 1):
            mask[d * 1440 + b_min: d * 1440 + e_min] = True
    return np.concatenate([[0], np.cumsum(mask)])


def working_seconds(starts: pd.Series, ends: pd.Series, cum: np.ndarray) -> np.ndarray:
    """Working time between start and end on a weekly calendar (minute resolution)."""
    monday = pd.Timestamp("1970-01-05", tz="UTC")  # a Monday

    def F(t):
        m = ((t - monday).dt.total_seconds() // 60).astype(np.int64).values
        week, rem = np.divmod(m, 7 * 1440)
        return week * cum[-1] + cum[rem]
    return (F(ends) - F(starts)).clip(min=0) * 60.0


def resource_calendar_cums(js: dict) -> dict:
    cals = {c["id"]: calendar_cumulative_minutes(c["time_periods"]) for c in js["resource_calendars"]}
    return {r["id"]: cals[r["calendar"]] for p in js["resource_profiles"] for r in p["resource_list"]}


def bpmn_task_names(bpmn_path: Path) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.parse(bpmn_path).getroot()
    return {el.get("id"): el.get("name") for el in root.iter() if el.tag.endswith("}task")}


# --------------------------------------------------------------------------------------------
# Simulation + error (src/metric_utils.compute_wass_err of the repo)
# --------------------------------------------------------------------------------------------
def inter_completion_times(case_ids, acts, ends) -> pd.DataFrame:
    d = pd.DataFrame({"case": case_ids, "act": acts, "end": ends}).sort_values(["case", "end"], kind="stable")
    d["delta"] = d.groupby("case")["end"].diff().dt.total_seconds().fillna(0.0)
    return d


def simulate(bpmn: Path, json_path: Path, total_cases: int, starting_at: str, out_csv: Path, seed: int) -> pd.DataFrame:
    import random
    from prosimos.simulation_engine import run_simulation
    random.seed(seed)
    np.random.seed(seed)
    run_simulation(str(bpmn), str(json_path), total_cases, stat_out_path=None, log_out_path=str(out_csv), starting_at=starting_at)
    sim = pd.read_csv(out_csv)
    sim["end_time"] = pd.to_datetime(sim["end_time"], utc=True, format="mixed")
    return sim


def wass_errors(real_ict: pd.DataFrame, sim: pd.DataFrame, n_cases: int, perc_head_tail: float, activities) -> dict:
    lo = int(n_cases * perc_head_tail / 2)
    ids = sorted(sim["case_id"].unique())
    keep = set(ids[lo:lo + n_cases])
    sim = sim[sim["case_id"].isin(keep)]
    s = inter_completion_times(sim["case_id"].values, sim["activity"].values, sim["end_time"].values)
    err = {}
    for a in activities:
        r = real_ict.loc[real_ict["act"] == a, "delta"].values
        v = s.loc[s["act"] == a, "delta"].values
        if len(v) == 0:
            v = np.zeros(len(r))
        err[a] = stats.wasserstein_distance(r, v) / 3600.0  # hours
    return err


# --------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    cs = args.case_study
    out_dir = BASE / "0b_vinci_trial" / cs
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir  # the Simod model is shared by both duration variants
    if args.working_time:
        out_dir = BASE / "0b_vinci_trial" / f"{cs}_working_time"
        out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"=== {cs}")
    fake = read_log(BASE / "case_studies" / cs / f"log_{cs}_no_start.xes")
    pix = read_log(BASE / "case_studies" / cs / f"log_{cs}.xes")
    assert (fake.index == pix.index).all() and (fake[END] == pix[END]).all()

    if cs in MISSING_RESOURCE:
        lanes = missing_resource_lanes(fake, MISSING_RESOURCE[cs])
        fake[RES] = lanes
        pix[RES] = lanes

    bpmn, base_json_path = discover_prosimos_model(pix, model_dir, cs, args.simod_iterations, args.rediscover)
    base_json = json.loads(base_json_path.read_text())
    names = bpmn_task_names(bpmn)

    mintime = compute_mintime(fake, cs)
    instant = fixed_instant_mask(fake, cs)
    activities = sorted(fake[ACT].unique())
    free_acts = sorted(fake.loc[~instant, ACT].unique())
    print(f"  {len(fake)} events, {fake[CASE].nunique()} cases | alpha searched for {len(free_acts)} activities: {free_acts}")

    n_cases = fake[CASE].nunique()
    total_cases = n_cases + int(n_cases * args.perc_head_tail)
    starting_at = fake[END].min().isoformat()
    real_ict = inter_completion_times(fake[CASE].values, fake[ACT].values, fake[END].values)

    res_cums = resource_calendar_cums(base_json) if args.working_time else None

    def evaluate(starts: pd.Series, tag: str) -> dict:
        if res_cums is None:
            dur = (fake[END] - starts).dt.total_seconds()
        else:
            dur = pd.Series(0.0, index=fake.index)
            for r, g in fake.groupby(RES):
                cum = res_cums.get(r)
                if cum is None:  # resource absent from the model: elapsed time
                    dur.loc[g.index] = (g[END] - starts[g.index]).dt.total_seconds()
                else:
                    dur.loc[g.index] = working_seconds(starts[g.index], g[END], cum)
        durations = {a: dur[fake[ACT] == a].values for a in activities}
        jp = out_dir / "iter_model.json"
        write_json_with_durations(base_json, names, durations, rng, jp)
        t0 = time.time()
        sim = simulate(bpmn, jp, total_cases, starting_at, out_dir / "iter_sim.csv", args.seed)
        err = wass_errors(real_ict, sim, n_cases, args.perc_head_tail, activities)
        print(f"  [{tag}] simulated in {time.time() - t0:.0f}s | mean error over searched activities "
              f"{np.mean([err[a] for a in free_acts]):.2f} h")
        return err

    rows = []
    # reference: the current pix estimate, through the same simulator and metric
    err_pix = evaluate(pix[START], "pix")
    rows.append({"iteration": "pix", **{f"err::{a}": err_pix[a] for a in activities}})

    # bisection, run.py of the repo
    alphas_tot = [{a: 0.0 for a in free_acts}, {a: 1.0 for a in free_acts}]
    errors = []
    for i in range(args.n_iterations):
        if i <= 1:
            alphas = dict(alphas_tot[i])
        else:
            alphas = {a: (alphas_tot[0][a] + alphas_tot[1][a]) / 2 for a in free_acts}
            alphas_tot.append(alphas)
        err = evaluate(apply_alphas(fake, mintime, instant, alphas), f"iter {i}")
        errors.append(err.copy())
        rows.append({"iteration": i, **{f"alpha::{a}": alphas[a] for a in free_acts}, **{f"err::{a}": err[a] for a in activities}})
        pd.DataFrame(rows).to_csv(out_dir / "iterations.csv", index=False)

        if i >= 2:
            for a in free_acts:
                e0, e1, ei = errors[0][a], errors[1][a], errors[i][a]
                if ei < e1 < e0 or e1 < ei < e0:   # replace the worse endpoint (left)
                    alphas_tot[0][a], errors[0][a] = alphas[a], ei
                elif ei < e0 < e1 or e0 < ei < e1:  # replace the worse endpoint (right)
                    alphas_tot[1][a], errors[1][a] = alphas[a], ei
            if i > 2 and all(abs(alphas_tot[-1][a] - alphas_tot[-2][a]) < 1e-3 for a in free_acts):
                print("  no more change in alpha: stop")
                break

    # best alpha per activity over all evaluated configurations
    hist = pd.DataFrame(rows[1:])
    best_alpha = {a: hist.loc[hist[f"err::{a}"].idxmin(), f"alpha::{a}"] for a in free_acts}
    starts_v = apply_alphas(fake, mintime, instant, best_alpha)
    err_v = evaluate(starts_v, "vinci best")
    rows.append({"iteration": "best", **{f"alpha::{a}": best_alpha[a] for a in free_acts}, **{f"err::{a}": err_v[a] for a in activities}})
    pd.DataFrame(rows).to_csv(out_dir / "iterations.csv", index=False)

    # ---------------- comparison vs pix ----------------
    dur_pix = (fake[END] - pix[START]).dt.total_seconds() / 3600
    dur_v = (fake[END] - starts_v).dt.total_seconds() / 3600
    dur_a1 = (fake[END] - apply_alphas(fake, mintime, instant, {a: 1.0 for a in free_acts})).dt.total_seconds() / 3600
    prev_case = fake.groupby(CASE)[END].shift(1)
    wait_pix = ((pix[START] - prev_case).dt.total_seconds() / 3600).clip(lower=0)
    wait_v = ((starts_v - prev_case).dt.total_seconds() / 3600).clip(lower=0)
    comp = []
    for a in activities:
        m = fake[ACT] == a
        comp.append({
            "activity": a, "n": int(m.sum()), "alpha": best_alpha.get(a, 0.0),
            "med_dur_pix_h": dur_pix[m].median(), "med_dur_vinci_h": dur_v[m].median(),
            "mean_dur_pix_h": dur_pix[m].mean(), "mean_dur_alpha1_h": dur_a1[m].mean(), "mean_dur_vinci_h": dur_v[m].mean(),
            "mean_wait_pix_h": wait_pix[m].mean(), "mean_wait_vinci_h": wait_v[m].mean(),
            "mean_abs_start_diff_h": (starts_v[m] - pix[START][m]).abs().dt.total_seconds().mean() / 3600,
            "err_pix_h": err_pix[a], "err_vinci_h": err_v[a],
        })
    comp = pd.DataFrame(comp).sort_values("n", ascending=False)
    comp.to_csv(out_dir / "comparison_vs_pix.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + comp.round(3).to_string(index=False))
    diff = (starts_v - pix[START]).abs().dt.total_seconds()
    print(f"\n  events whose start changes (>1 min): {(diff > 60).mean():.1%} | mean |delta start| {diff.mean() / 3600:.2f} h")
    print(f"  total processing time: vinci = {dur_v.sum() / dur_pix.sum():.1%} of pix")
    print(f"  mean error over searched activities: pix {np.mean([err_pix[a] for a in free_acts]):.2f} h, "
          f"vinci {np.mean([err_v[a] for a in free_acts]):.2f} h")


if __name__ == "__main__":  # prosimos/simod may spawn processes (Windows)
    main()
