#!/usr/bin/env python3
"""
Estimate the activity start times of a case-study log with the simulation-based method of Vinci et al.,
"Improving organizational processes in healthcare through simulation-driven resource allocation",
AI in Medicine 176 (2026), Appendix A -- code in start_time_estimator/.

!! Run it from the conda env `prosimos_env` (pix_env + prosimos 2.1.0), user site-packages disabled:
!!     C:\\Users\\Utente\\.conda\\envs\\prosimos_env\\python.exe -s 0_estimate_start_times_vinci.py

For each case study (files under case_studies/<cs>/):

  STEP 0  read the log without W_ activities, log_<cs>_no_start_W_filtered.xes (0_filtering_W_activities.py);
          only its completion times are used
  STEP 1  discover the Prosimos simulation model from the fitness-1 Petri net
          discovery_output/petri_net_fitness1_<cs>.pnml (discovery/petri_net_discovery_fitness1.py):
          BPMN + JSON with resource pools, calendars, inter-arrival time, branching probabilities
  STEP 2  estimate one alpha per activity by bisection, simulating with Prosimos at every iteration
  STEP 3  write log_<cs>.xes: the input log with start:timestamp = end - alpha * (end - mintime)

Intermediate files go to case_studies/<cs>/start_time_estimator/: simulation_model.bpmn,
simulation_parameters.json, alphas.json, iterations.csv (alpha and error of every activity at every
iteration), and the files of the last simulation.

Usage 
-----
    python -s 0_estimate_start_times_vinci.py
    python -s 0_estimate_start_times_vinci.py --case_studies BPI12
    C:\\Users\\Utente\\.conda\\envs\\prosimos_env\\python.exe -s 0_estimate_start_times_vinci.py
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import pm4py

from start_time_estimator.estimator import ACT, CASE, END, RES, START, estimate_alphas, minimum_start_times, start_times_from_alphas
from start_time_estimator.simulation_model import discover_simulation_model

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
CASE_STUDIES = ["BPI12", "bpi17_before", "bpi17_after"]
INPUT_LOG = "log_{cs}_no_start_W_filtered.xes"
PETRI_NET = "discovery_output/petri_net_fitness1_{cs}.pnml"
OUTPUT_LOG = "log_{cs}.xes"
WORK_DIR = "start_time_estimator"

# Step 1 (discovery)
CALENDAR_COVERAGE = 0.95  # calendars: weekdays, then hours, kept until they cover this share of the completions
# Step 2 (estimation, defaults of the repository's run.py)
N_ITERATIONS = 20         # bisection iterations = simulations (all run; default of the repository's run.py)
PERC_HEAD_TAIL = 0.1      # extra simulated cases, half discarded at the head (warm-up), half at the tail
SEED = 42


def process_case_study(cs):
    case_dir = BASE_DIR / "case_studies" / cs
    work_dir = case_dir / WORK_DIR
    t0 = time.time()

    # ---- STEP 0: the log (completion events, per case in completion order; file order breaks ties) ----
    input_path = case_dir / INPUT_LOG.format(cs=cs)
    print(f"\n=== {cs}\n[0] reading {input_path.name}")
    raw = pd.DataFrame(pm4py.read_xes(str(input_path), show_progress_bar=False)).reset_index(drop=True)
    log = raw[[CASE, ACT, RES, END]].sort_values([CASE, END], kind="stable").copy()
    log[RES] = log[RES].astype(str)
    log[END] = pd.to_datetime(log[END], utc=True)
    print(f"    {len(log)} events, {log[CASE].nunique()} cases, {log[ACT].nunique()} activities, {log[RES].nunique()} resources")

    # ---- STEP 1: the simulation model ----
    print(f"[1] discovering the simulation model from {PETRI_NET.format(cs=cs)}")
    bpmn_path, json_path = discover_simulation_model(log, case_dir / PETRI_NET.format(cs=cs), work_dir,
                                                     CALENDAR_COVERAGE, SEED)
    simulation_parameters = json.loads(json_path.read_text())
    print(f"    {len(simulation_parameters['resource_profiles'])} resource pools, "
          f"arrivals {simulation_parameters['arrival_time_distribution']['distribution_name']}, "
          f"{len(simulation_parameters['gateway_branching_probabilities'])} XOR splits -> {json_path.name}")

    # ---- STEP 2: the alphas ----
    print(f"[2] estimating the alphas ({N_ITERATIONS} simulations)")
    alphas, history = estimate_alphas(log, bpmn_path, simulation_parameters, work_dir, N_ITERATIONS, PERC_HEAD_TAIL, SEED)
    history.to_csv(work_dir / "iterations.csv", index=False)
    (work_dir / "alphas.json").write_text(json.dumps(alphas, indent=2))
    for a in sorted(alphas):
        print(f"    {a}: alpha = {alphas[a]:.3f}")

    # ---- STEP 3: the log with the estimated start times (same rows, order and attributes as the input) ----
    starts = start_times_from_alphas(log, minimum_start_times(log), alphas)
    out = raw.copy()
    out[START] = starts.reindex(out.index)
    formatted = pm4py.format_dataframe(out, case_id=CASE, activity_key=ACT, timestamp_key=END, start_timestamp_key=START)
    output_path = case_dir / OUTPUT_LOG.format(cs=cs)
    pm4py.write_xes(formatted[list(raw.columns)], str(output_path))  # drop pm4py's helper columns
    print(f"[3] wrote {output_path} ({(time.time() - t0) / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Start-time estimation of Vinci et al. (2026, Appendix A).")
    parser.add_argument("--case_studies", default=",".join(CASE_STUDIES), help="Comma-separated case studies.")
    for cs in [c.strip() for c in parser.parse_args().case_studies.split(",") if c.strip()]:
        process_case_study(cs)


if __name__ == "__main__":
    main()
