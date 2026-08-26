#!/usr/bin/env python3
"""
Builds per-case evaluation tables from what 3_run_experiment.py /
4_run_recommendation_simulation.py have already produced on disk for a case
study: recommendations_{case_study}_{method}_top{rank}of{k}.csv files under
case_studies/<case_study>/recommendations/, and their simulated logs under
case_studies/<case_study>/prosit_simulation_results/<method>/<rank>/ (plus
the rank-independent baseline/ folder).

For every (method, rank) combination found on disk, builds one table with
one row per case_id:
  - case_study, method, rank, k_total
  - case:concept:name, rec_activity, rec_resource
  - sim_status_method_mean, sim_remaining_time_method_mean
      (averaged over that rank's n_sim simulation runs)
  - sim_status_baseline_mean, sim_remaining_time_baseline_mean
      (averaged over the baseline's n_sim runs -- rank/method independent,
      computed once per case study and reused)
  - pred_status_with_rec, pred_remaining_time_with_rec
      (predictive_outcome_model / predictive_time_model .joblib predictions
      on the case's prefix with NEXT_ACTIVITY/NEXT_RESOURCE = the recommendation)
  - pred_status_no_rec, pred_remaining_time_no_rec
      (same models on the prefix with the actual historical NEXT_ACTIVITY/
      NEXT_RESOURCE -- rank/method independent, computed once and reused)

Saves both the per-(method, rank) tables and, stacked across all ranks, one
final table per method, under case_studies/<case_study>/evaluation_tables/.

No delta_CO / delta_RT computation here -- those are trivial to derive from
these tables later (sim_status_method_mean - sim_status_baseline_mean, etc.)
and weren't needed for this pass.

Usage:
    python 5_result_computation.py
    python 5_result_computation.py --case_studies BPI12
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_normalization import fit_remaining_time_scalers, sigmoid_mm_to_remaining_time
from utils.simulation_functions import (
    getting_remaining_time,
    status_encoding,
    compute_res_and_status,
    case_id_name,
    start_date_name,
    end_date_name,
    activity_column_name,
    resource_column_name,
)
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.recommendation_functions import build_query_instances, align_query_instance_with_model
from utils.get_features import load_case_study, get_case_study_features

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------
CASE_STUDIES = ["BAC", "BPI12", "bpi17_after", "bpi17_before"]
METHODS = ["exhaustive", "nsga2"]
BASELINE_FOLDER_NAME = "baseline"
SIM_SUBDIR = "prosit_simulation_results"
OUTPUT_SUBDIR = "evaluation_tables"

# Activity marking a *positive* case outcome. Not needed for BAC, whose
# outcome logic (forbidden-activity set) is already hardcoded inside
# utils.simulation_functions.status_encoding.
ENCODED_ACTIVITY_BY_CASE_STUDY = {
    "BPI12": "O_ACCEPTED",
    "bpi17_after": "O_Accepted",
    "bpi17_before": "O_Accepted",
}

BPI12_DTYPE_CASE_STUDIES = {"BPI12"}

_SIM_FILE_RE = re.compile(r"^sim_(\d+)\.csv$")
_RANK_FILE_RE_TEMPLATE = r"^recommendations_{case_study}_{method}_top(\d+)of(\d+)\.csv$"


# ---------------------------------------------------------------------------
# Simulation-side helpers (per case_id averages over n_sim runs)
# ---------------------------------------------------------------------------
def _detect_n_sim(sim_folder: Path) -> int:
    """Count contiguous sim_<i>.csv files in sim_folder (ignores diagnostic CSVs like sim_1_unreachable_recommendations.csv)."""
    indices = [int(m.group(1)) for f in sim_folder.glob("sim_*.csv") if (m := _SIM_FILE_RE.match(f.name))]
    return max(indices) if indices else 0


def load_simulation_runs(sim_folder: Path, case_study: str, n_sim: int, encoded_activity) -> pd.DataFrame:
    """Load sim_1.csv..sim_n.csv, compute remaining_time/status per event, and suffix case ids with the run index so runs don't collide when concatenated.

    Args:
        sim_folder: directory containing sim_<i>.csv.
        case_study: name of the case study (e.g., "BAC", "BPI12").
        n_sim: number of simulation runs to load.
        encoded_activity: activity name marking a positive case outcome (see status_encoding).

    Returns:
        pandas.DataFrame: all n_sim runs concatenated, case ids suffixed "_<run_index>".
    """
    dataframes = []
    for i in range(n_sim):
        sim_path = sim_folder / f"sim_{i + 1}.csv"
        sim = pd.read_csv(sim_path, dtype={case_id_name: str})
        if case_study in BPI12_DTYPE_CASE_STUDIES:
            sim = convert_dtypes_bpi12(sim, "simulation")
        sim = sim[[case_id_name, start_date_name, end_date_name, activity_column_name, resource_column_name]]
        sim = getting_remaining_time(sim, case_id_name, end_date_name)
        sim = status_encoding(sim, case_study, encoded_activity)
        sim[case_id_name] = sim[case_id_name].astype(str) + "_" + str(i + 1)
        dataframes.append(sim)
    return pd.concat(dataframes, ignore_index=True).reset_index(drop=True)


def build_repl_id_map(test_log: pd.DataFrame) -> pd.Series:
    """Prefix cut-point per case (number of historical events - 1), keyed by str(case_id). Only depends on test_log, not on method/rank/recommendation."""
    repl_id_map = test_log.groupby(case_id_name).size() - 1
    repl_id_map.index = repl_id_map.index.astype(str)
    return repl_id_map


def compute_sim_averages(
    sim_folder: Path,
    case_study: str,
    encoded_activity,
    repl_id_map: pd.Series,
    case_ids: list[str],
) -> tuple[dict, dict]:
    """Mean remaining_time and status per case_id, averaged over every sim_<i>.csv found in sim_folder.

    Args:
        sim_folder: directory containing sim_<i>.csv (a <method>/<rank>/ folder, or the baseline/ folder).
        case_study: name of the case study.
        encoded_activity: activity name marking a positive case outcome.
        repl_id_map: case_id (str) -> prefix cut-point, from build_repl_id_map.
        case_ids: case ids (str) to compute averages for.

    Returns:
        (mean_remaining_time_by_case, mean_status_by_case): both {case_id: float}. Empty dicts if sim_folder has no sim_<i>.csv.
    """
    n_sim = _detect_n_sim(sim_folder)
    if n_sim == 0:
        return {}, {}

    test_simu = load_simulation_runs(sim_folder, case_study, n_sim, encoded_activity)

    repl_ids = [repl_id_map.get(cid, np.nan) for cid in case_ids]
    # res_1 is unused by compute_res_and_status's own logic, but for BPI12-like
    # case studies it unconditionally BPI12-converts rec_df expecting a res_1
    # column (see convert_dtypes_bpi12's 'simulation_' mode) -- a placeholder
    # avoids a KeyError there.
    rec_df = pd.DataFrame({case_id_name: case_ids, "repl_id": repl_ids, "res_1": ""})
    missing = rec_df["repl_id"].isna()
    if missing.any():
        bad_ids = rec_df.loc[missing, case_id_name].tolist()
        print(
            f"    [WARNING] {len(bad_ids)} case id(s) have no matching prefix "
            f"in test_log.csv and will be skipped for {sim_folder}: {bad_ids[:10]}"
            + (" ..." if len(bad_ids) > 10 else "")
        )
        rec_df = rec_df.loc[~missing].reset_index(drop=True)

    return compute_res_and_status(case_study, rec_df, test_simu, n_sim)


# ---------------------------------------------------------------------------
# Predictive-model-side helper (single (act, res) pair, no simulation)
# ---------------------------------------------------------------------------
def predict_batch(
    query_instances: list,
    acts: list,
    resources: list,
    predictive_outcome_model,
    predictive_time_model,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted (status, remaining_time) for many (query_instance, act, res) triples via two model
    calls total -- one predict_proba and one predict over the whole batch -- instead of one pair of
    calls per row. Each call into the sklearn/CatBoost pipelines pays a fixed overhead (ColumnTransformer
    fit-time transform, CatBoost's own per-call setup) that's roughly constant regardless of batch size,
    so batching hundreds of rows into one call is far cheaper than hundreds of single-row calls.
    """
    outcome_rows, time_rows = [], []
    for qi, act, res in zip(query_instances, acts, resources):
        o_row = align_query_instance_with_model(qi, predictive_outcome_model).iloc[0].to_dict()
        o_row["NEXT_ACTIVITY"] = act
        o_row["NEXT_RESOURCE"] = res
        outcome_rows.append(o_row)

        t_row = align_query_instance_with_model(qi, predictive_time_model).iloc[0].to_dict()
        t_row["NEXT_ACTIVITY"] = act
        t_row["NEXT_RESOURCE"] = res
        time_rows.append(t_row)

    predicted_status = predictive_outcome_model.predict_proba(pd.DataFrame(outcome_rows))[:, 1]
    predicted_rt = predictive_time_model.predict(pd.DataFrame(time_rows))
    return predicted_status, predicted_rt


# ---------------------------------------------------------------------------
# Per-case-study pipeline
# ---------------------------------------------------------------------------
def find_rank_recommendation_files(case_dir: Path, case_study: str, method: str) -> dict[int, tuple[int, Path]]:
    """rank -> (k_total, path) for every recommendations_{case_study}_{method}_top{rank}of{k}.csv found under case_dir/recommendations/."""
    rec_dir = case_dir / "recommendations"
    if not rec_dir.exists():
        return {}
    pattern = re.compile(_RANK_FILE_RE_TEMPLATE.format(case_study=re.escape(case_study), method=re.escape(method)))
    found: dict[int, tuple[int, Path]] = {}
    for f in rec_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            found[int(m.group(1))] = (int(m.group(2)), f)
    return dict(sorted(found.items()))


def compute_case_study_tables(base_dir: Path, case_study: str, output_subdir: str) -> None:
    """Build and save the per-rank and stacked evaluation tables for one case study (both methods)."""
    case_dir = base_dir / "case_studies" / case_study
    encoded_activity = ENCODED_ACTIVITY_BY_CASE_STUDY.get(case_study)

    print(f"  Loading data and models for {case_study}...")
    train_data, test_data, test_log = load_case_study(case_study)
    if case_study in BPI12_DTYPE_CASE_STUDIES:
        test_data = convert_dtypes_bpi12(test_data, "experiment")
        test_log = convert_dtypes_bpi12(test_log, "experiment")

    # pred_remaining_time_* comes out of the .joblib time model in "sigmoid_mm" space (see
    # get_case_study_features), a different scale than the simulation's raw-seconds remaining time.
    # fit_remaining_time_scalers reconstructs the StandardScaler/MinMaxScaler pair that produced that
    # space, so sigmoid_mm_to_remaining_time below can map predictions back to seconds.
    rt_std_scaler, rt_mm_scaler = fit_remaining_time_scalers(train_data)

    (
        predictive_outcome_model,
        predictive_time_model,
        case_id_name_local,
        _activity_column_name_local,
        _resource_column_name_local,
        _continuous_features,
        _categorical_features,
        _columns_to_remove,
    ) = get_case_study_features(case_study)

    query_instances_by_case = {
        str(cid): qi for cid, qi in build_query_instances(test_data, case_id_name_local).items()
    }
    repl_id_map = build_repl_id_map(test_log)
    all_case_ids = [str(c) for c in test_data[case_id_name_local].unique()]

    # -------------------------------------------------------------------
    # Case-study-level constants (rank/method independent) -- computed once.
    # -------------------------------------------------------------------
    print("  Computing baseline simulation averages (rank/method independent)...")
    baseline_folder = case_dir / SIM_SUBDIR / BASELINE_FOLDER_NAME
    baseline_rt, baseline_status = compute_sim_averages(
        baseline_folder, case_study, encoded_activity, repl_id_map, all_case_ids
    )
    if not baseline_rt:
        print(f"    [WARNING] No baseline simulation runs found at {baseline_folder}.")

    print("  Computing 'no recommendation' .joblib predictions (rank/method independent)...")
    no_rec_ids = [cid for cid in all_case_ids if query_instances_by_case.get(cid) is not None]
    pred_no_rec: dict[str, tuple[float, float]] = {}
    if no_rec_ids:
        no_rec_qis = [query_instances_by_case[cid] for cid in no_rec_ids]
        no_rec_acts = [qi["NEXT_ACTIVITY"] for qi in no_rec_qis]
        no_rec_ress = [qi["NEXT_RESOURCE"] for qi in no_rec_qis]
        pred_status_no_arr, pred_rt_no_arr = predict_batch(
            no_rec_qis, no_rec_acts, no_rec_ress, predictive_outcome_model, predictive_time_model
        )
        pred_rt_no_arr = sigmoid_mm_to_remaining_time(pred_rt_no_arr, rt_std_scaler, rt_mm_scaler)
        pred_no_rec = {
            cid: (float(status), float(rt)) for cid, status, rt in zip(no_rec_ids, pred_status_no_arr, pred_rt_no_arr)
        }

    # -------------------------------------------------------------------
    # Per-(method, rank) tables.
    # -------------------------------------------------------------------
    out_dir = case_dir / output_subdir
    for method in METHODS:
        rank_files = find_rank_recommendation_files(case_dir, case_study, method)
        if not rank_files:
            print(f"  [SKIPPED] {case_study}/{method}: no recommendations_{case_study}_{method}_top*of*.csv found.")
            continue

        rank_tables = []
        for rank, (k_total, rec_path) in rank_files.items():
            sim_folder = case_dir / SIM_SUBDIR / method / str(rank)
            if not sim_folder.exists() or _detect_n_sim(sim_folder) == 0:
                print(f"    [SKIPPED] {case_study}/{method} rank {rank}/{k_total}: no simulations at {sim_folder}.")
                continue

            print(f"    Processing {case_study}/{method} rank {rank}/{k_total}...")
            rec_df = pd.read_csv(rec_path, dtype={case_id_name_local: str})
            if case_study in BPI12_DTYPE_CASE_STUDIES:
                rec_df = convert_dtypes_bpi12(rec_df, "simulation_prep")

            missing = rec_df["Next_activity"].isna() | rec_df["Next_resource"].isna()
            if missing.any():
                bad_ids = rec_df.loc[missing, case_id_name_local].tolist()
                print(
                    f"      [WARNING] {len(bad_ids)} case id(s) have no recommendation "
                    f"and will be skipped: {bad_ids[:10]}" + (" ..." if len(bad_ids) > 10 else "")
                )
                rec_df = rec_df.loc[~missing].reset_index(drop=True)

            case_ids_rank = rec_df[case_id_name_local].tolist()
            method_rt, method_status = compute_sim_averages(
                sim_folder, case_study, encoded_activity, repl_id_map, case_ids_rank
            )

            valid_rows = []
            for _, r in rec_df.iterrows():
                cid = r[case_id_name_local]
                qi = query_instances_by_case.get(cid)
                if qi is None:
                    print(f"      [WARNING] case id {cid} not found in query instances, skipping.")
                    continue
                valid_rows.append((cid, r["Next_activity"], r["Next_resource"], qi))

            rows = []
            if valid_rows:
                rec_cids, rec_acts, rec_resources, rec_qis = zip(*valid_rows)
                pred_status_with_arr, pred_rt_with_arr = predict_batch(
                    list(rec_qis), list(rec_acts), list(rec_resources), predictive_outcome_model, predictive_time_model
                )
                pred_rt_with_arr = sigmoid_mm_to_remaining_time(pred_rt_with_arr, rt_std_scaler, rt_mm_scaler)

                for cid, act, res, pred_status_with, pred_rt_with in zip(
                    rec_cids, rec_acts, rec_resources, pred_status_with_arr, pred_rt_with_arr
                ):
                    pred_status_no, pred_rt_no = pred_no_rec.get(cid, (np.nan, np.nan))
                    rows.append({
                        "case_study": case_study,
                        "method": method,
                        "rank": rank,
                        "k_total": k_total,
                        case_id_name_local: cid,
                        "rec_activity": act,
                        "rec_resource": res,
                        "sim_status_method_mean": method_status.get(cid, np.nan),
                        "sim_remaining_time_method_mean": method_rt.get(cid, np.nan),
                        "sim_status_baseline_mean": baseline_status.get(cid, np.nan),
                        "sim_remaining_time_baseline_mean": baseline_rt.get(cid, np.nan),
                        "pred_status_with_rec": float(pred_status_with),
                        "pred_remaining_time_with_rec": float(pred_rt_with),
                        "pred_status_no_rec": pred_status_no,
                        "pred_remaining_time_no_rec": pred_rt_no,
                    })

            rank_df = pd.DataFrame(rows)
            method_out_dir = out_dir / method
            method_out_dir.mkdir(parents=True, exist_ok=True)
            rank_path = method_out_dir / f"rank_{rank}_of_{k_total}.csv"
            rank_df.to_csv(rank_path, index=False)
            print(f"      Saved {len(rank_df)} rows to {rank_path}")
            rank_tables.append(rank_df)

        if rank_tables:
            stacked = pd.concat(rank_tables, ignore_index=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            stacked_path = out_dir / f"{method}_all_ranks.csv"
            stacked.to_csv(stacked_path, index=False)
            print(f"  [OK] {case_study}/{method}: saved {len(stacked)} rows ({len(rank_tables)} ranks) to {stacked_path}")


def main():
    """Build per-case evaluation tables (simulation averages + .joblib predictions, with/without recommendation) for every configured case study, from whatever recommendations/simulations already exist on disk."""
    parser = argparse.ArgumentParser(
        description="Build per-case evaluation tables from existing recommendation/simulation files."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_studies", type=str, default=",".join(CASE_STUDIES),
                         help=f"Comma-separated case studies to process (default: {','.join(CASE_STUDIES)})")
    parser.add_argument("--output_subdir", type=str, default=OUTPUT_SUBDIR,
                         help=f"Subfolder under each case study to save tables into (default: {OUTPUT_SUBDIR})")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    case_studies = [c.strip() for c in args.case_studies.split(",") if c.strip()]

    for case_study in case_studies:
        print(f"Processing {case_study}...")
        try:
            compute_case_study_tables(base_dir, case_study, args.output_subdir)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [SKIPPED] {case_study}: {e}")


if __name__ == "__main__":
    main()
