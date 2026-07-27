
import argparse
from pathlib import Path
 
import pandas as pd
 
from utils.simulation_functions import (
    preparing_data_for_simulation,
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
 
# ---------------------------------------------------------------------------
# Fixed configuration (as given)
# ---------------------------------------------------------------------------
CASE_STUDIES = ["BPI12", "BAC", "bpi17_after", "bpi17_before"]
#CASE_STUDIES = ["BAC"]
METHODS = ["exhaustive", "nsga2"]
BASELINE_FOLDER_NAME = "baseline"
SIM_SUBDIR = "prosit_simulation_results"
 
# Activity marking a *positive* case outcome. Not needed for BAC, whose
# outcome logic (forbidden-activity set) is already hardcoded inside
# utils.simulation_functions.status_encoding.
ENCODED_ACTIVITY_BY_CASE_STUDY = {
    "BPI12": "O_ACCEPTED",
    "bpi17_after": "O_Accepted",
    "bpi17_before": "O_Accepted",
}
 
BPI12_DTYPE_CASE_STUDIES = {
    "BPI12", 
}
 
 
def load_simulation_runs(sim_folder: Path, case_study: str, n_sim: int, encoded_activity):
    """
    Reads and prepares multiple simulation run files from a specified folder.
    This function loads `sim_1.csv` through `sim_n.csv`, applies case-study-specific 
    data type conversions (e.g., for BPI12), filters necessary columns, computes the 
    remaining time, and encodes the status. It also appends a unique suffix to the 
    case IDs for each simulation run to distinguish them before concatenating everything 
    into a single DataFrame.

    Args:
        sim_folder (pathlib.Path): The directory containing the simulation CSV files.
        case_study (str): The name of the case study (e.g., "BAC", "BPI12").
        n_sim (int): The total number of simulation runs to load.
        encoded_activity (str): The activity name that marks a positive case outcome.

    Returns:
        pandas.DataFrame: A single concatenated DataFrame containing all prepared simulation runs.
    """
    dataframes = []
    for i in range(n_sim):
        sim_path = sim_folder / f"sim_{i + 1}.csv"
        sim = pd.read_csv(sim_path, dtype={case_id_name: str})
        if case_study in BPI12_DTYPE_CASE_STUDIES:
            sim = convert_dtypes_bpi12(sim, "simulation")
        sim = sim[[case_id_name, start_date_name, end_date_name,
                    activity_column_name, resource_column_name]]
        sim = getting_remaining_time(sim, case_id_name, end_date_name)
        sim = status_encoding(sim, case_study, encoded_activity)
        sim[case_id_name] = sim[case_id_name].astype(str) + "_" + str(i + 1)
        dataframes.append(sim)
    return pd.concat(dataframes, ignore_index=True).reset_index(drop=True)
 
 
def find_recommendations_csv(case_dir: Path, case_study: str, method: str) -> Path:
    candidates = [
        case_dir / "recommendations" / f"recommendations_{case_study}_{method}.csv",
        case_dir / f"recommendations_{case_study}_{method}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Recommendations CSV not found. Tried: " + ", ".join(str(c) for c in candidates)
    )
 
 
def compute_case_study_method(base_dir: Path, case_study: str, method: str, n_sim: int) -> dict:
    """
    Computes the performance metrics (delta_CO and delta_RT) for a specific recommendation method.
    This function aligns the recommended simulations against the baseline simulations, 
    calculating the changes in Case Outcome (delta_CO) and Remaining Time (delta_RT). 
    It drops unmatchable traces, extracts the relevant features, and calculates both 
    the relative deltas and the raw averages for diagnostics.

    Args:
        base_dir (pathlib.Path): The root directory containing the 'case_studies' folder.
        case_study (str): The name of the case study being evaluated.
        method (str): The recommendation method being evaluated.
        n_sim (int): The number of simulation runs to process.

    Returns:
        dict: A dictionary containing the summary statistics:
            - case_study (str): The name of the case study.
            - method (str): The recommendation method.
            - n_traces (int): The number of common traces evaluated.
            - delta_co (float): The average change in case outcome.
            - delta_rt (float): The average relative change in remaining time.
            - mean_status_method (float): Raw average outcome status for the method.
            - mean_status_baseline (float): Raw average outcome status for the baseline.
            - mean_rt_method (float): Raw average remaining time for the method.
            - mean_rt_baseline (float): Raw average remaining time for the baseline.
            
    Raises:
        ValueError: If there are no common traces between the baseline and the method results.
    """
    case_dir = base_dir / "case_studies" / case_study
    encoded_activity = ENCODED_ACTIVITY_BY_CASE_STUDY.get(case_study)
 
    result_csv = find_recommendations_csv(case_dir, case_study, method)
    result_df = pd.read_csv(result_csv, dtype={case_id_name: str})
    test_log = pd.read_csv(case_dir / "test_log.csv", dtype={case_id_name: str})
 
    rec_df = preparing_data_for_simulation(result_df, test_log, case_id_name, end_date_name, case_study)
 
    # Defensive check: repl_id is NaN when a case id in the recommendations
    # file has no matching prefix in test_log.csv (or some other mismatch
    # between the two files). Rather than crashing on this (KeyError: nan
    # inside compute_res_and_status), drop those cases and report them, so
    # the rest of the analysis can still run.
    nan_repl_mask = rec_df["repl_id"].isna()
    if nan_repl_mask.any():
        bad_ids = rec_df.loc[nan_repl_mask, case_id_name].tolist()
        preview = bad_ids[:10]
        print(
            f"  [WARNING] {case_study}/{method}: {len(bad_ids)} case id(s) have no "
            f"matching prefix in test_log.csv (repl_id is NaN) and will be "
            f"excluded from this comparison: {preview}"
            + (" ..." if len(bad_ids) > 10 else "")
        )
        rec_df = rec_df.loc[~nan_repl_mask].reset_index(drop=True)
 
    # --- Method (recommendation) simulations ---
    method_folder = case_dir / SIM_SUBDIR / method
    method_sim = load_simulation_runs(method_folder, case_study, n_sim, encoded_activity)
    rt_method, status_method = compute_res_and_status(case_study, rec_df, method_sim, n_sim)
 
    # --- Baseline (no-recommendation) simulations ---
    baseline_folder = case_dir / SIM_SUBDIR / BASELINE_FOLDER_NAME
    baseline_sim = load_simulation_runs(baseline_folder, case_study, n_sim, encoded_activity)
    rt_baseline, status_baseline = compute_res_and_status(case_study, rec_df, baseline_sim, n_sim)
 
    # --- align on the traces present in both sets of results ---
    common_ids = sorted(set(rt_method) & set(rt_baseline))
    only_baseline = set(rt_baseline) - set(rt_method)
    only_method = set(rt_method) - set(rt_baseline)
    if only_baseline or only_method:
        print(
            f"  [WARNING] {case_study}/{method}: trace id mismatch between "
            f"baseline and method results ({len(only_baseline)} only in "
            f"baseline, {len(only_method)} only in method). Using the "
            f"{len(common_ids)} traces common to both."
        )
 
    if not common_ids:
        raise ValueError(
            f"No common traces between baseline and method results for "
            f"{case_study}/{method} after alignment -- cannot compute deltas. "
            f"({len(only_baseline)} only in baseline, {len(only_method)} only in method)"
        )
 
    co_diffs = [status_method[t] - status_baseline[t] for t in common_ids]
    rt_diffs = [rt_baseline[t] - rt_method[t] for t in common_ids]
    rt_gt_values = [rt_baseline[t] for t in common_ids]
 
    delta_co = sum(co_diffs) / len(co_diffs)
    mean_rt_diff = sum(rt_diffs) / len(rt_diffs)
    mean_rt_gt = sum(rt_gt_values) / len(rt_gt_values)
    delta_rt = mean_rt_diff / mean_rt_gt if mean_rt_gt != 0 else float("nan")
 
    # Raw diagnostics: average of each quantity BEFORE taking the delta, to
    # help spot whether an unexpected result comes from the baseline side,
    # the method side, or is genuinely a result of the recommendation.
    mean_status_method = sum(status_method[t] for t in common_ids) / len(common_ids)
    mean_status_baseline = sum(status_baseline[t] for t in common_ids) / len(common_ids)
    mean_rt_method = sum(rt_method[t] for t in common_ids) / len(common_ids)
    mean_rt_baseline = sum(rt_baseline[t] for t in common_ids) / len(common_ids)
 
    return {
        "case_study": case_study,
        "method": method,
        "n_traces": len(common_ids),
        "delta_co": delta_co,
        "delta_rt": delta_rt,
        "mean_status_method": mean_status_method,
        "mean_status_baseline": mean_status_baseline,
        "mean_rt_method": mean_rt_method,
        "mean_rt_baseline": mean_rt_baseline,
    }
 
 
def main():
    """
    Main entry point for the script to compute delta_CO and delta_RT metrics.

    Parses command-line arguments to determine the base directory, number of simulations, 
    and output file. It iterates over the globally configured CASE_STUDIES and METHODS, 
    computes the metrics for each valid combination, handles missing files or errors gracefully, 
    and exports a final summary table to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Compute delta_CO / delta_RT (paper Eq. 6) for every "
                     "case_study/method vs. its baseline."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--n_sim", type=int, default=10,
                         help="Number of simulation runs to average over (default: 10)")
    parser.add_argument("--out_csv", type=str, default="delta_co_rt_results.csv",
                         help="Where to save the summary table")
    args = parser.parse_args()
 
    base_dir = Path(args.base_dir)
    results = []
    for case_study in CASE_STUDIES:
        for method in METHODS:
            print(f"Processing {case_study}/{method}...")
            try:
                res = compute_case_study_method(base_dir, case_study, method, args.n_sim)
                results.append(res)
                print(
                    f"  [OK] delta_CO={res['delta_co']:.4f}, "
                    f"delta_RT={res['delta_rt']:.4f} ({res['n_traces']} traces)"
                )
                print(
                    f"       raw: status_method={res['mean_status_method']:.4f}, "
                    f"status_baseline={res['mean_status_baseline']:.4f}, "
                    f"rt_method={res['mean_rt_method']:.1f}, "
                    f"rt_baseline={res['mean_rt_baseline']:.1f}"
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  [SKIPPED] {e}")
 
    if not results:
        print("No results were computed -- check your paths.")
        return
 
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved summary table to {args.out_csv}")
    print(df.to_string(index=False))
 
 
if __name__ == "__main__":
    main()
 