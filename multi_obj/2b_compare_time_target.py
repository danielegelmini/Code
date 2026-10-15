#!/usr/bin/env python3
"""
2b_compare_time_target.py

Comparison of three ways of learning the remaining time (sigmoid_mm removed on 2026-10-12). The target is the
remaining time in seconds; the CatBoost regressor learns it in one of three model spaces and its predictions are
always transformed back to seconds (log spaces: log-normal mean; the median is reported too):
  - "none":          seconds as they are;
  - "log1p_seconds": log(1 + seconds) -- a pure log, every error relative;
  - "log1p_days":    log(1 + seconds / 86400) -- linear below about one day, relative above.

Only the time regressor is trained, exactly as in step 2 (2_training_predictive_model.py: same search space,
Optuna trials, early stopping, doubled training set with the no-recommendation copies, sigma-scaling), once per
transform. The models go to case_studies/<case>/model/time_target_test/<transform>/ (the production models in
case_studies/<case>/model/ are not touched). Errors are reported in days for all three; the calibration diagnostics
are in each model's own space, where the predictive distribution is Gaussian (a +/- k sigma interval there is the
same quantile interval in seconds, so the coverages are comparable).

Writes, in this folder: 2b_time_target_comparison.txt and .csv (rewritten after every trained model).

Truncated log-normal (2026-10-13): with a log space the plain log-normal mean e^(mu + var/2) explodes on the few rows
with a very large model-space variance. Models trained since then store U = the maximum remaining time of the training
set and use the log-normal TRUNCATED at U (mean, median and std of T given T <= U; see UncertaintyRegressor), so the
main table of a new run already uses it. For models trained before, --truncation_only does not train: it loads the
saved log-space models, sets U from the training data and appends a "TRUNCATED LOG-NORMAL" section (and rows
"<transform>+trunc" in the csv), next to the untruncated results.

Usage
-----
    python 2b_compare_time_target.py
    python 2b_compare_time_target.py --case_studies BPI12 --optuna_trials 20
    python 2b_compare_time_target.py --truncation_only     (no training: the saved models, truncated at U)
"""
import argparse
import importlib
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.get_features import get_features
from utils.pre_processing_functions import convert_dtypes_bpi12, add_no_recommendation_copy
from utils.predictive_models_functions import train_ml_model, TIME_TARGET, SECONDS_PER_DAY

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
STEP2 = importlib.import_module("2_training_predictive_model")  # search spaces, trials, early stopping of step 2
BPI12_LIKE = {"BPI12_not_reordered", "BPI12_not_reordered_sim", "BPI12", "BPI12_sim"}
END, START = "time:timestamp", "start:timestamp"


def load_training_data(case_study, data_suffix):
    """Train and test data as step 2 loads them (BPI12 dtypes, no-recommendation copies of the training rows)."""
    data_dir = HERE / "case_studies" / case_study
    train_data = pd.read_csv(data_dir / f"train_data{data_suffix}.csv", parse_dates=[END, START])
    test_data = pd.read_csv(data_dir / f"test_data{data_suffix}.csv", parse_dates=[END, START])
    if case_study in BPI12_LIKE:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data = convert_dtypes_bpi12(test_data, "experiment")
    return add_no_recommendation_copy(train_data), test_data


def summary_row(case_study, transform, res, minutes, n_train, n_test):
    unc = res["uncertainty"]
    return {
        "case_study": case_study, "transform": transform,
        "train_rows": n_train, "test_rows": n_test,
        "train_RMSE_d": res["train_score"], "train_MAE_d": unc.get("train_mae"),
        "test_RMSE_d": res["test_score"], "test_MAE_d": unc["test_mae"],
        "test_RMSE_median_d": unc.get("test_rmse_median"), "test_MAE_median_d": unc.get("test_mae_median"),
        "cov_1sigma": unc["coverage_1sigma"], "cov_2sigma": unc["coverage_2sigma"],
        "ENCE": unc["ence"], "sigma_scale": unc["sigma_scale"],
        "mean_std_d": unc["mean_std"], "epistemic_share": unc["epistemic_var_fraction"],
        "trees": res["n_trees_final_model"], "trials_complete": res["n_trials_complete"],
        "best_params": str(res["best_params"]), "minutes": minutes,
    }


def write_report(rows, args, out_txt, out_csv):
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    lines = ["=" * 110,
             "REMAINING TIME: SECONDS vs LOG(1 + SECONDS) vs LOG(1 + DAYS) AS MODEL SPACE (time regressor only, as in step 2)",
             f"Written by 2b_compare_time_target.py - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
             f"Optuna trials {args.optuna_trials} | data train_data{args.data_suffix}.csv / test_data{args.data_suffix}.csv",
             "=" * 110,
             "RMSE / MAE      : in days, on the test set (train: on the doubled training set). log spaces: log-normal mean;",
             "                  'median' columns: the log-normal median as point prediction.",
             "cov +/-1s, +/-2s: share of test rows inside the predictive interval (targets 68.3% / 95.4%), recalibrated;",
             "                  computed in each model's space = the same quantile interval in seconds.",
             "ENCE            : expected normalized calibration error (recalibrated, model space). std: mean predictive std in days.",
             ""]
    # huge values (e.g. the log-normal mean of a model with a very large log-space variance) in scientific notation
    fmt = lambda v, f: "-" if v is None or pd.isna(v) else (format(v, ".2e") if abs(v) >= 1e6 else format(v, f))
    for cs, d in table.groupby("case_study", sort=False):
        lines.append(f"{cs}  (train rows {int(d['train_rows'].iloc[0])}, test rows {int(d['test_rows'].iloc[0])})")
        view = pd.DataFrame({
            "transform": d["transform"],
            "test RMSE": d["test_RMSE_d"].map(lambda v: fmt(v, ".3f")),
            "test MAE": d["test_MAE_d"].map(lambda v: fmt(v, ".3f")),
            "RMSE median": d["test_RMSE_median_d"].map(lambda v: fmt(v, ".3f")),
            "MAE median": d["test_MAE_median_d"].map(lambda v: fmt(v, ".3f")),
            "train MAE": d["train_MAE_d"].map(lambda v: fmt(v, ".3f")),
            "cov +/-1s": d["cov_1sigma"].map(lambda v: fmt(100 * v, ".1f")),
            "cov +/-2s": d["cov_2sigma"].map(lambda v: fmt(100 * v, ".1f")),
            "ENCE %": d["ENCE"].map(lambda v: fmt(100 * v, ".1f")),
            "std d": d["mean_std_d"].map(lambda v: fmt(v, ".2f")),
            "trees": d["trees"],
            "min": d["minutes"].map(lambda v: fmt(v, ".1f")),
        })
        lines += [view.to_string(index=False), ""]
    Path(out_txt).write_text("\n".join(lines), encoding="utf-8")


TRUNC_HEADER = "TRUNCATED LOG-NORMAL"


def evaluate_truncation(args):
    """No training: for every saved log-space model, the test (and training) errors with the predictive log-normal
    truncated at U = the maximum remaining time of the training set, appended to the txt/csv report (an earlier
    truncation section is replaced)."""
    rows, lines = [], []
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        train_data, test_data = load_training_data(cs, args.data_suffix)
        y_train = train_data["remaining_time"].to_numpy(dtype=float) / SECONDS_PER_DAY
        y_test = test_data["remaining_time"].to_numpy(dtype=float) / SECONDS_PER_DAY
        max_seconds = float(train_data["remaining_time"].max())
        for transform in [t.strip() for t in args.transforms.split(",") if t.strip() and t.strip() != "none"]:
            model_dir = (Path(args.models_dir) / cs if args.models_dir else HERE / "case_studies" / cs / "model" / "time_target_test") / transform
            pipeline = joblib.load(model_dir / f"catboost_model_{TIME_TARGET}.joblib")
            step, transformation = pipeline.named_steps["prediction"], pipeline.named_steps["transformation"]
            X_train, X_test = transformation.transform(train_data), transformation.transform(test_data)
            row = {"case_study": cs, "transform": f"{transform}+trunc", "train_rows": len(train_data),
                   "test_rows": len(test_data), "U_days": max_seconds / SECONDS_PER_DAY}
            for point in ("mean", "median"):
                for truncated in (False, True):
                    step.max_seconds, step.log_point = (max_seconds if truncated else None), point
                    pred = step.predict(X_test) / SECONDS_PER_DAY
                    tag = f"{point}{'_trunc' if truncated else ''}"
                    row[f"test_RMSE_{tag}"] = float(np.sqrt(np.mean((pred - y_test) ** 2)))
                    row[f"test_MAE_{tag}"] = float(np.mean(np.abs(pred - y_test)))
                    row[f"pred_over_U_{tag}"] = int(np.sum(pred > max_seconds / SECONDS_PER_DAY + 1e-9))
                    row[f"pred_max_d_{tag}"] = float(pred.max())
            step.max_seconds, step.log_point = max_seconds, "mean"
            row["train_MAE_mean_trunc"] = float(np.mean(np.abs(step.predict(X_train) / SECONDS_PER_DAY - y_train)))
            unc = step.predict_uncertainty(X_test)
            row["mean_std_d_trunc"] = float(np.mean(unc["total_std"]) / SECONDS_PER_DAY)
            rows.append(row)
            print(f"{cs} {transform}: test MAE {row['test_MAE_mean']:.3f} -> truncated {row['test_MAE_mean_trunc']:.3f} d")

    table = pd.DataFrame(rows)
    fmt = lambda v, f: "-" if v is None or pd.isna(v) else (format(v, ".2e") if abs(v) >= 1e6 else format(v, f))
    lines += ["", "=" * 110,
              f"{TRUNC_HEADER} (added {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 2b_compare_time_target.py --truncation_only;",
              "the same saved models as above, no retraining; only the conversion from the model space to days changes)",
              "=" * 110,
              "U        : maximum remaining time of the training set (days); the predictive log-normal is truncated at U:",
              "           mean = E[T | T <= U], median = median of T given T <= U (see UncertaintyRegressor).",
              "plain    : log-normal without truncation (as in the tables above; 'mean' = the 'test MAE/RMSE' columns there).",
              ">U       : test predictions above U (days); max: largest test prediction (days).",
              "Calibration (coverage, ENCE) is in the model space and does not change with the truncation.", ""]
    for cs, d in table.groupby("case_study", sort=False):
        lines.append(f"{cs}  (U = {d['U_days'].iloc[0]:.1f} days)")
        view = pd.DataFrame({
            "transform": d["transform"],
            "MAE mean": d["test_MAE_mean"].map(lambda v: fmt(v, ".3f")),
            "MAE mean trunc": d["test_MAE_mean_trunc"].map(lambda v: fmt(v, ".3f")),
            "RMSE mean": d["test_RMSE_mean"].map(lambda v: fmt(v, ".3f")),
            "RMSE mean trunc": d["test_RMSE_mean_trunc"].map(lambda v: fmt(v, ".3f")),
            ">U mean": d["pred_over_U_mean"],
            "max mean": d["pred_max_d_mean"].map(lambda v: fmt(v, ".1f")),
            "max trunc": d["pred_max_d_mean_trunc"].map(lambda v: fmt(v, ".1f")),
            "MAE median trunc": d["test_MAE_median_trunc"].map(lambda v: fmt(v, ".3f")),
            "RMSE median trunc": d["test_RMSE_median_trunc"].map(lambda v: fmt(v, ".3f")),
            "train MAE trunc": d["train_MAE_mean_trunc"].map(lambda v: fmt(v, ".3f")),
            "std d trunc": d["mean_std_d_trunc"].map(lambda v: fmt(v, ".2f")),
        })
        lines += [view.to_string(index=False), ""]

    # append to the txt (replacing an earlier truncation section) and to the csv
    txt = Path(args.out_txt)
    old = txt.read_text(encoding="utf-8") if txt.exists() else ""
    cut = old.find("\n" + "=" * 110 + "\n" + TRUNC_HEADER)
    if cut >= 0:
        old = old[:cut]
    txt.write_text(old.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")
    csv = Path(args.out_csv)
    if csv.exists():
        base = pd.read_csv(csv)
        base = base[~base["transform"].astype(str).str.endswith("+trunc")]
        pd.concat([base, table], ignore_index=True).to_csv(csv, index=False)
    else:
        table.to_csv(csv, index=False)
    print(f"\nAppended the truncated results to {txt}")


def main():
    parser = argparse.ArgumentParser(description="Seconds vs log1p(seconds) vs log1p(days) as model space of the time regressor.")
    parser.add_argument("--case_studies", default="BAC,BPI12,bpi17_before,bpi17_after")
    parser.add_argument("--transforms", default="none,log1p_seconds,log1p_days")
    parser.add_argument("--optuna_trials", type=int, default=STEP2.params["optuna_trials"],
                        help=f"Optuna trials per model (default: step 2's, {STEP2.params['optuna_trials']}).")
    parser.add_argument("--data_suffix", default="", help="Read train_data<suffix>.csv / test_data<suffix>.csv.")
    parser.add_argument("--models_dir", default=None,
                        help="Where to save the models (default case_studies/<case>/model/time_target_test); "
                             "a <case>/<transform> subfolder is created.")
    parser.add_argument("--out_txt", default=str(HERE / "2b_time_target_comparison.txt"))
    parser.add_argument("--out_csv", default=str(HERE / "2b_time_target_comparison.csv"))
    parser.add_argument("--truncation_only", action="store_true",
                        help="Do not train: evaluate the saved log-space models with the log-normal truncated at the "
                             "maximum remaining time of the training set, and append the results to the report.")
    args = parser.parse_args()
    if args.truncation_only:
        evaluate_truncation(args)
        return

    rows = []
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        train_data, test_data = load_training_data(cs, args.data_suffix)
        case_id_name, _, _, continuous_features, categorical_features, columns_to_remove = get_features(cs)
        for transform in [t.strip() for t in args.transforms.split(",") if t.strip()]:
            out_dir = (Path(args.models_dir) / cs if args.models_dir else HERE / "case_studies" / cs / "model" / "time_target_test") / transform
            params = {**STEP2.params, "optuna_trials": args.optuna_trials, "time_target_transform": transform,
                      "targets": [TIME_TARGET], "output_dir": str(out_dir)}
            print(f"\n=== {cs} | time target: {transform} | {args.optuna_trials} Optuna trials -> {out_dir}")
            t0 = time.time()
            res = train_ml_model(train_data=train_data, test_data=test_data, case_id_name=case_id_name,
                                 columns_to_remove=columns_to_remove, continuous_features=continuous_features,
                                 categorical_features=categorical_features, case_study=cs, params=params)[TIME_TARGET]
            rows.append(summary_row(cs, transform, res, (time.time() - t0) / 60, len(train_data), len(test_data)))
            write_report(rows, args, args.out_txt, args.out_csv)  # after every model, in case the run stops
    print(f"\nWritten {args.out_txt}")


if __name__ == "__main__":
    main()
