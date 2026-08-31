import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.get_features import get_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.predictive_models_functions import train_ml_model

end_date_name = 'time:timestamp'
start_date_name = 'start:timestamp'

params = {
    "case_study" : "BPI12",
    #"case_study": ["BAC", "BPI12", "bpi17_before", "bpi17_after"],
    "optuna_trials": 80,
    "optuna_timeout": 1200,  # seconds per Optuna study; 
    "early_stopping_rounds": 50,
    
    "search_spaces": {
        "label": {
            "learning_rate": {"type": "float", "min": 0.005, "max": 0.3, "log": True},
            "depth": {"type": "int", "min": 3, "max": 10},
            "l2_leaf_reg": {"type": "float", "min": 1.0, "max": 30.0},
            "colsample_bylevel": {"type": "float", "min": 0.6, "max": 1.0},
            "bootstrap_type": {"type": "categorical", "choices": ["Bayesian", "Bernoulli", "MVS"]},
        },
        # SGLB (posterior_sampling) already injects Gaussian noise into the
        # gradients, so:
        #  - learning_rate floor raised (0.005 -> 0.02): very low rates + SGLB's
        #    model-shrink do not converge within the timeout on 150k rows;
        #  - depth up to 10: a first run pinned depth at the old cap of 8, so
        #    the regressor wants the extra capacity;
        #  - l2_leaf_reg on a log scale, higher ceiling -- the many correlated
        #    one-hot columns need it;
        #  - bootstrap_type pinned to MVS (see const_params); subsample kept
        #    high (0.7-1.0) so SGLB's dynamics and the epistemic estimate stay
        #    clean;
        #  - random_strength dropped: a first run drove it to ~0 (Optuna does
        #    not want extra split noise on top of the Langevin noise);
        #  - grow_policy left open so Optuna can trade the fast, self-regularising
        #    symmetric trees for depthwise trees with a real min_data_in_leaf
        #    floor (a no-op with SymmetricTree).
        "sigmoid_mm": {
            "learning_rate": {"type": "float", "min": 0.02, "max": 0.25, "log": True},
            "depth": {"type": "int", "min": 4, "max": 10},
            "l2_leaf_reg": {"type": "float", "min": 1.0, "max": 40.0, "log": True},
            "colsample_bylevel": {"type": "float", "min": 0.5, "max": 1.0},
            "subsample": {"type": "float", "min": 0.7, "max": 1.0},
            "grow_policy": {"type": "categorical", "choices": ["SymmetricTree", "Depthwise"]},
        },
    },
}

def run_for_case_study(case_study, runtime_params):
    print("\n" + "="*50)
    print(f" >>> LOADING DATASETS FOR {case_study.upper()} <<< ")
    print("="*50)

    data_dir = Path(f"./case_studies/{case_study}")
    case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove = get_features(case_study)

    train_data = pd.read_csv(data_dir / "train_data.csv", parse_dates=[end_date_name, start_date_name])
    test_data = pd.read_csv(data_dir / "test_data.csv", parse_dates=[end_date_name, start_date_name])

    if case_study == "BPI12":
        print("\nApplying BPI12 specific data type conversions...")
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data  = convert_dtypes_bpi12(test_data, "experiment")

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_study": case_study,
        "optuna_trials": runtime_params["optuna_trials"],
        "early_stopping_rounds": runtime_params["early_stopping_rounds"],
        **{f"search_{k}": str(v) for k, v in runtime_params["search_spaces"].items()}
    }
    df_record = pd.DataFrame([record])
    output_dir = Path(f"./case_studies/{case_study}/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_record.to_csv(output_dir / "params.csv", mode='w', header=True, index=False)

    print("\n" + "="*50)
    print(f" >>> TRAINING ML MODEL WITH OPTUNA <<< ")
    print("="*50)

    case_results = train_ml_model(
        train_data=train_data,
        test_data=test_data,
        case_id_name=case_id_name,
        columns_to_remove=columns_to_remove,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        case_study=case_study,
        params=runtime_params,
    )

    print("\n" + "*"*50)
    print(f" TRAINING PROCESS COMPLETE FOR {case_study.upper()}! ")
    print("*"*50 + "\n")

    return case_results


def _format_search_space(search_spaces_config):
    lines = []
    for name, cfg in search_spaces_config.items():
        extra = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "type")
        lines.append(f"  - {name}: type={cfg.get('type')}, {extra}")
    return "\n".join(lines)


def _format_best_params(best_params):
    parts = []
    for k, v in best_params.items():
        parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts)


def write_training_report(all_results, runtime_params, output_path):
    """
    Writes one aggregated txt report with, for every trained case study and
    every target (label/sigmoid_mm): the search space used, the winning
    Optuna trial and its validation score, the best hyperparameters, the
    number of trees in the final model, and the train/test scores -- so
    the whole run can be reviewed without opening any per-model file.

    For the regressor it also reports the predictive-uncertainty diagnostics
    (average aleatoric / epistemic / total std, epistemic share of the
    variance, sharpness, and the empirical coverage of the mean +/- 1 sigma
    and +/- 2 sigma intervals).
    """
    model_labels = {
        "label": "Model 1 (label - Classifier)",
        "sigmoid_mm": "Model 2 (sigmoid_mm - Regressor with uncertainty)",
    }

    lines = []
    lines.append("=" * 70)
    lines.append("HYPERPARAMETER AND METRICS SUMMARY PER DATASET (CATBOOST)")
    lines.append(f"Auto-generated by 2_training_predictive_model.py - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for i, (case_study, case_results) in enumerate(all_results.items(), start=1):
        lines.append(f"{i}. DATASET {case_study.upper()}")
        lines.append("-" * 70)
        lines.append(
            f"* Optuna trials: {runtime_params['optuna_trials']} "
            f"(timeout {runtime_params.get('optuna_timeout', 1200)}s) | "
            f"Early stopping rounds: {runtime_params['early_stopping_rounds']}"
        )
        lines.append("")

        for target_name in ["label", "sigmoid_mm"]:
            res = case_results.get(target_name)
            if res is None:
                continue
            lines.append(f"* Metrics for {model_labels[target_name]}:")
            target_space = runtime_params.get("search_spaces", {}).get(target_name)
            if target_space:
                lines.append("  - Search space:")
                for space_line in _format_search_space(target_space).splitlines():
                    lines.append(f"  {space_line}")
            lines.append(f"  - Best Optuna trial: #{res['best_trial_number']} ({res['metric_name']} validation = {res['best_validation_score']:.5f})")
            lines.append(f"  - Best hyperparameters: {_format_best_params(res['best_params'])}")
            lines.append(f"  - Trees in final model (refit on 100% training data): {res['n_trees_final_model']}")
            lines.append(f"  - {res['metric_name']} score of training set: {res['train_score']:.5f}")
            lines.append(f"  - {res['metric_name']} score of test set: {res['test_score']:.5f}")
            unc = res.get("uncertainty")
            if unc:
                lines.append(f"  - MAE score of test set: {unc['test_mae']:.5f}")
                lines.append(
                    f"  - Predictive uncertainty (test avg): data/aleatoric std = {unc['mean_data_std']:.5f}, "
                    f"knowledge/epistemic std = {unc['mean_knowledge_std']:.5f}, total std = {unc['mean_total_std']:.5f}"
                )
                lines.append(
                    f"  - Epistemic share of predictive variance: {unc['epistemic_var_fraction'] * 100:.1f}% "
                    f"(rest is irreducible noise); median total std (sharpness) = {unc['median_total_std']:.5f}"
                )
                lines.append(
                    f"  - Calibration: sigma recalibration factor s = {unc.get('sigma_scale', 1.0):.3f} "
                    f"(fitted on a held-out slice; does NOT affect the Pareto front, only the interval width)"
                )
                lines.append(
                    f"    mean +/- 1 sigma coverage (target ~68%): raw {unc.get('coverage_1sigma_raw', float('nan')) * 100:.1f}% "
                    f"-> recalibrated {unc['coverage_1sigma'] * 100:.1f}%"
                )
                lines.append(
                    f"    mean +/- 2 sigma coverage (target ~95%): raw {unc.get('coverage_2sigma_raw', float('nan')) * 100:.1f}% "
                    f"-> recalibrated {unc['coverage_2sigma'] * 100:.1f}%"
                )
            lines.append("")

    lines.append("=" * 70)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    runtime_params = params.copy()
    case_studies = runtime_params["case_study"]
    if isinstance(case_studies, str):
        case_studies = [case_studies]

    all_results = {}
    for case_study in case_studies:
        all_results[case_study] = run_for_case_study(case_study, runtime_params)

    report_path = Path("./2_training_results.txt")
    write_training_report(all_results, runtime_params, report_path)
    print(f"\nAggregated report for all datasets saved to: {report_path}")


if __name__ == "__main__":
    main()

# Running commands:
# python 2_training_predictive_model.py
# In params, "case_study" can be a single string (e.g. "BAC") or a list of
# strings (e.g. ["BAC", "BPI12", "bpi17_before", "bpi17_after"]) to train
# all of them one after another with the same optuna/search_spaces settings.