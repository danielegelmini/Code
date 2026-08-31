import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
from optuna.integration import CatBoostPruningCallback

from utils.train_test_split import extract_internal_running_validation

DEFAULT_VIRTUAL_ENSEMBLES_COUNT = 10


class UncertaintyRegressor:
    """
    Wraps a CatBoost regressor trained with loss_function='RMSEWithUncertainty'
    (and posterior_sampling=True) so it can sit as the final step of an sklearn
    Pipeline while still exposing CatBoost's uncertainty estimates.

    - predict(X) returns only the mean prediction (column 0 of CatBoost's
      2-column RMSEWithUncertainty output), so every existing caller that
      expects a plain 1-D array of predicted 'sigmoid_mm' values keeps working
      unchanged.
    - predict_uncertainty(X) additionally runs a virtual ensemble to split the
      predictive variance into its data (aleatoric) and knowledge (epistemic)
      parts, and returns both plus the total standard deviation.

    `sigma_scale` is a single post-hoc recalibration factor (>1 inflates,
    <1 shrinks). CatBoost's RMSEWithUncertainty tends to be over-confident --
    fewer than 68% of test targets land inside mean +/- 1 sigma -- so every std
    returned by predict_uncertainty is multiplied by this factor, fitted on a
    held-out slice at training time (see train_ml_model). It rescales all three
    components by the same amount, so their ratio (and any ranking of cases by
    uncertainty) is unchanged; only the absolute interval width moves.

    The target is NOT transformed here: the previous ExpM1Regressor undid a
    log1p transform of the target, but 'sigmoid_mm' is already a bounded [0, 1]
    target and an ablation showed the transform did not help (and slightly hurt
    the uncertainty model), so it was dropped.
    """

    def __init__(self, fitted_model, virtual_ensembles_count=DEFAULT_VIRTUAL_ENSEMBLES_COUNT,
                 sigma_scale=1.0):
        self.fitted_model = fitted_model
        self.virtual_ensembles_count = virtual_ensembles_count
        self.sigma_scale = float(sigma_scale)

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return self.fitted_model is not None

    @staticmethod
    def _mean_only(preds):
        preds = np.asarray(preds)
        return preds[:, 0] if preds.ndim == 2 else preds

    def predict(self, X):
        return self._mean_only(self.fitted_model.predict(X))

    def predict_uncertainty(self, X):
        """
        Returns a dict of 1-D numpy arrays:
          mean           - point prediction (same as predict())
          data_std       - aleatoric uncertainty  = sqrt(data variance)
          knowledge_std  - epistemic uncertainty  = sqrt(knowledge variance)
          total_std      - sqrt(data variance + knowledge variance)

        data variance is the noise the model expects for that input even with
        infinite training data; knowledge variance is the disagreement between
        the virtual sub-ensembles, i.e. how little the model has seen inputs
        like this one.
        """
        n_trees = self.fitted_model.tree_count_ or 0
        ve_count = int(max(1, min(self.virtual_ensembles_count, n_trees)))
        out = np.asarray(
            self.fitted_model.virtual_ensembles_predict(
                X,
                prediction_type="TotalUncertainty",
                virtual_ensembles_count=ve_count,
            ),
            dtype=float,
        )
        mean = out[:, 0]
        knowledge_var = np.clip(out[:, 1], 0.0, None)
        data_var = np.clip(out[:, 2], 0.0, None)
        s = self.sigma_scale
        return {
            "mean": mean,
            "data_std": s * np.sqrt(data_var),
            "knowledge_std": s * np.sqrt(knowledge_var),
            "total_std": s * np.sqrt(data_var + knowledge_var),
        }

    def get_params(self, deep=True):
        return {
            "fitted_model": self.fitted_model,
            "virtual_ensembles_count": self.virtual_ensembles_count,
            "sigma_scale": self.sigma_scale,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def prepare_df_for_ml(df, case_id_name, columns_to_remove=None):
    """
    Prepares the dataframe for machine learning by separating features and targets.

    This function extracts the targets 'label' and 'sigmoid_mm', drops the specified case ID column, 
    and optionally removes other specified columns.

    Args:
        df (pandas.DataFrame): The input dataframe to process.
        case_id_name (str): The name of the column containing the case identifier to drop.
        columns_to_remove (list of str, optional): A list of additional column names to drop. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The feature matrix (X).
            - pandas.Series: The 'label' target variable (y1).
            - pandas.Series: The 'sigmoid_mm' target variable (y2).
    """
    df = df.drop(columns=[case_id_name], errors='ignore')
    
    y1 = df.label
    y2 = df.sigmoid_mm

    if columns_to_remove is not None:
        df = df.drop(columns=columns_to_remove, axis="columns", errors='ignore')

    X = df.drop(columns=["label", "sigmoid_mm", "outcome"], errors='ignore')
    
    return X, y1, y2

def filter_features(features, dataset_columns, feature_type):
    """
    Filters a list of features to keep only those present in the dataset columns.

    It also prints a warning for any features that are missing from the dataset.

    Args:
        features (list of str): The list of feature names to check.
        dataset_columns (list or pandas.Index): The available columns in the dataset.
        feature_type (str): A descriptive string of the feature type (e.g., 'continuous', 'categorical') used for the warning message.

    Returns:
        list of str: A list of feature names that are present in the dataset columns.
    """
    present = [f for f in features if f in dataset_columns]
    missing = [f for f in features if f not in dataset_columns]
    if missing:
        print(f"Warning: the following {feature_type} features are not in training data and will be skipped: {missing}")
    return present

def train_ml_model(train_data, test_data, case_id_name, columns_to_remove,
                   continuous_features, categorical_features, case_study=None, params=None):
    """
    Trains and optimizes machine learning models using CatBoost and Optuna.

    This function processes the data, sets up a scikit-learn ColumnTransformer pipeline for continuous 
    and categorical features, and runs hyperparameter optimization via Optuna for both classification 
    ('label') and regression ('sigmoid_mm') targets. It then trains final models, evaluates their 
    performance, and serializes the resulting pipelines and best parameters to disk.

    Args:
        train_data (pandas.DataFrame): The training dataset.
        test_data (pandas.DataFrame): The test dataset.
        case_id_name (str): The name of the case ID column to drop.
        columns_to_remove (list of str): Columns to explicitly remove from the feature set.
        continuous_features (list of str): A list of continuous feature names.
        categorical_features (list of str): A list of categorical feature names.
        case_study (str, optional): The name of the case study, used to define the output directory path. Defaults to None.
        params (dict, optional): A dictionary of configuration parameters including 'optuna_trials', 'early_stopping_rounds', and 'search_spaces'. Defaults to None.

    Returns:
        dict: {"label": {...}, "sigmoid_mm": {...}}, one entry per target with
        the metric name used (Logloss/RMSE), the winning Optuna trial number
        and validation score, the best hyperparameters, the number of trees
        used for the final refit, and the train/test scores of the final
        model -- everything needed to write a training report without having
        to re-parse any file.
    """
    if params is None:
        params = {}

    optuna_trials = params.get("optuna_trials", 80)
    optuna_timeout = params.get("optuna_timeout", 1200)
    early_stopping_rounds = params.get("early_stopping_rounds", 50)
    # search_spaces is now one sub-dict per target: {"label": {...}, "sigmoid_mm": {...}}.
    # Fall back to treating a flat dict as "same space for both" for backward compatibility.
    all_search_spaces = params.get("search_spaces", {})
    if all_search_spaces and not any(k in ("label", "sigmoid_mm") for k in all_search_spaces):
        all_search_spaces = {"label": all_search_spaces, "sigmoid_mm": all_search_spaces}

    X_train_raw, y_train1, y_train2 = prepare_df_for_ml(train_data, case_id_name,  columns_to_remove)
    X_test_raw,  y_test1,  y_test2 = prepare_df_for_ml(test_data, 
    case_id_name,  columns_to_remove)

    continuous_features = filter_features(continuous_features, X_train_raw.columns, "continuous")
    categorical_features = filter_features(categorical_features, X_train_raw.columns, "categorical")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    print("Pre-processing features...")
    X_train_trans = transformations.fit_transform(X_train_raw)
    X_test_trans = transformations.transform(X_test_raw)

    results = {}

    for y_train, y_test in [(y_train1, y_test1), (y_train2, y_test2)]:
        print(f"\n--- Optuna Hyperparameter Optimization for: {y_train.name} ---")

        is_regression_target = (y_train.name == "sigmoid_mm")
        search_spaces_config = all_search_spaces.get(y_train.name, {})

        if y_train.name == "label":
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            calculated_balance = float(n_neg / n_pos) if n_pos > 0 else 1.0
        else:
            calculated_balance = 1.0

        const_params = {
            "task_type": "CPU",
            "iterations": 3000,
            "early_stopping_rounds": early_stopping_rounds,
            "logging_level": "Silent",
            "allow_writing_files": False,
        }
        
        if y_train.name == "label":
            const_params.update({"loss_function": "Logloss", "eval_metric": "Logloss"})
        else:
            # Probabilistic regression: the model predicts a mean and a
            # variance, and posterior_sampling (Langevin boosting) lets a
            # virtual ensemble later split that variance into data (aleatoric)
            # and knowledge (epistemic) uncertainty. Early stopping / trial
            # selection still run on plain RMSE of the mean.
            # bootstrap_type is pinned to MVS (minimum-variance importance
            # sampling): Bernoulli's uniform row dropping would add a second
            # stochastic source on top of SGLB's Langevin noise and distort the
            # epistemic estimate. Its companion `subsample` is tuned via the
            # search space instead.
            const_params.update({
                "loss_function": "RMSEWithUncertainty",
                "eval_metric": "RMSE",
                "posterior_sampling": True,
                "bootstrap_type": "MVS",
            })
        
        def objective(trial):
            """
            Objective function for Optuna hyperparameter optimization.
            This function samples hyperparameters from the defined search spaces, configures and 
            trains a CatBoost model (Classifier or Regressor depending on the target), and evaluates 
            its performance on an internal validation set. 

            Args:
                trial (optuna.trial.Trial): An Optuna trial object used to sample hyperparameters.

            Returns:
                float: The eval_metric score (Logloss for classification, RMSE for regression) at the best iteration, which Optuna will attempt to minimize.
            """
            trial_params = const_params.copy()

            for param_name, config in search_spaces_config.items():
                p_type = config.get("type")
                if p_type == "float":
                    trial_params[param_name] = trial.suggest_float(param_name, config["min"], config["max"], log=config.get("log", False))
                elif p_type == "int":
                    trial_params[param_name] = trial.suggest_int(param_name, config["min"], config["max"])
                elif p_type == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            
            # bootstrap_type may come from the search space or be pinned in
            # const_params; only auto-attach its companion parameter when the
            # search space did not already declare it.
            bootstrap_type = trial_params.get("bootstrap_type")
            if bootstrap_type == "Bayesian" and "bagging_temperature" not in search_spaces_config:
                trial_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
            elif bootstrap_type in ("Bernoulli", "MVS") and "subsample" not in search_spaces_config:
                trial_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            # min_data_in_leaf is ignored by the default SymmetricTree growth,
            # so it is only worth sampling when a non-symmetric policy was picked.
            if (
                trial_params.get("grow_policy") in ("Depthwise", "Lossguide")
                and "min_data_in_leaf" not in search_spaces_config
            ):
                trial_params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 200, log=True)

            if y_train.name == "label":
                scale_low = max(0.1, calculated_balance * 0.5)
                scale_high = max(scale_low, calculated_balance * 1.5)
                trial_params["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", scale_low, scale_high
                )
            
            X_tr, X_val, y_tr, y_val = extract_internal_running_validation(
                X_trans=X_train_trans, 
                y_train=y_train, 
                train_data=train_data, 
                case_id_name=case_id_name, 
                train_ratio=0.8
            )

            if y_train.name == "label":
                model = CatBoostClassifier(**trial_params)
            else:
                model = CatBoostRegressor(**trial_params)
            pruning_callback = CatBoostPruningCallback(trial, const_params["eval_metric"])

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=0,
                callbacks=[pruning_callback]
            )
            
            pruning_callback.check_pruned()

            # Save the number of trees early stopping picked for this trial,
            # so the winning trial's tree count can be reused for the final
            # refit on 100% of the training data (see below).
            trial.set_user_attr("best_iteration", int(model.get_best_iteration()))

            if is_regression_target:
                # Post-hoc uncertainty recalibration factor, fitted on this
                # trial's held-out validation slice (clean: the model was not
                # trained on it). RMSEWithUncertainty predict() returns
                # [mean, variance]; we scale sigma by s so that the median
                # standardised absolute residual matches a standard normal's
                # (0.6745), i.e. roughly ~68% of |y - mean| land within
                # s * sigma. The median form is used (not sqrt(mean(r^2/var)))
                # because CatBoost occasionally predicts a near-zero variance
                # and a moment estimator explodes on those rows. Uses the plain
                # data-only variance -- fine, the epistemic part is negligible.
                # NB: the validation slice is the most recent training cases,
                # not the "running cases" test set, and the final model is
                # refit on 100% of the data, so s transfers only approximately;
                # the report shows raw vs recalibrated coverage so the gap is
                # visible. For guaranteed coverage, split-conformal on a
                # dedicated holdout would be the rigorous alternative.
                val_pred = np.asarray(model.predict(X_val), dtype=float)
                resid = y_val.to_numpy(dtype=float) - val_pred[:, 0]
                sigma = np.sqrt(np.clip(val_pred[:, 1], 1e-9, None))
                s = float(np.median(np.abs(resid) / sigma) / 0.674489)
                trial.set_user_attr("sigma_scale", float(np.clip(s, 0.5, 5.0)))

            # Same metric as eval_metric (Logloss or RMSE), read directly from
            # CatBoost's best validation score -- keeps early stopping, pruning
            # and trial selection aligned on one metric.
            return model.get_best_score()["validation"][const_params["eval_metric"]]

        # Training the model with Optuna hyperparameter optimization
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            direction="minimize"
        )
        study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout)
        print(f"Best trial found for {y_train.name} with score {study.best_value:.5f}")

        # Final refit with the best hyperparameters, on 100% of the training
        # data: the internal train/validation split was only needed to pick
        # hyperparameters and the number of trees (early stopping). Both are
        # now fixed, so holding back part of the training data for this fit
        # would only throw away signal for no benefit -- the untouched test
        # set below still gives an honest generalization estimate.
        best_iteration = study.best_trial.user_attrs["best_iteration"]

        final_params = const_params.copy()
        final_params.update(study.best_params)
        final_params["iterations"] = best_iteration + 1
        final_params.pop("early_stopping_rounds", None)
        final_params["logging_level"] = "Verbose"

        if y_train.name == "label":
            final_model = CatBoostClassifier(**final_params)
        else:
            final_model = CatBoostRegressor(**final_params)

        final_model.fit(X_train_trans, y_train, verbose=500)

        if is_regression_target:
            # Reuse the winning trial's uncertainty recalibration factor. It was
            # fitted on that trial's clean 20% holdout; after this 100% refit
            # there is no untouched slice left to refit it on, and s captures a
            # systematic miscalibration of the loss (not something 20% more data
            # would move), so carrying it over is the pragmatic choice.
            sigma_scale = float(study.best_trial.user_attrs.get("sigma_scale", 1.0))
            prediction_step = UncertaintyRegressor(final_model, sigma_scale=sigma_scale)
        else:
            prediction_step = final_model
 
        print("\n[INFO] Training complete. Evaluating performance...")

        uncertainty_report = None
        if y_train.name == "label":
            metric_name = "Logloss"
            y_train_proba = prediction_step.predict_proba(X_train_trans)[:, 1]
            y_test_proba = prediction_step.predict_proba(X_test_trans)[:, 1]
            train_score = log_loss(y_train, y_train_proba)
            test_score = log_loss(y_test, y_test_proba)
            print("Logloss score of training set:", train_score)
            print("Logloss score of test set:", test_score)
        else:
            # Native 'sigmoid_mm' scale now (no log1p transform). Trial
            # selection tracked RMSE, so that stays the headline metric.
            metric_name = "RMSE"
            y_train_pred = prediction_step.predict(X_train_trans)
            y_test_pred = prediction_step.predict(X_test_trans)
            train_score = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            test_score = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            train_mae = float(mean_absolute_error(y_train, y_train_pred))
            test_mae = float(mean_absolute_error(y_test, y_test_pred))
            print(f"RMSE / MAE of training set: {train_score:.5f} / {train_mae:.5f}")
            print(f"RMSE / MAE of test set:     {test_score:.5f} / {test_mae:.5f}")

            # Uncertainty diagnostics on the test set. predict_uncertainty
            # already returns the recalibrated stds (multiplied by sigma_scale);
            # dividing by it recovers CatBoost's raw output for the "before"
            # coverage numbers.
            unc = prediction_step.predict_uncertainty(X_test_trans)
            abs_err = np.abs(y_test.to_numpy(dtype=float) - unc["mean"])
            total_std_cal = unc["total_std"]
            total_std_raw = total_std_cal / sigma_scale
            mean_total_var = float(np.mean(total_std_cal ** 2))
            uncertainty_report = {
                "sigma_scale": sigma_scale,
                "mean_data_std": float(np.mean(unc["data_std"])),
                "mean_knowledge_std": float(np.mean(unc["knowledge_std"])),
                "mean_total_std": float(np.mean(total_std_cal)),
                # share of the average predictive variance that is epistemic
                # (reducible with more/other training data) vs aleatoric.
                "epistemic_var_fraction": (
                    float(np.mean(unc["knowledge_std"] ** 2) / mean_total_var) if mean_total_var > 0 else 0.0
                ),
                # coverage with CatBoost's raw sigma vs the recalibrated sigma
                # (target ~0.68 / ~0.95 for a well-calibrated Gaussian).
                "coverage_1sigma_raw": float(np.mean(abs_err <= total_std_raw)),
                "coverage_2sigma_raw": float(np.mean(abs_err <= 2.0 * total_std_raw)),
                "coverage_1sigma": float(np.mean(abs_err <= total_std_cal)),
                "coverage_2sigma": float(np.mean(abs_err <= 2.0 * total_std_cal)),
                # median predicted total std -- "sharpness", how tight the
                # intervals are regardless of whether they are calibrated.
                "median_total_std": float(np.median(total_std_cal)),
                "test_mae": test_mae,
            }
            print(
                f"Uncertainty (test avg): data/aleatoric std = {uncertainty_report['mean_data_std']:.5f}, "
                f"knowledge/epistemic std = {uncertainty_report['mean_knowledge_std']:.5f}, "
                f"total std = {uncertainty_report['mean_total_std']:.5f} "
                f"(epistemic share {uncertainty_report['epistemic_var_fraction'] * 100:.1f}%)"
            )
            print(
                f"Calibration: sigma_scale = {sigma_scale:.3f} | "
                f"coverage +/-1s {uncertainty_report['coverage_1sigma_raw']:.3f} -> {uncertainty_report['coverage_1sigma']:.3f}, "
                f"+/-2s {uncertainty_report['coverage_2sigma_raw']:.3f} -> {uncertainty_report['coverage_2sigma']:.3f}"
            )
        print("--------------------------------------------------")

        best_pipeline = Pipeline(steps=[
            ("transformation", transformations),
            ("prediction", prediction_step)
        ])

        output_dir = Path(f"./case_studies/{case_study}/model")
        joblib.dump(best_pipeline, output_dir / f"catboost_model_{y_train.name}.joblib")

        with open(output_dir / f"best_hyperparams_{y_train.name}.json", 'w') as f:
            json.dump(study.best_params, f, indent=4)

        results[y_train.name] = {
            "metric_name": metric_name,
            "best_trial_number": study.best_trial.number,
            "best_validation_score": study.best_value,
            "best_params": study.best_params,
            "n_trees_final_model": final_params["iterations"],
            "train_score": float(train_score),
            "test_score": float(test_score),
            "uncertainty": uncertainty_report,
        }

    return results