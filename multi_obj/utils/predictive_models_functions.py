import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
from optuna.integration import CatBoostPruningCallback

from utils.train_test_split import extract_internal_running_validation

class ExpM1Regressor:
    """
    Wrapper minimale attorno a un modello gia' addestrato (es. CatBoostRegressor)
    che e' stato allenato a predire log1p(target). Applica np.expm1() in modo
    trasparente in predict(), cosi' che l'output della pipeline sia sempre nella
    scala originale del target, senza dover ricordare di farlo a mano ogni volta
    che si usa il modello in produzione.
    """
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model

    def fit(self, X, y=None):
        # No-op: fitted_model e' gia' stato addestrato a monte.
        # Il metodo e' necessario perche' scikit-learn (>=1.6 circa)
        # richiede che ogni step di una Pipeline esponga 'fit' per
        # essere riconosciuto come estimator valido da check_is_fitted().
        return self

    def __sklearn_is_fitted__(self):
        # Consideriamo l'oggetto "fitted" se il modello interno lo e'.
        return self.fitted_model is not None

    def predict(self, X):
        preds_log_scale = self.fitted_model.predict(X)
        return np.expm1(preds_log_scale)

    def get_params(self, deep=True):
        return {"fitted_model": self.fitted_model}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def prepare_df_for_ml(df, case_id_name, columns_to_remove=None):
    df = df.drop(columns=[case_id_name], errors='ignore')
    
    y1 = df.label
    y2 = df.sigmoid_mm

    if columns_to_remove is not None:
        df = df.drop(columns=columns_to_remove, axis="columns", errors='ignore')

    X = df.drop(columns=["label", "sigmoid_mm", "outcome"], errors='ignore')
    
    return X, y1, y2

def filter_features(features, dataset_columns, feature_type):
    present = [f for f in features if f in dataset_columns]
    missing = [f for f in features if f not in dataset_columns]
    if missing:
        print(f"Warning: the following {feature_type} features are not in training data and will be skipped: {missing}")
    return present

def train_ml_model(train_data, test_data, case_id_name, columns_to_remove,
                   continuous_features, categorical_features, case_study=None, params=None):

    if params is None:
        params = {}

    optuna_trials = params.get("optuna_trials", 80)
    early_stopping_rounds = params.get("early_stopping_rounds", 50)
    search_spaces_config = params.get("search_spaces", {})

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

    for y_train, y_test in [(y_train1, y_test1), (y_train2, y_test2)]:
        print(f"\n--- Optuna Hyperparameter Optimization for: {y_train.name} ---")

        is_regression_target = (y_train.name == "sigmoid_mm")

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
            const_params.update({"loss_function": "Logloss", "eval_metric": "AUC"})
        else:
            const_params.update({"loss_function": "MAE", "eval_metric": "R2"})

        def objective(trial):
            trial_params = const_params.copy()

            for param_name, config in search_spaces_config.items():
                p_type = config.get("type")
                if p_type == "float":
                    trial_params[param_name] = trial.suggest_float(param_name, config["min"], config["max"], log=config.get("log", False))
                elif p_type == "int":
                    trial_params[param_name] = trial.suggest_int(param_name, config["min"], config["max"])
                elif p_type == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            
            if trial_params.get("bootstrap_type") == "Bayesian":
                trial_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
            elif trial_params.get("bootstrap_type") in ["Bernoulli", "MVS"]:
                trial_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            
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

            if is_regression_target:
                y_tr_fit = np.log1p(y_tr)
                y_val_eval = np.log1p(y_val)
            else:
                y_tr_fit = y_tr
                y_val_eval = y_val

            if y_train.name == "label":
                model = CatBoostClassifier(**trial_params)
            else:
                model = CatBoostRegressor(**trial_params)
            pruning_callback = CatBoostPruningCallback(trial, const_params["eval_metric"])
            
            model.fit(
                X_tr, y_tr_fit,
                eval_set=[(X_val, y_val_eval)],
                verbose=0,
                callbacks=[pruning_callback]
            )
            
            pruning_callback.check_pruned()
            
            preds = model.predict(X_val)
            if y_train.name == "label":
                return np.mean(preds == y_val)
            else:
                preds_original_scale = np.expm1(preds)
                return r2_score(y_val, preds_original_scale)

        # Training the model with Optuna hyperparameter optimization
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20), 
            direction="maximize"
        )
        study.optimize(objective, n_trials=optuna_trials, timeout=600)
        print(f"Best trial found for {y_train.name} with score {study.best_value:.5f}")
        
        # Final training of the model with the best hyperparameters found by Optuna
        X_tr_f, X_val_f, y_tr_f, y_val_f = extract_internal_running_validation(
            X_trans=X_train_trans, 
            y_train=y_train, 
            train_data=train_data, 
            case_id_name=case_id_name, 
            train_ratio=0.80
        )

        final_params = const_params.copy()
        final_params.update(study.best_params)
        final_params["logging_level"] = "Verbose"

        if y_train.name == "label":
            final_model = CatBoostClassifier(**final_params)
        else:
            final_model = CatBoostRegressor(**final_params)

        if is_regression_target:
            y_tr_f_fit = np.log1p(y_tr_f)
            y_val_f_eval = np.log1p(y_val_f)
        else:
            y_tr_f_fit = y_tr_f
            y_val_f_eval = y_val_f

        final_model.fit(
            X_tr_f, y_tr_f_fit,
            eval_set=[(X_val_f, y_val_f_eval)],
            verbose=500
        )

        if is_regression_target:
            prediction_step = ExpM1Regressor(final_model)
        else:
            prediction_step = final_model
 
        print("\n[INFO] Training complete. Evaluating performance...")
        y_train_predicted = prediction_step.predict(X_train_trans)
        y_test_predicted = prediction_step.predict(X_test_trans)
 
        if y_train.name == "label":
            print("Accuracy score of training set:", np.mean(y_train == y_train_predicted))
            print("Accuracy score of test set:", np.mean(y_test == y_test_predicted))
        else:
            print("R2 score of training set:", r2_score(y_train, y_train_predicted))
            print("R2 score of test set:", r2_score(y_test, y_test_predicted))
        print("--------------------------------------------------")

        best_pipeline = Pipeline(steps=[
            ("transformation", transformations),
            ("prediction", prediction_step)
        ])

        output_dir = Path(f"./case_studies/{case_study}/model")
        joblib.dump(best_pipeline, output_dir / f"catboost_model_{y_train.name}.joblib")
        
        with open(output_dir / f"best_hyperparams_{y_train.name}.json", 'w') as f:
            json.dump(study.best_params, f, indent=4)