from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import catboost
from catboost import CatBoostRegressor
import argparse

SECONDS_TO_HOURS = 1/(60 * 60)
SECONDS_TO_DAYS = 1/(60 * 60 * 24)
RANDOM_SEED = 42

def prepare_df_for_ml(df, case_id_name, outcome_name, columns_to_remove=None):
    """
    Prepares a DataFrame for machine learning by removing specified columns and separating features and target variable.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        case_id_name (str): The name of the column to be removed that contains case IDs.
        outcome_name (str): The name of the column that contains the target variable.
        columns_to_remove (list of str, optional): A list of additional column names to be removed. Defaults to None.
    
    Returns:
        tuple: A tuple containing two elements:
            - X (pandas.DataFrame): The DataFrame containing the features.
            - y (pandas.Series): The Series containing the target variable.
    """
    # Before training for ml we need to remove columns that can are not needed for ML model.
    df = df.drop(columns=[case_id_name], errors='ignore')
    if columns_to_remove is not None:
        # Adding errors='ignore' prevents a KeyError if case_id_name is in this list
        df = df.drop(columns=columns_to_remove, axis="columns", errors='ignore')

    X = df.drop([outcome_name], axis=1)
    y = df[outcome_name]
    return X, y

def train_ml_model(train_data, test_data, case_id_name, outcome_name, columns_to_remove,
                   continuous_features, categorical_features, learning_rate, depth,
                   n_iterations, case_study, n_splits=5, use_native_categorical=False,
                   early_stopping_rounds=5):
    """
    Trains a CatBoostRegressor for predicting combined outcome, using K-fold CV
    to estimate generalization performance, then a final model fit on all of
    train_data and evaluated once on the held-out test_data.
    """

    X_train, y_train = prepare_df_for_ml(train_data, case_id_name, outcome_name, columns_to_remove)
    X_test, y_test = prepare_df_for_ml(test_data, case_id_name, outcome_name, columns_to_remove)

    params = {"task_type": "CPU", "learning_rate": learning_rate,
          "early_stopping_rounds": early_stopping_rounds,
          "logging_level": "Silent", "iterations": n_iterations,
          "depth": depth, "loss_function": "MAE", "eval_metric": "R2",
          "random_seed": RANDOM_SEED}

    def fit_fold(X_tr, y_tr, X_val, y_val):
        """Fits one fold's transformer + CatBoost with a real eval_set, returns a fitted pipeline-like object."""
        if use_native_categorical:
            model = CatBoostRegressor(cat_features=categorical_features, **params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            return model
        else:
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            transformations = ColumnTransformer(transformers=[
                ('num', numeric_transformer, continuous_features),
                ('cat', categorical_transformer, categorical_features)])

            X_tr_t = transformations.fit_transform(X_tr, y_tr)
            X_val_t = transformations.transform(X_val)

            model = CatBoostRegressor(**params)
            model.fit(X_tr_t, y_tr, eval_set=(X_val_t, y_val))

            return Pipeline(steps=[("transformation", transformations), ("prediction", model)])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    cv_r2_scores = []
    cv_mae_scores = []

    # Reset indices to ensure safe integer location mapping during KFold slicing
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_reset, y_train_reset), 1):
        X_tr, y_tr = X_train_reset.iloc[train_idx], y_train_reset.iloc[train_idx]
        X_val, y_val = X_train_reset.iloc[val_idx], y_train_reset.iloc[val_idx]

        fold_model = fit_fold(X_tr, y_tr, X_val, y_val)

        val_preds = fold_model.predict(X_val)
        fold_r2 = r2_score(y_val, val_preds)
        fold_mae = mean_absolute_error(y_val, val_preds)

        cv_r2_scores.append(fold_r2)
        cv_mae_scores.append(fold_mae)
        print(f"Fold {fold} -> Validation R2: {fold_r2:.4f} | Validation MAE: {fold_mae:.4f}")

    print("\nCross-Validation Summary:")
    print(f"Mean CV R2 Score: {np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores):.4f})")
    print(f"Mean CV MAE: {np.mean(cv_mae_scores):.4f} (+/- {np.std(cv_mae_scores):.4f})")

    # Final model fit on all of train_data, evaluated once on the held-out test_data
    print("\nFitting final model on all training data and evaluating on test data...")
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED)

    catboost_pipeline = fit_fold(X_tr_final, y_tr_final, X_val_final, y_val_final)

    print("\nTraining results (fit on the held-in portion used for the final model):")
    y_train_predicted = catboost_pipeline.predict(X_train)
    print("R2 score of training set:", r2_score(y_train, y_train_predicted))
    mae_train = mean_absolute_error(y_train, y_train_predicted)
    print('Mean Absolute Error: {}'.format(mae_train))

    print("Testing results:")
    y_test_predicted = catboost_pipeline.predict(X_test)
    print("R2 score of test set:", r2_score(y_test, y_test_predicted))
    mae = mean_absolute_error(y_test, y_test_predicted)
    print('Mean Absolute Error: {}'.format(mae))

    # Save the trained model to a file
    output_dir = Path(f"./case_studies/{case_study}/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "catboost_model.joblib"
    joblib.dump(catboost_pipeline, model_path)

    return catboost_pipeline