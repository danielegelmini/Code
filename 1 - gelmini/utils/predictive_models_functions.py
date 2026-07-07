import pandas as pd
import numpy as np
import joblib
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
    df = df.drop([case_id_name], axis="columns")
    if columns_to_remove != None:
        df = df.drop(columns_to_remove, axis="columns")
    X = df.drop([outcome_name], axis=1)
    y = df[outcome_name]
    return X, y

def train_ml_model(train_data, test_data, case_id_name, outcome_name, columns_to_remove, 
                   continuous_features, categorical_features, learning_rate, depth, 
                   n_iterations, case_study):
    """
    Trains a CatBoostRegressor for predicting combined outcome.
    """

    X_train, y_train = prepare_df_for_ml(train_data, case_id_name, outcome_name, columns_to_remove)
    X_test, y_test = prepare_df_for_ml(test_data, case_id_name, outcome_name, columns_to_remove)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])
    
    params = {"task_type": "CPU", "learning_rate": learning_rate, "early_stopping_rounds": 5,
          "logging_level": "Silent", "iterations": n_iterations, 
          "depth": depth, "loss_function": "MAE", "eval_metric": "R2", "l2_leaf_reg": 30}
    
    catboost_pipeline = Pipeline(steps=[
    ("transformation", transformations),
    ("prediction", CatBoostRegressor(**params))
    ])

    catboost_pipeline.fit(X_train, y_train)

    print("Training results:")
    y_train_predicted = catboost_pipeline.predict(X_train)
    print("R2 score of training set:", r2_score(y_train, y_train_predicted))
    mae_train = mean_absolute_error(y_train, y_train_predicted) 
    print('Mean Absolute Error: {}'.format(mae_train))

    print("Testing results:")
    y_test_predicted = catboost_pipeline.predict(X_test)
    print("R2 score of test set:", r2_score(y_test, y_test_predicted)) 
    mae = mean_absolute_error(y_test, y_test_predicted) 
    print('Mean Absolute Error: {}'.format(mae))

    joblib.dump(catboost_pipeline, f'./case_studies/{case_study}/model/catboost_model.joblib') # Save models
