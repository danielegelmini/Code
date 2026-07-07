from pathlib import Path
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import hyperopt

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
    df = df.drop(columns=[case_id_name], errors='ignore')

    y1, y2 = df.label, df.sigmoid_mm

    if columns_to_remove is not None:
        df = df.drop(columns=columns_to_remove, axis="columns", errors='ignore')

    X = df.drop([outcome_name], axis=1)
    return X, y1, y2


def evaluate_model(pipeline, X, y):
    # validation set of 15% randomly selected from the training set
    train_size_percentage = 0.85
    train_idx = np.random.choice(X.index, size=int(train_size_percentage * len(X)), replace=False)
    val_idx = X.index.difference(train_idx)

    scores = []

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    y_val_pred = pipeline.predict(X_val)

    if y.name == "label":
        scores.append(np.mean(y_val_pred == y_val))
    else:
        scores.append(r2_score(y_val, y_val_pred))
    return np.mean(scores)


class CatBoostModelObjective(object):
    def __init__(self, X, y, const_params, transformations):
        self._X = X
        self._y = y
        self._const_params = const_params.copy()
        self._transformations = transformations
        self._evaluated_count = 0
        
    def _to_catboost_params(self, hyper_params):
        params = {}
        if 'learning_rate' in hyper_params:
            params['learning_rate'] = hyper_params['learning_rate']
        if 'depth' in hyper_params:
            params['depth'] = int(hyper_params['depth'])
        if 'iterations' in hyper_params:
            params['iterations'] = int(hyper_params['iterations'])
        return params
    
    def __call__(self, hyper_params):
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)
        
        if self._y.name == "label":
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)
            
        pipeline = Pipeline(steps=[
            ("transformation", self._transformations),
            ("prediction", model)
        ])
        
        score = evaluate_model(pipeline, self._X, self._y)
        
        self._evaluated_count += 1
        print(f'evaluating params={params}', file=sys.stdout)
        print(f'evaluated {self._evaluated_count} times, cv_score={score:.5f}', file=sys.stdout)
        sys.stdout.flush()
        
        return {'loss': -score, 'status': hyperopt.STATUS_OK}


def train_ml_model(train_data, test_data, case_id_name, outcome_name, columns_to_remove, 
                   continuous_features, categorical_features, learning_rate, depth, 
                   n_iterations, case_study, learning_rate_min=None, learning_rate_max=None,
                   depth_min=None, depth_max=None, n_iterations_min=None, n_iterations_max=None,
                   random_search_trials=10, cv_folds=5):
    """
    Trains a CatBoostRegressor and CatBoostClassifier using optional randomized cross validation search via Hyperopt.
    """

    X_train, y_train1, y_train2 = prepare_df_for_ml(train_data, case_id_name, outcome_name, columns_to_remove)
    X_test, y_test1, y_test2 = prepare_df_for_ml(test_data, case_id_name, outcome_name, columns_to_remove)

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])

    for y_train, y_test in [(y_train1, y_test1), (y_train2, y_test2)]:
        print(f"Training for outcome: {y_train.name}")
        
        const_params = {
            "task_type": "CPU",
            "devices": "0",
            "early_stopping_rounds": 5,
            "logging_level": "Silent",
            "l2_leaf_reg": 30,
        }
        
        if y_train.name == "label":
            const_params.update({"loss_function": "Logloss", "eval_metric": "AUC"})
        else:
            const_params.update({"loss_function": "MAE", "eval_metric": "R2"})

        if (learning_rate_min is not None and learning_rate_max is not None) or \
           (depth_min is not None and depth_max is not None) or \
           (n_iterations_min is not None and n_iterations_max is not None):

            parameter_space = {}

            if learning_rate_min is not None and learning_rate_max is not None:
                parameter_space['learning_rate'] = hyperopt.hp.uniform('learning_rate', learning_rate_min, learning_rate_max)
            else:
                const_params['learning_rate'] = learning_rate
                
            if depth_min is not None and depth_max is not None:
                parameter_space['depth'] = hyperopt.hp.quniform('depth', depth_min, depth_max, 1)
            else:
                const_params['depth'] = depth
                
            if n_iterations_min is not None and n_iterations_max is not None:
                parameter_space['iterations'] = hyperopt.hp.quniform('iterations', n_iterations_min, n_iterations_max, 1)
            else:
                const_params['iterations'] = n_iterations

            objective = CatBoostModelObjective(
                X=X_train, 
                y=y_train, 
                const_params=const_params, 
                transformations=transformations
            )
            
            trials = hyperopt.Trials()
            
            best_hyper_params = hyperopt.fmin(
                fn=objective,
                space=parameter_space,
                algo=hyperopt.tpe.suggest,
                max_evals=random_search_trials,
                #rstate=np.random.RandomState(seed=42),
                trials=trials
            )

            print(f"Best random search params for {y_train.name}: {best_hyper_params}")
            
            final_params = const_params.copy()
            if 'learning_rate' in best_hyper_params:
                final_params['learning_rate'] = best_hyper_params['learning_rate']
            if 'depth' in best_hyper_params:
                final_params['depth'] = int(best_hyper_params['depth'])
            if 'iterations' in best_hyper_params:
                final_params['iterations'] = int(best_hyper_params['iterations'])
                
        else:
            final_params = const_params.copy()
            final_params['learning_rate'] = learning_rate
            final_params['depth'] = depth
            final_params['iterations'] = n_iterations

        if y_train.name == "label":
            final_model = CatBoostClassifier(**final_params)
        else:
            final_model = CatBoostRegressor(**final_params)
            
        best_pipeline = Pipeline(steps=[
            ("transformation", transformations),
            ("prediction", final_model)
        ])

        best_pipeline.fit(X_train, y_train)

        print("Training results:")
        y_train_predicted = best_pipeline.predict(X_train)
        if y_train.name == "label":
            print("Accuracy score of training set:", np.mean(y_train == y_train_predicted))
        else:
            print("R2 score of training set:", r2_score(y_train, y_train_predicted))
            
        mae_train = mean_absolute_error(y_train, y_train_predicted)
        print('Mean Absolute Error: {}'.format(mae_train))

        print("Testing results:")
        y_test_predicted = best_pipeline.predict(X_test)
        if y_test.name == "label":
            print("Accuracy score of test set:", np.mean(y_test == y_test_predicted))
        else:
            print("R2 score of test set:", r2_score(y_test, y_test_predicted))
            
        mae = mean_absolute_error(y_test, y_test_predicted)
        print('Mean Absolute Error: {}'.format(mae))
        print("--------------------------------------------------")

        output_dir = Path(f"./case_studies/{case_study}/model")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"catboost_model_{y_train.name}.joblib"
        joblib.dump(best_pipeline, model_path)