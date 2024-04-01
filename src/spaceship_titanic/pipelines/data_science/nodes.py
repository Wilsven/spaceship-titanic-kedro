"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.3
"""

import logging
import os

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from spaceship_titanic.pipelines.data_science.config import get_tuning_grid
from spaceship_titanic.pipelines.data_science.utils import (
    compare_models,
    get_train_data,
)

logger = logging.getLogger(__name__)


def split_data(
    modeling_data: pd.DataFrame, tuning_params: dict, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into features and targets training and test sets.

    Args:
        modeling_data (pd.DataFrame): Data containing features and target.
        tuning_params (dict): A dictionary of tuning parameters.
        params (dict): A dictionary of parameters.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            The split data containing features and targets training and test sets.
    """
    X, y = get_train_data(modeling_data, params)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tuning_params["test_size"],
        random_state=tuning_params["random_state"],
    )
    mlflow.log_param("test_size", tuning_params["test_size"])

    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def tune_candidate_models(
    X_train: pd.DataFrame, y_train: pd.DataFrame, tuning_params: dict
) -> pd.DataFrame:
    """Tune different models and get the best from the test data.

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.DataFrame): The training labels.
        tuning_params (dict): A dictionary of tuning parameters.

    Returns:
        pd.DataFrame: An empty `pandas.DataFrame`.
    """
    # Defining the pipeline preprocessing steps
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    mlflow.log_param("numeric_features", numeric_features)
    mlflow.log_param("categorical_features", categorical_features)
    logger.info(f"Total Features: {len(numeric_features + categorical_features)}")

    for grid_name in tuning_params["grid_names"]:
        logger.info(f"Tuning {grid_name} model")
        tuning_grid = get_tuning_grid(grid_name, numeric_features, categorical_features)

        param_grid = tuning_grid["param_grid"]
        pipeline = tuning_grid["pipeline"]

        # Perform random search to find the best hyperparameters
        tuning = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=tuning_params["n_iters"],
            cv=5,
            scoring=tuning_params["scoring"],
            verbose=0,
        )

        tuning.fit(X_train, y_train.values.ravel())

        mlflow.log_params({grid_name: tuning.best_params_})

        # Save the best model found by grid search
        model = tuning.best_estimator_
        best_score = round(tuning.best_score_, 3)
        mlflow.sklearn.log_model(model, grid_name)
        mlflow.log_metric(f"{grid_name}_tuning_score", best_score)
        logger.info(f"{grid_name}_tuning_score: {best_score} \n")

    return pd.DataFrame()


def evaluate_candidate_models(
    data: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, tuning_params: dict
) -> Pipeline:
    """
    Evaluates candidate models using the given test data and parameters.

    Args:
        data (pd.DataFrame): An empty `pandas.DataFrame`.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.DataFrame): The test data labels.
        tuning_params (dict): A dictionary of tuning parameters.

    Returns:
        Pipeline: The best candidate model.
    """
    run_id = mlflow.active_run().info.run_id

    model_results = []
    y_test = y_test.values.ravel()

    for grid_name in tuning_params["grid_names"]:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{grid_name}")
        acc = np.round(accuracy_score(y_test, model.predict(X_test)), 3)
        auc = np.round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3)

        mlflow.log_metric(f"{grid_name}_accuracy", acc)
        mlflow.log_metric(f"{grid_name}_auc_score", auc)

        logger.info(f"{grid_name}_accuracy: {acc}")
        logger.info(f"{grid_name}_auc_score:{auc}")

        model_results.append(auc)

        try:
            feature_importances = pd.Series(
                model["classifier"].feature_importances_,
                model[:-1].get_feature_names_out(),
            ).sort_values(ascending=True)
        except AttributeError:
            feature_importances = pd.Series(
                model["classifier"].coef_[0], model[:-1].get_feature_names_out()
            ).sort_values(ascending=True)

        # Create a bar chart of feature importance
        plt.figure(figsize=(10, 18))
        plt.barh(feature_importances.index, feature_importances.values)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature importance for the model: {grid_name}")
        plt.tight_layout()

        # Save the chart as a temporary image
        plot_file = f"feature_importances_{grid_name}.png"
        plt.savefig(plot_file)
        plt.close()  # Make sure to close the graph to free up memory
        mlflow.log_artifact(plot_file)
        os.remove(plot_file)

    best_model_index = np.argmax(model_results)
    best_model_name = tuning_params["grid_names"][best_model_index]
    best_candidate_model = mlflow.sklearn.load_model(
        f"runs:/{run_id}/{best_model_name}"
    )
    acc = np.round(accuracy_score(y_test, best_candidate_model.predict(X_test)), 3)
    auc = np.round(
        roc_auc_score(y_test, best_candidate_model.predict_proba(X_test)[:, 1]), 3
    )

    mlflow.log_param("best_candidate_model_name", best_model_name)

    mlflow.log_metric("best_model_accuracy", acc)
    mlflow.log_metric("best_model_auc_score", auc)

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model score: {model_results[best_model_index]}")

    return best_candidate_model


def compare_candidate_models(
    candidate_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    training_params: dict,
    tuning_params: dict,
):
    """
    A function to compare candidate models with a production model
    and potentially register the candidate as the new production model
    if it performs better.

    Args:
        candidate_model (Pipeline): The candidate model to be evaluated.
        X_train (pd.DataFrame): The training input features.
        y_train (pd.DataFrame): The training target values.
        X_test (pd.DataFrame): The testing input features.
        y_test (pd.DataFrame): The testing target values.
        training_params (dict): A dictionary of training parameters.
        tuning_params (dict): A dictionary of tuning parameters.
    """
    client = mlflow.MlflowClient()
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test]).values.ravel()

    y_test = y_test.values.ravel()

    candidate_model_score = round(
        accuracy_score(y_test, candidate_model.predict(X_test)), 3
    )

    try:
        production_model = mlflow.sklearn.load_model(
            model_uri=f"models:/{training_params['model_name']}/Production"
        )
    # Production Model does not exist
    except mlflow.MlflowException:
        logger.info("No production model found")
        if candidate_model_score > tuning_params["min_accuracy_score"]:
            logger.info(
                """Candidate model is better than minimum score.
                        Registering as production model..."""
            )

            candidate_model.fit(X, y)
            mlflow.sklearn.log_model(
                artifact_path="Production",
                sk_model=candidate_model,
                registered_model_name=training_params["model_name"],
                input_example=X.iloc[0:3],
            )
            client.transition_model_version_stage(
                name=training_params["model_name"], version=1, stage="Production"
            )
        else:
            logger.info("Candidate model is not better than minimum score")
        return

    candidate_better_than_production = compare_models(
        candidate_model,
        production_model,
        X_train,
        y_train,
        X_test,
        y_test,
        scoring=accuracy_score,
    )

    if candidate_better_than_production:
        production_model_score = round(
            accuracy_score(y_test, production_model.predict(X_test)), 3
        )

        improvement_percent = round(
            (candidate_model_score - production_model_score) / production_model_score, 3
        )
        logger.info(f"Candidate is better than production by {improvement_percent}%")
        mlflow.log_metric("pct_model_improvement", improvement_percent)

        candidate_model.fit(X, y)
        mlflow.sklearn.log_model(
            artifact_path="Production",
            sk_model=candidate_model,
            registered_model_name=training_params["model_name"],
            input_example=X.iloc[0:3],
        )
        client.transition_model_version_stage(
            name=training_params["model_name"],
            version=client.get_latest_versions(training_params["model_name"])[
                0
            ].version,
            stage="Staging",
        )
