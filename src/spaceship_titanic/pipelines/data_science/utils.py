import copy
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_train_data(
    modeling_data: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Get train data from the given modeling data and parameters.

    Args:
        modeling_data (pd.DataFrame): The input modeling data.
        params (dict): The parameters for data processing.

    Returns:
        tuple[pd.DataFrame, pd.Series]:
            A tuple containing the input features and the target variable.
    """
    target_col = params["col_maps"]["target_col"]
    modeling_data[target_col] = np.where(modeling_data[target_col], 1, 0)
    features = modeling_data.columns.difference(
        [params["col_maps"]["id_col"], target_col]
    )

    X = modeling_data[features]
    y = modeling_data[target_col]

    return X, y


def compare_models(
    candidate_model_input: Pipeline,
    model_input: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    scoring: Callable,
) -> bool:
    """
    Compares two machine learning models on the given dataset and returns a
    boolean indicating whether the candidate model performs better than the base model.

    Parameters:
        candidate_model_input (Pipeline): The candidate model to compare.
        model_input (Pipeline): The base model to compare against.
        X_train (pd.DataFrame): The training data features.
        y_train (np.ndarray): The training data labels.
        X_test (pd.DataFrame): The test data features.
        y_test (np.ndarray): The test data labels.
        scoring (Callable): The scoring function to use for evaluating the models.

    Returns:
        bool: True if candidate model performs better than the base model else False.
    """
    candidate_model = copy.deepcopy(candidate_model_input)
    model = copy.deepcopy(model_input)

    candidate_model.fit(X_train, y_train)
    model.fit(X_train, y_train)

    candidate_model_score = scoring(y_test, candidate_model.predict(X_test))
    model_score = scoring(y_test, model.predict(X_test))

    if candidate_model_score > model_score:
        return True
    return False
