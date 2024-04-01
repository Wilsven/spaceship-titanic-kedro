"""
This is a boilerplate pipeline 'model_inference'
generated using Kedro 0.19.3
"""

import mlflow
import numpy as np
import pandas as pd

from spaceship_titanic.pipelines.data_processing.nodes import preprocess_data


def prepare_inference_data(
    data: pd.DataFrame, preprocessing_params: dict
) -> pd.DataFrame:
    """
    Prepares the data for inference.

    Args:
        data (pd.DataFrame): The input data to be prepared for inference.
        preprocessing_params (dict): A dictionary of preprocessing parameters.

    Returns:
        pd.DataFrame: The preprocessed data ready for inference.
    """
    inference_data = preprocess_data(data, preprocessing_params)
    return inference_data


def predict_inference(
    inference_data: pd.DataFrame, training_params: dict
) -> pd.DataFrame:
    """
    A function to perform inference using a trained model on the provided data.

    Parameters:
        inference_data (pd.DataFrame): The data which inference is performed.
        training_params (dict): A dictionary of training parameters.

    Returns:
        pd.DataFrame:
            A DataFrame containing the PassengerId and predicted
            `Transported` values.
    """
    inference_model = mlflow.sklearn.load_model(
        model_uri=f"models:/{training_params['model_name']}/Production"
    )
    features = inference_model.feature_names_in_

    X = inference_data[features]
    predictions = inference_model.predict(X)

    submission_data = pd.DataFrame(
        {"PassengerId": inference_data["PassengerId"], "Transported": predictions}
    )
    submission_data["Transported"] = np.where(
        submission_data["Transported"] == 1, True, False
    )
    return submission_data
