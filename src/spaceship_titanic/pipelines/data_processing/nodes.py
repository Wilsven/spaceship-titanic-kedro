"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

import numpy as np
import pandas as pd

from spaceship_titanic.pipelines.data_processing.utils import (
    create_age_cat_features,
    create_cabin_region,
    create_num_features_from_cat_features,
)


def preprocess_data(
    input_data: pd.DataFrame, preprocessing_params: dict
) -> pd.DataFrame:
    """
    Preprocesses the input data by performing various transformations on it.

    Args:
        input_data (pd.DataFrame): The input data to be preprocessed.
        preprocessing_params (dict): A dictionary of preprocessing parameters.

    Returns:
        pd.DataFrame: The preprocessed data after applying the transformations.
    """
    processed_data = input_data.copy()
    processed_data[["CabinDeck", "CabinNumber", "CabinSide"]] = processed_data[
        "Cabin"
    ].str.split("/", expand=True)
    processed_data = processed_data.astype({"CabinNumber": "float64"})
    processed_data = create_cabin_region(processed_data)

    processed_data["LastName"] = processed_data["Name"].str.split(" ").str[1]
    processed_data["Group"] = (
        processed_data["PassengerId"].apply(lambda x: x.split("_")[0]).astype(int)
    )
    processed_data = processed_data.drop(columns=["Cabin", "Name"])

    processed_data = create_num_features_from_cat_features(processed_data)
    processed_data = create_age_cat_features(processed_data)

    for feature in preprocessing_params["filter_cat_features"].keys():
        processed_data[feature] = np.where(
            processed_data[feature].isin(
                preprocessing_params["filter_cat_features"][feature]
            ),
            "Other",
            processed_data[feature],
        )

    return processed_data
