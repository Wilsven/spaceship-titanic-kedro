import pandas as pd


def gen_counts_for_cat_feature(
    data: pd.DataFrame, cat_col: str, feature_name: str
) -> pd.DataFrame:
    """Generates the counts for a categorical feature.

    Args:
        data (pd.DataFrame): A `pandas.DataFrame`.
        cat_col (str): The categorical feature to generate counts from.
        feature_name (str): The feature name to rename to.

    Returns:
        pd.DataFrame:
            Output `pandas.DataFrame` with newly generated
            count column from the categorical column.
    """
    df_count = data.groupby(cat_col).size().reset_index(name=feature_name)
    data = data.merge(df_count, on=[cat_col], how="left")
    return data


def create_num_features_from_cat_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates numerical features from `CabinNumber`, `CabinDeck`
    and `LastName` columns.

    Args:
        input_data (pd.DataFrame): The input `pandas.DataFrame`.

    Returns:
        pd.DataFrame:
            Output `pandas.DataFrame` with newly
            generated numerical features from `CabinNumber`,
            `CabinDeck` and `LastName` columns.

    """
    cat_counts_params = {
        "CabinNumber": {"FeatureName": "PeopleInCabinNumber", "remove_col": True},
        "CabinDeck": {"FeatureName": "PeopleInCabinDeck", "remove_col": False},
        "LastName": {"FeatureName": "FamilySize", "remove_col": True},
        "Group": {"FeatureName": "GroupSize", "remove_col": True},
    }

    data = input_data.copy()

    for col in list(cat_counts_params.keys()):
        # Generate the counts for each of the categorical features
        data = gen_counts_for_cat_feature(
            data, col, cat_counts_params[col]["FeatureName"]
        )

        if cat_counts_params[col]["remove_col"]:
            data = data.drop(columns=[col])
    return data


def create_age_cat_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Creates a categorical feature from the `Age` column.

    Args:
        input_data (pd.DataFrame): The input `pandas.DataFrame`.

    Returns:
        pd.DataFrame:
            Output `pandas.DataFrame` with newly
            generated categorical feature from the
            `Age` column.
    """
    data = input_data.copy()
    data["AgeCategory"] = pd.cut(
        data["Age"],
        bins=[0, 12, 18, 25, 50, 200],
        labels=["Child", "Teenager", "PreAdult", "Adult", "Elder"],
    )
    data = data.drop(columns=["Age"])
    return data


def create_expenditure_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Creates two numerical features regarding expenditures.

    Args:
        input_data (pd.DataFrame): The input `pandas.DataFrame`.

    Returns:
        pd.DataFrame:
            Output `pandas.DataFrame` with newly
            generated expenditure features.
    """
    data = input_data.copy()
    expenditure_features = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    data["Expenditure"] = data[expenditure_features].sum(axis=1)
    data["NoExpenditure"] = (data["Expenditure"] == 0).astype(int)
    return data


def create_cabin_region(input_data: pd.DataFrame) -> pd.DataFrame:
    """Create a categorical feature from the `CabinNumber` column.

    Args:
        input_data (pd.DataFrame): The input `pandas.DataFrame`.

    Returns:
        pd.DataFrame:
            Output `pandas.DataFrame` with newly
            generated categorical feature from
            from the `CabinNumber` column..
    """

    def _return_cabin_region(cabin_num: int) -> str:
        """Returns a category based on the `CabinNumber`."""
        if cabin_num <= 300:
            return "A"
        elif cabin_num < 600:
            return "B"
        elif cabin_num < 900:
            return "C"
        elif cabin_num < 1200:
            return "D"
        elif cabin_num < 1500:
            return "E"
        elif cabin_num < 1800:
            return "F"
        else:
            return "G"

    data = input_data.copy()
    data["CabinRegion"] = data["CabinNumber"].apply(lambda x: _return_cabin_region(x))
    return data
