import xgboost as xgb
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


def get_tuning_grid(
    grid_name: str, numeric_features: list[str], categorical_features: list[str]
) -> dict:
    numeric_transformer = Pipeline(
        steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    tuning_grids = {
        "default": {
            "pipeline": Pipeline(
                [("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier)]
            ),
            "param_grid": {},
        },
        "xgboost": {
            "pipeline": Pipeline(
                [("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier())]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 150, 200, 250, 300],
                "classifier__learning_rate": [0.1, 0.05],
                "classifier__gamma": [0, 0.5, 1, 1.5, 2, 5],
                "classifier__reg_alpha": [0, 0.1, 0.3, 0.5, 0.8],
                "classifier__reg_lambda": [0.2, 0.4, 0.6, 0.8, 1],
                "classifier__min_child_weight": [1, 3, 5],
                "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1],
                "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier()),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [350, 500, 600],
                "classifier__max_features": randint(1, 16),
                "classifier__max_depth": [None] + list(range(5, 21, 5)),
                "classifier__min_samples_split": randint(2, 15),
                "classifier__min_samples_leaf": randint(1, 15),
            },
        },
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("poly", PolynomialFeatures()),
                    ("classifier", LogisticRegression()),
                ]
            ),
            "param_grid": {
                "poly__degree": [1, 2],
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["elasticnet"],
                "classifier__l1_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1],
                "classifier__solver": ["saga"],
                "classifier__max_iter": [10000],
            },
        },
        "support_vector": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", SVC(probability=True)),
                ]
            ),
            "param_grid": {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": [0.1, 1, 10, 100],
                "classifier__kernel": ["linear", "poly"],
                "classifier__degree": [1, 2],
            },
        },
    }

    return tuning_grids[grid_name]
