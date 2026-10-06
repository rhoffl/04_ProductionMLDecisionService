from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_baseline": Pipeline(
            [("preprocess", preprocessor()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.07, max_leaf_nodes=24, l2_regularization=1.0, random_state=seed
                    ),
                ),
            ]
        ),
    }

