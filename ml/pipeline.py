"""
FIXED (the leakage fix):
WRONG: df[col].fillna(df[col].mean()) and pd.get_dummies(df[...])
on the FULL dataframe, before train_test_split. That means the mean and the
category list were computed using rows that later became your test set.

Here, imputation + encoding are wrapped in a sklearn ColumnTransformer.
It only ever calls .fit() on X_train (done in train.py) — X_test just gets
.transform()'d with statistics learned from train. Nothing from test leaks in.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "symboling", "normalized-losses", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-L/100km", "highway-mpg",
    "num-of-doors", "num-of-cylinders",
]

CATEGORICAL_FEATURES = [
    "make", "fuel-type", "aspiration", "body-style",
    "drive-wheels", "engine-location", "engine-type", "fuel-system",
]


def build_preprocessor() -> ColumnTransformer:
    # same strategy you used, now leakage-safe
    numeric_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = OneHotEncoder(
        drop="first", handle_unknown="ignore")

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_model_pipeline(model) -> Pipeline:
    """Wraps any sklearn-compatible model (LinearRegression, RandomForest, XGBoost)
    with the same leakage-safe preprocessing."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
