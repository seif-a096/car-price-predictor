"""
Deterministic cleaning — no statistic here is *learned* from the data
(no mean, no correlation), so it's safe to run before train/test split.
"""
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # data doesn't have NaN values, but has "?" values, which are considered
    # as missing values. So we need to replace "?" with NaN values.
    df.replace("?", np.nan, inplace=True)

    # droping duplicates, so the dataset will be more accurate and reliable.
    # Duplicates can skew the results of the analysis and lead to incorrect conclusions.
    df.drop_duplicates(inplace=True)

    # convert non-numeric data to numeric data, so that we can use it for
    # analysis and modeling.
    numeric_cols = ["normalized-losses", "bore",
                    "stroke", "horsepower", "peak-rpm", "price"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # data formating.. miles per gallon to kilometer per liter, pounds to kilogram, etc.
    df["city-mpg"] = 235 / df["city-mpg"]
    df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)

    # convert number-of-doors via map method
    mapping = {"two": 2, "four": 4}
    df["num-of-doors"] = df["num-of-doors"].map(mapping)
    # same for number-of-cylinders
    mapping = {"two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "eight": 8, "twelve": 12}
    df["num-of-cylinders"] = df["num-of-cylinders"].map(mapping)

    return df


# NOTE: your original code also did mean-imputation and one-hot encoding here,
# BEFORE train_test_split. That's the leakage bug. Both of those now live in
# pipeline.py, where they get fit ONLY on X_train (see train.py).
