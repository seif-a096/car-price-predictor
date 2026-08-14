"""
Split FIRST, then everything else is fit only on X_train.
Combines your model comparison (Linear Regression, Random Forest, XGBoost)
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

from ml.preprocessing import clean_data
from ml.pipeline import build_model_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_and_prepare(raw_csv_path: str):
    df = pd.read_csv(raw_csv_path)
    df = clean_data(df)
    df = df.dropna(subset=["price"])  # target must be present

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["price"]
    return X, y


def train_and_evaluate(raw_csv_path: str, model_out_path: str = "models/model.pkl"):
    X, y = load_and_prepare(raw_csv_path)

    # 2. Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    best_name, best_score, best_pipeline = None, -float("inf"), None

    for name, model in candidates.items():
        pipe = build_model_pipeline(model)
        pipe.fit(X_train, y_train)  # preprocessor is fit HERE, on train only

        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # cross-validation to evaluate the model's performance more robustly.
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

        results[name] = {"r2": r2, "mse": mse, "cv_mean_r2": cv_scores.mean()}
        print(
            f"{name}: R²={r2:.4f}  MSE={mse:.2f}  CV R² (mean)={cv_scores.mean():.4f}")

        if r2 > best_score:
            best_name, best_score, best_pipeline = name, r2, pipe
        if name == "LinearRegression":
            # dump the linear regression model for later use (e.g., for inference)
            joblib.dump(pipe, "models/linear_regression_model.pkl")

    print(f"\nBest model: {best_name} (R²={best_score:.4f})")
    joblib.dump(best_pipeline, model_out_path)
    print(f"Saved to {model_out_path}")

    return results, best_pipeline


if __name__ == "__main__":
    train_and_evaluate("data/raw/imports-85.csv")
