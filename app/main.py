"""
fast-api backend for the ML model
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.schemas import PredictionRequest
import pandas as pd
import joblib

app = FastAPI(title="ML Model API", version="1.0")
# Load the trained model
model = joblib.load("models/linear_regression_model.pkl")

COLUMN_RENAME_MAP = {
    "normalized_losses": "normalized-losses",
    "wheel_base": "wheel-base",
    "curb_weight": "curb-weight",
    "engine_size": "engine-size",
    "compression_ratio": "compression-ratio",
    "peak_rpm": "peak-rpm",
    "city_L_100km": "city-L/100km",
    "highway_mpg": "highway-mpg",
    "num_of_doors": "num-of-doors",
    "num_of_cylinders": "num-of-cylinders",
    "fuel_type": "fuel-type",
    "body_style": "body-style",
    "drive_wheels": "drive-wheels",
    "engine_location": "engine-location",
    "engine_type": "engine-type",
    "fuel_system": "fuel-system",
}


@app.get("/")
def root():
    return JSONResponse(content={
        "message": "Car price predictor API. Use /predict to get a prediction."
    })


@app.post("/predict")
def predict(features: PredictionRequest):
    data = features.model_dump()  # convert Pydantic model to dict
    renamed = {COLUMN_RENAME_MAP.get(k, k): v for k, v in data.items()}
    df = pd.DataFrame([renamed])
    prediction = model.predict(df)[0]
    return {"predicted_price": float(prediction)}
