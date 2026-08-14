"""
Downloads the raw UCI 'imports-85' dataset and assigns column headers.

"""
import pandas as pd
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

HEADERS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base",
    "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
    "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]


def download_raw_data(output_path: str = "data/raw/imports-85.csv") -> pd.DataFrame:
    df = pd.read_csv(URL, header=None)
    df.columns = HEADERS

    # works on your machine. Use a relative path instead so anyone (or Docker)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = download_raw_data()
    print(df.head())
