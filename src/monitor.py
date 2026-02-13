import requests
import pandas as pd
import yaml
import os
from sklearn.metrics import accuracy_score


API_URL = "http://127.0.0.1:5000/predict"


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    production_dir = config["data"]["production_dir"]
    threshold = config["deployment"]["threshold"]

    # Evaluate drifted dataset
    prod_version = "v3"
    prod_path = os.path.join(production_dir, f"{prod_version}_prod.csv")

    if not os.path.exists(prod_path):
        raise FileNotFoundError(f"{prod_path} not found")

    df = pd.read_csv(prod_path)

    # Apply same preprocessing used in training
    drop_cols = []
    for col in ["UDI", "Product ID"]:
        if col in df.columns:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    sample_size = 200

    X = df.iloc[:, :-1].head(sample_size)
    y_true = df.iloc[:, -1].head(sample_size)


    predictions = []

    for _, row in X.iterrows():
        payload = {"features": list(row.values)}
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            raise RuntimeError("API call failed during monitoring")

        pred = response.json()["prediction"]
        predictions.append(pred)

    production_accuracy = accuracy_score(y_true, predictions)
    production_error = 1 - production_accuracy

    print("MONITORING REPORT")
    print(f"Evaluated Dataset: {prod_version}")
    print(f"Production Accuracy: {production_accuracy}")
    print(f"Production Error: {production_error}")
    print(f"Threshold: {threshold}")

    if production_error > threshold:
        print("Drift detected — Retraining required.")
    else:
        print("Model performance within acceptable range.")


if __name__ == "__main__":
    main()
