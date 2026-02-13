import os
import sys
import yaml
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found. Training cannot proceed.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config()

    data_config = config["data"]
    model_config = config["model"]
    deployment_config = config["deployment"]

    current_version = data_config["current_version"]
    processed_dir = data_config["processed_dir"]

    # Build training dataset path dynamically
    train_filename = f"{current_version}_train.csv"
    train_path = os.path.join(processed_dir, train_filename)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at {train_path}")

     # Load dataset
    df = pd.read_csv(train_path)

    # Drop identifier columns if present
    drop_cols = []
    for col in ["UDI", "Product ID"]:
        if col in df.columns:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # One-hot encode categorical columns
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Initialize model using config hyperparameters
    if model_config["algorithm"] != "RandomForest":
        raise ValueError("Currently only RandomForest is supported.")

    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        random_state=model_config["random_state"]
    )

    # Train model
    model.fit(X, y)

    # Save model dynamically
    model_filename = f"model_{current_version}.pkl"
    model_path = os.path.join(deployment_config["model_save_dir"], model_filename)

    # Prevent overwriting
    if os.path.exists(model_path):
        raise FileExistsError(
            f"Model for version {current_version} already exists. "
            "Please update current_version in config.yaml."
        )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Training completed successfully.")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
