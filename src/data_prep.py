import os
import sys
import yaml
import pandas as pd
from datetime import datetime


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found. Script cannot proceed.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def append_manifest(script_name, input_path, output_paths, description):
    manifest_path = "manifest.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = []
    entry.append(f"Timestamp: {timestamp}")
    entry.append(f"Script: {script_name}")
    entry.append(f"Input: {input_path}")
    entry.append("Output:")
    for path in output_paths:
        entry.append(f"  - {path}")
    entry.append(f"Description: {description}")

    with open(manifest_path, "a") as f:
        f.write("\n".join(entry))


def main():
    # Load config
    config = load_config()

    # Read config values
    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    production_dir = config["data"]["production_dir"]
    current_version = config["data"]["current_version"]
    train_size = config["data"]["train_size"]

    # Validate raw file exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    # Load dataset
    df = pd.read_csv(raw_path)

    if train_size >= len(df):
        raise ValueError("train_size must be smaller than total dataset length.")

    # Chronological split
    train_df = df.iloc[:train_size]
    prod_df = df.iloc[train_size:]

    # Construct filenames dynamically
    train_filename = f"{current_version}_train.csv"
    prod_filename = f"{current_version}_prod.csv"

    train_path = os.path.join(processed_dir, train_filename)
    prod_path = os.path.join(production_dir, prod_filename)

    # Prevent overwriting existing versions
    if os.path.exists(train_path) or os.path.exists(prod_path):
        raise FileExistsError(
            f"Version {current_version} already exists. "
            f"Please update current_version in config.yaml."
        )

    # Save files
    train_df.to_csv(train_path, index=False)
    prod_df.to_csv(prod_path, index=False)

    # Append to manifest
    description = (
        f"Chronological split ({train_size} train / "
        f"{len(df) - train_size} production)"
    )

    append_manifest(
        script_name="data_prep.py",
        input_path=raw_path,
        output_paths=[train_path, prod_path],
        description=description,
    )

    print("Data preparation completed successfully.")
    print(f"Train saved to: {train_path}")
    print(f"Production saved to: {prod_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
