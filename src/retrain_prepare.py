import pandas as pd
import os
import yaml
from datetime import datetime


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def append_manifest(script_name, input_paths, output_path, description):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = []
    entry.append(f"Timestamp: {timestamp}")
    entry.append(f"Script: {script_name}")
    entry.append("Input:")
    for path in input_paths:
        entry.append(f"  - {path}")
    entry.append("Output:")
    entry.append(f"  - {output_path}")
    entry.append(f"Description: {description}")
    with open("manifest.txt", "a") as f:
        f.write("\n".join(entry))


def main():
    config = load_config()

    processed_dir = config["data"]["processed_dir"]
    production_dir = config["data"]["production_dir"]

    old_train = os.path.join(processed_dir, "v2_train.csv")
    drift_prod = os.path.join(production_dir, "v3_prod.csv")

    new_train = os.path.join(processed_dir, "v3_train.csv")

    df_train = pd.read_csv(old_train)
    df_prod = pd.read_csv(drift_prod)

    # Combine for retraining
    df_combined = pd.concat([df_train, df_prod], ignore_index=True)

    df_combined.to_csv(new_train, index=False)

    append_manifest(
        script_name="retrain_prepare.py",
        input_paths=[old_train, drift_prod],
        output_path=new_train,
        description="Combined v2_train and v3_prod for retraining"
    )

    print(f"New retraining dataset created at {new_train}")


if __name__ == "__main__":
    main()
