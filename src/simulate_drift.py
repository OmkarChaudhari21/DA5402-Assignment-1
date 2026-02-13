import pandas as pd
import os
import yaml
from datetime import datetime


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def append_manifest(script_name, input_path, output_path, description):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = []
    entry.append(f"Timestamp: {timestamp}")
    entry.append(f"Script: {script_name}")
    entry.append(f"Input: {input_path}")
    entry.append("Output:")
    entry.append(f"  - {output_path}")
    entry.append(f"Description: {description}")
    with open("manifest.txt", "a") as f:
        f.write("\n".join(entry))


def main():
    config = load_config()

    production_dir = config["data"]["production_dir"]
    current_version = config["data"]["current_version"]

    input_path = os.path.join(production_dir, f"{current_version}_prod.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Production dataset not found at {input_path}")

    # Increment version manually
    drift_version = "v3"
    output_path = os.path.join(production_dir, f"{drift_version}_prod.csv")

    df = pd.read_csv(input_path)

    # DRIFT LOGIC
    if "Torque [Nm]" in df.columns:
        df["Torque [Nm]"] *= 1.05  # +5%

    if "Air temperature [K]" in df.columns:
        df["Air temperature [K]"] += 2  # +2 Kelvin

    df.to_csv(output_path, index=False)

    append_manifest(
        script_name="simulate_drift.py",
        input_path=input_path,
        output_path=output_path,
        description="Simulated sensor drift: Torque +5%, Air temperature +2K"
    )

    print(f"Drift simulation complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
