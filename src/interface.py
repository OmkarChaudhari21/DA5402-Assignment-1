import os
import yaml
import pickle
import subprocess
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# CONFIG LOADING

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found. API cannot start.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "Git hash unavailable"


# LOAD CONFIG

config = load_config()

data_config = config["data"]
deployment_config = config["deployment"]

current_version = data_config["current_version"]
model_save_dir = deployment_config["model_save_dir"]

model_filename = f"model_{current_version}.pkl"
model_path = os.path.join(model_save_dir, model_filename)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)


# DEPLOYMENT LOGGING

deployment_log_path = "deployment_log.csv"

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
git_hash = get_git_commit_hash()

log_entry = f"{timestamp},{current_version},{model_path},{git_hash}\n"

if not os.path.exists(deployment_log_path):
    with open(deployment_log_path, "w") as f:
        f.write("timestamp,model_version,model_path,git_commit_hash\n")

with open(deployment_log_path, "a") as f:
    f.write(log_entry)

# FASTAPI APP

app = FastAPI()


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.features])
    prediction = model.predict(input_df)

    return {
        "model_version": current_version,
        "prediction": int(prediction[0])
    }
