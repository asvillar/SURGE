import json

import joblib
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    joblib.dump(model, path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
