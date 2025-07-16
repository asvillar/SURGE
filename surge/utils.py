import json
import os
from pathlib import Path

import joblib
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    joblib.dump(model, path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def setup_surge_path():
    """
    Check if SURGE environment variable is set and create it if needed.

    Returns:
        Path: The SURGE repository path

    Raises:
        FileNotFoundError: If SURGE path cannot be determined or doesn't exist
    """
    # Check if SURGE environment variable exists
    surge_path = os.environ.get('SURGE')

    if surge_path is None:
        # Try to determine SURGE path from current file location
        current_file = Path(__file__).resolve()
        # Navigate up from surge/utils.py to find the repo root
        potential_surge_path = current_file.parent.parent

        # Verify this is indeed the SURGE repo by checking for key directories
        if (potential_surge_path / 'surge').exists() and (potential_surge_path / 'notebooks').exists():
            surge_path = str(potential_surge_path)
            # Set the environment variable for this session and future use
            os.environ['SURGE'] = surge_path
            print(f"✅ SURGE environment variable set to: {surge_path}")
        else:
            raise FileNotFoundError(
                "Could not automatically determine SURGE repository path. "
                "Please set the SURGE environment variable manually: "
                "export SURGE=/path/to/your/SURGE/repo"
            )
    else:
        print(f"✅ Using existing SURGE environment variable: {surge_path}")

    # Validate the path exists
    surge_path_obj = Path(surge_path)
    if not surge_path_obj.exists():
        raise FileNotFoundError(f"SURGE path does not exist: {surge_path}")

    return surge_path_obj


def get_data_path(dataset_name="example-data"):
    """
    Get the path to a specific dataset in the SURGE repository.

    Args:
        dataset_name (str): Name of the dataset directory

    Returns:
        Path: Path to the dataset directory
    """
    surge_path = setup_surge_path()
    data_path = surge_path / "data" / "datasets" / dataset_name

    if not data_path.exists():
        available_datasets = []
        datasets_dir = surge_path / "data" / "datasets"
        if datasets_dir.exists():
            available_datasets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]

        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {data_path}.\n"
            f"Available datasets: {available_datasets}"
        )

    return data_path
