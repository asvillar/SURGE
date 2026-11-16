#!/usr/bin/env python
"""Short installation check: import surge, print its version and list which
optional backends import. Works from a pip install or a source checkout.

    python scripts/utils/check_installation.py
"""
from __future__ import annotations

import importlib
import sys

OPTIONAL_BACKENDS = {
    "torch": "PyTorch models",
    "tensorflow": "Keras models",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "gpflow": "GPflow Gaussian processes",
    "botorch": "BoTorch Gaussian processes",
    "optuna": "hyperparameter search",
    "mlflow": "run tracking",
    "h5py": "benchmark datasets (HDF5)",
    "onnx": "ONNX export",
}


def main() -> int:
    try:
        import surge
    except ImportError as exc:
        print(f"surge cannot be imported: {exc}")
        print("Install it with: pip install surge-ml")
        return 1
    print(f"surge {surge.__version__} ({sys.version.split()[0]}) at {surge.__file__}")
    print("Optional backends:")
    for name, purpose in OPTIONAL_BACKENDS.items():
        try:
            importlib.import_module(name)
            status = "available"
        except Exception:  # missing or broken install
            status = "not available"
        print(f"  {name:<10} {status:<14} {purpose}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
