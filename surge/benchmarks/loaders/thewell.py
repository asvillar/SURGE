"""TheWell dataset loader for SURGE.

TheWell (Ohana et al., NeurIPS 2024 D&B) is a large-scale collection of
physics-based PDE datasets totalling ~15 TB across 16 distinct physics
systems — ranging from MHD turbulence and Rayleigh-Bénard convection to
Gray-Scott reaction-diffusion and supernova remnant simulations.

Homepage: https://github.com/PolymathicAI/the_well
Paper:    https://arxiv.org/abs/2412.00568

Status: **Tier 4 — future, manual-trigger only**.  Not included in the
default benchmark run (--tier 0/1/2/3).  These benchmarks must be run
explicitly by key (``python -m surge.benchmarks.run --benchmark thewell.*``).

Installation
------------
    pip install the-well  # Hugging Face streaming; data is NOT cached locally by default

Usage::

    from surge.benchmarks.loaders.thewell import load_thewell, list_thewell_datasets
    X_train, y_train, X_test, y_test = load_thewell("gray_scott")

Guard: requires ``the_well`` package.  Raises ``ImportError`` with
install instructions if absent.

Dataset inventory
-----------------
+---------------------+------------------+-------+
| key                 | physics          | ~size |
+=====================+==================+=======+
| gray_scott          | reaction-diffusion| 6.9 GB|
| turbulence_2d       | 2-D turbulence   | varies|
| mhd                 | MHD turbulence   | varies|
+---------------------+------------------+-------+

Reference
---------
Ohana, R. et al. (2024). The Well: a Large-Scale Collection of Diverse
Physics Simulations for Machine Learning.  NeurIPS 2024 Datasets and
Benchmarks.  arXiv:2412.00568.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_LOG = logging.getLogger("surge.benchmarks.thewell")

THEWELL_AVAILABLE = False
try:
    import the_well  # noqa: F401
    THEWELL_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Mapping: SURGE key → (the_well dataset name, input_field, output_field, description)
_DATASETS: dict[str, dict[str, Any]] = {
    "gray_scott": {
        "well_name": "gray_scott_reaction_diffusion",
        "description": "Gray-Scott reaction-diffusion (2D, 6.9 GB)",
        "shape_hint": "128×128 grid, 2 channels → next timestep",
    },
    "turbulence_2d": {
        "well_name": "turbulence_2d",
        "description": "2D homogeneous turbulence",
        "shape_hint": "128×128 grid, velocity field → next timestep",
    },
    "mhd": {
        "well_name": "mhd_64",
        "description": "3D MHD turbulence (64³)",
        "shape_hint": "64³ grid, 7 fields → next timestep",
    },
}


def list_thewell_datasets() -> list[str]:
    """Return supported TheWell dataset keys."""
    return sorted(_DATASETS)


def load_thewell(
    key: str,
    *,
    n_train: int = 500,
    n_test: int = 100,
    seed: int = 42,
    streaming: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a TheWell dataset via ``the_well`` package.

    Samples are loaded from the first available split, flattened to 2-D
    NumPy arrays ``(n_samples, n_features)`` so they are compatible with
    the SURGE benchmark pipeline.

    Parameters
    ----------
    key:
        One of the keys in :func:`list_thewell_datasets`.
    n_train, n_test:
        Number of samples for training and testing.
    seed:
        Random seed for shuffling.
    streaming:
        If True, use Hugging Face streaming mode (default).  Set to False
        to download the full dataset locally.

    Returns
    -------
    ``(X_train, y_train, X_test, y_test)`` as 2-D float arrays.
    """
    if not THEWELL_AVAILABLE:
        raise ImportError(
            "the_well is required for TheWell benchmarks.\n"
            "Install it with: pip install the-well\n"
            "Or: pip install 'surge-ml[thewell]'"
        )

    if key not in _DATASETS:
        raise KeyError(f"Unknown TheWell dataset {key!r}. Use list_thewell_datasets().")

    meta = _DATASETS[key]
    well_name = meta["well_name"]

    _LOG.info("Loading TheWell dataset %s via the_well …", well_name)

    try:
        from the_well.data import WellDataset

        ds = WellDataset(well_name=well_name, split="train", streaming=streaming)

        rng = np.random.default_rng(seed)
        X_list, y_list = [], []
        for i, sample in enumerate(ds):
            if len(X_list) >= n_train + n_test:
                break
            # Flatten all fields at time t → input, at t+1 → output.
            if hasattr(sample, "field"):
                fields = sample.field
            elif isinstance(sample, dict):
                field_keys = [k for k in sample if "field" in k.lower() or "state" in k.lower()]
                if not field_keys:
                    field_keys = list(sample.keys())
                # Use first timestep as input, second as output.
                arr = np.asarray(sample[field_keys[0]], dtype=float)
                if arr.ndim >= 3:
                    x_flat = arr[0].ravel()
                    y_flat = arr[1].ravel() if arr.shape[0] > 1 else arr[0].ravel()
                else:
                    x_flat = arr.ravel()
                    y_flat = arr.ravel()
                X_list.append(x_flat)
                y_list.append(y_flat)
                continue
            else:
                arr = np.asarray(sample, dtype=float)
                x_flat = arr.ravel()
                X_list.append(x_flat)
                y_list.append(x_flat)

        X = np.array(X_list[: n_train + n_test])
        y = np.array(y_list[: n_train + n_test])

        idx = rng.permutation(len(X))
        X_train = X[idx[:n_train]]
        y_train = y[idx[:n_train]]
        X_test = X[idx[n_train : n_train + n_test]]
        y_test = y[idx[n_train : n_train + n_test]]
        return X_train, y_train, X_test, y_test

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TheWell dataset {well_name!r}: {exc}\n"
            "Make sure the_well is installed and you have internet access "
            "for Hugging Face streaming."
        ) from exc
