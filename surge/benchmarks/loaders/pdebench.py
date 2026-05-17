"""PDEBench HDF5 loader for SURGE.

PDEBench (Takamoto et al., NeurIPS 2022) provides a standardised PDE
surrogate benchmark suite.  Data is stored as HDF5 files, downloadable
from DaRUS / Zenodo.

This module downloads HDF5 files on first use (like ``fetch_openml``)
and caches them in ``~/.surge/data/pdebench/``.  The loader converts the
raw data into flat NumPy arrays ``(X, y)`` suitable for the SURGE
benchmark pipeline.

Usage::

    from surge.benchmarks.loaders.pdebench import load_pdebench
    X_train, y_train, X_test, y_test = load_pdebench("burgers_1d")

Guard: all imports and downloads are gated on ``h5py`` availability.
If ``h5py`` is absent, a ``ImportError`` with a clear install message is
raised when you try to load data.

Dataset inventory
-----------------
+-------------------+------------------+----------------------------+
| key               | approx. size     | DaRUS link (default URL)   |
+===================+==================+============================+
| burgers_1d        | ~57 MB           | see _URLS below            |
| darcy_2d          | ~1.2 GB          | see _URLS below            |
| shallow_water_2d  | ~4.2 GB          | see _URLS below            |
+-------------------+------------------+----------------------------+

References
----------
Takamoto, M. et al. (2022). PDEBench: An Extensive Benchmark for
Scientific Machine Learning.  NeurIPS 2022 Datasets and Benchmarks.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("surge.benchmarks.pdebench")

H5PY_AVAILABLE = False
try:
    import h5py  # noqa: F401
    H5PY_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Dataset URLs (DaRUS / Zenodo)
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".surge" / "data" / "pdebench"

# File names and download URLs.
# The URLs point to publicly accessible HDF5 files hosted on DaRUS.
_DATASETS: dict[str, dict[str, Any]] = {
    "burgers_1d": {
        "filename": "1D_Burgers_Sols_Nu0.01.hdf5",
        "url": (
            "https://darus.uni-stuttgart.de/api/access/datafile/"
            ":persistentId?persistentId=doi:10.18419/darus-2986/3"
        ),
        "n_samples_train": 900,
        "n_samples_test": 100,
        "description": "1D Burgers, ν=0.01, shape (1000, 101, 1024, 1)",
        "loader": "_load_burgers_hdf5",
    },
    "darcy_2d": {
        "filename": "2D_DarcyFlow_beta1.0_Train.hdf5",
        "url": (
            "https://darus.uni-stuttgart.de/api/access/datafile/"
            ":persistentId?persistentId=doi:10.18419/darus-2986/8"
        ),
        "n_samples_train": 900,
        "n_samples_test": 100,
        "description": "2D Darcy Flow, beta=1.0, 128×128 grid",
        "loader": "_load_darcy_hdf5",
    },
    "shallow_water_2d": {
        "filename": "2D_rdb_NA_NA.hdf5",
        "url": (
            "https://darus.uni-stuttgart.de/api/access/datafile/"
            ":persistentId?persistentId=doi:10.18419/darus-2986/27"
        ),
        "n_samples_train": 900,
        "n_samples_test": 100,
        "description": "2D Shallow Water Equations, 128×128 grid",
        "loader": "_load_shallow_water_hdf5",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_pdebench_datasets() -> list[str]:
    """Return supported PDEBench dataset keys."""
    return sorted(_DATASETS)


def is_cached(key: str) -> bool:
    """Return True if the HDF5 file is already downloaded."""
    if key not in _DATASETS:
        raise KeyError(f"Unknown PDEBench dataset {key!r}. Use list_pdebench_datasets().")
    return (_CACHE_DIR / _DATASETS[key]["filename"]).exists()


def load_pdebench(
    key: str,
    *,
    download: bool = True,
    cache_dir: Path | str | None = None,
    train_samples: int | None = None,
    test_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a PDEBench dataset, downloading on first use.

    Parameters
    ----------
    key:
        One of the keys in :func:`list_pdebench_datasets`.
    download:
        If True (default), download the file if not cached.  Set to False
        to skip downloading and raise FileNotFoundError if absent.
    cache_dir:
        Override the default ``~/.surge/data/pdebench/`` cache directory.
    train_samples, test_samples:
        Limit the number of samples loaded (useful for quick tests).
    seed:
        Random seed for splitting.

    Returns
    -------
    (X_train, y_train, X_test, y_test) as 2-D numpy arrays:
        Shape (n_samples, n_features) where features = flattened spatial grid.
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for PDEBench loaders.\n"
            "Install it with: pip install h5py\n"
            "Or: pip install 'surge-ml[benchmarks]'"
        )

    if key not in _DATASETS:
        raise KeyError(f"Unknown PDEBench dataset {key!r}. Use list_pdebench_datasets().")

    meta = _DATASETS[key]
    root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    fpath = root / meta["filename"]

    if not fpath.exists():
        if not download:
            raise FileNotFoundError(
                f"PDEBench file not found at {fpath}.\n"
                "Run with download=True to download it automatically."
            )
        _download(meta["url"], fpath)

    # Dispatch to dataset-specific loader.
    loader_fn = globals()[meta["loader"]]
    X_train, y_train, X_test, y_test = loader_fn(
        fpath,
        n_train=train_samples or meta["n_samples_train"],
        n_test=test_samples or meta["n_samples_test"],
        seed=seed,
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    _LOG.info("Downloading %s → %s", url, dest)
    print(f"[pdebench] Downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"[pdebench] Saved to {dest}")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Download failed for {url!r}.\n"
            "You can manually download the file and place it at:\n"
            f"  {dest}\n"
            "PDEBench data is available at: https://darus.uni-stuttgart.de/dataverse/pdebench"
        ) from exc


# ---------------------------------------------------------------------------
# Dataset-specific HDF5 loaders
# ---------------------------------------------------------------------------


def _load_burgers_hdf5(
    path: Path, *, n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Burgers 1D: HDF5 shape ``(1000, T, n_x, 1)``.
    Task: predict u(x, T_last) from u(x, 0).
    """
    import h5py
    from sklearn.model_selection import train_test_split

    with h5py.File(path, "r") as f:
        # Field: (n_samples, T, n_x, 1)
        key = list(f.keys())[0]
        data = f[key][()]  # Load fully; shape (N, T, nx, 1)

    N, T, nx, _ = data.shape
    n_use = min(N, n_train + n_test)
    data = data[:n_use, :, :, 0]  # (n_use, T, nx)
    X = data[:, 0, :]   # IC: (n_use, nx)
    y = data[:, -1, :]  # Final state: (n_use, nx)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test / n_use, random_state=seed
    )
    return X_tr[:n_train], y_tr[:n_train], X_te[:n_test], y_te[:n_test]


def _load_darcy_hdf5(
    path: Path, *, n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Darcy Flow 2D: HDF5 shape ``(N, nx, ny)``.
    Task: predict u(x, y) from permeability field a(x, y).
    Flatten spatial dims → (N, nx*ny).
    """
    import h5py
    from sklearn.model_selection import train_test_split

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        # Typically: "nu" (permeability) and "sol" (solution)
        perm_key = [k for k in keys if "nu" in k.lower() or "perm" in k.lower()][0]
        sol_key = [k for k in keys if "sol" in k.lower()][0]
        a = f[perm_key][()]  # (N, nx, ny)
        u = f[sol_key][()]   # (N, nx, ny)

    N = min(a.shape[0], n_train + n_test)
    X = a[:N].reshape(N, -1).astype(float)
    y = u[:N].reshape(N, -1).astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test / N, random_state=seed
    )
    return X_tr[:n_train], y_tr[:n_train], X_te[:n_test], y_te[:n_test]


def _load_shallow_water_hdf5(
    path: Path, *, n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shallow Water Equations 2D.
    Task: predict water height h(x, y, T) from h(x, y, 0).
    """
    import h5py
    from sklearn.model_selection import train_test_split

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        # Usually has a "data" field: (N, T, nx, ny, channels)
        data_key = keys[0]
        data = f[data_key][()]

    if data.ndim == 5:
        N, T, nx, ny, C = data.shape
        # Use height channel (index 0) only.
        data = data[:, :, :, :, 0]
    elif data.ndim == 4:
        N, T, nx, ny = data.shape
    else:
        raise ValueError(f"Unexpected HDF5 shape {data.shape} for shallow water")

    N_use = min(N, n_train + n_test)
    X = data[:N_use, 0, :, :].reshape(N_use, -1).astype(float)
    y = data[:N_use, -1, :, :].reshape(N_use, -1).astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test / N_use, random_state=seed
    )
    return X_tr[:n_train], y_tr[:n_train], X_te[:n_test], y_te[:n_test]
