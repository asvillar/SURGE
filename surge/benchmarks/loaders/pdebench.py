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

HDF5 file structure (from pdebench/PDEBench source)
----------------------------------------------------
1D Burgers (``1D_Burgers_Sols_Nu*.hdf5``, extension ``.hdf5``):
  - ``f["tensor"]``        → (N, T, nx)  — solution u(x, t)
  - ``f["x-coordinate"]``  → (nx,)

2D Darcy Flow (``2D_DarcyFlow_beta*.hdf5``, extension ``.hdf5``):
  - ``f["tensor"]``        → (N, 1, nx, ny) — solution u(x, y)
  - ``f["nu"]``            → (N, nx, ny)    — permeability a(x, y)
  - Task: predict u from a  (IC → solution)

2D Shallow Water (``2D_rdb_NA_NA.h5``, extension ``.h5``):
  - Top-level keys are trajectory IDs: ``"0001"``, ``"0002"``, …
  - ``f[key]["data"]``     → (T, nx, ny, nc)  — per-trajectory fields
  - ``f["0001"]["grid"]["x"]`` etc. — coordinate grids

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
        # Direct file ID from pdebench/PDEBench pdebench_data_urls.csv
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/281363",
        "n_samples_train": 900,
        "n_samples_test": 100,
        "description": "1D Burgers ν=0.01, shape (N, T=101, nx=1024)",
        "loader": "_load_burgers_hdf5",
    },
    "darcy_2d": {
        "filename": "2D_DarcyFlow_beta1.0_Train.hdf5",
        # Direct file ID from pdebench/PDEBench pdebench_data_urls.csv
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/133219",
        "n_samples_train": 900,
        "n_samples_test": 100,
        "description": "2D Darcy Flow β=1.0, 128×128 grid",
        "loader": "_load_darcy_hdf5",
    },
    "shallow_water_2d": {
        "filename": "2D_rdb_NA_NA.h5",
        # Direct file ID from pdebench/PDEBench pdebench_data_urls.csv
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/133021",
        "n_samples_train": 900,
        "n_samples_test": 100,
        "description": "2D Shallow Water Equations (radial dam-break), 128×128",
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
    reduced_resolution: int = 1,
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
        Limit the number of samples loaded.
    reduced_resolution:
        Spatial downsampling factor (1 = full resolution).  Use 4 or 8 to
        reduce 128×128 → 32×32 or 16×16 for faster tabular-model baselines.
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

    loader_fn = globals()[meta["loader"]]
    X_train, y_train, X_test, y_test = loader_fn(
        fpath,
        n_train=train_samples or meta["n_samples_train"],
        n_test=test_samples or meta["n_samples_test"],
        reduced_resolution=reduced_resolution,
        seed=seed,
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    """Download *url* → *dest*, following HTTP 303 redirects (DaRUS → S3)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _LOG.info("Downloading %s → %s", url, dest)
    size_hint = _DATASETS.get(_key_for_filename(dest.name), {}).get("description", "")
    print(f"[pdebench] Downloading {dest.name}  ({size_hint}) …")
    print(f"[pdebench] This may take a few minutes — files are several hundred MB.")
    try:
        # urllib follows redirects automatically; DaRUS returns 303 → S3 presigned URL
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"[pdebench] ✓ Saved {dest.name}  ({size_mb:.0f} MB)")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Download failed for {url!r}.\n"
            "You can manually download the file and place it at:\n"
            f"  {dest}\n"
            "PDEBench data: https://darus.uni-stuttgart.de/dataverse/pdebench\n"
            "Direct file IDs in: https://github.com/pdebench/PDEBench/blob/main/"
            "pdebench/data_download/pdebench_data_urls.csv"
        ) from exc


def _key_for_filename(filename: str) -> str:
    """Reverse-lookup dataset key by filename (for download messages)."""
    for k, v in _DATASETS.items():
        if v["filename"] == filename:
            return k
    return ""


# ---------------------------------------------------------------------------
# Dataset-specific HDF5 loaders
# ---------------------------------------------------------------------------


def _load_burgers_hdf5(
    path: Path, *, n_train: int, n_test: int, reduced_resolution: int = 1, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Burgers 1D (file extension ``.hdf5``).

    HDF5 layout (from PDEBench source)::

        f["tensor"]       → (N, T, nx)   # u(x, t)  N=10000, T=201, nx=1024
        f["x-coordinate"] → (nx,)

    Task: predict u(x, T_last) from u(x, 0)  →  IC → final state.
    X shape: (N, nx//r),  y shape: (N, nx//r).
    """
    import h5py
    from sklearn.model_selection import train_test_split

    r = max(1, int(reduced_resolution))
    with h5py.File(path, "r") as f:
        data = f["tensor"][:n_train + n_test, :, ::r]  # (n_use, T, nx//r)

    N, T, nx = data.shape
    data = data.astype(np.float32)
    X = data[:, 0, :]   # IC:          (N, nx//r)
    y = data[:, -1, :]  # Final state: (N, nx//r)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test / N, random_state=seed
    )
    return X_tr[:n_train], y_tr[:n_train], X_te[:n_test], y_te[:n_test]


def _load_darcy_hdf5(
    path: Path, *, n_train: int, n_test: int, reduced_resolution: int = 1, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Darcy Flow 2D (file extension ``.hdf5``).

    HDF5 layout (from PDEBench source)::

        f["tensor"]       → (N, 1, nx, ny)  # solution u(x, y)  nx=ny=128
        f["nu"]           → (N, nx, ny)     # permeability a(x, y)
        f["x-coordinate"] → (nx,)
        f["y-coordinate"] → (ny,)

    Task: predict u from a  →  flatten both to (N, (nx//r)*(ny//r)).
    ``reduced_resolution=4`` → 32×32=1024 features (RF-friendly baseline).
    """
    import h5py
    from sklearn.model_selection import train_test_split

    r = max(1, int(reduced_resolution))
    n_use = n_train + n_test
    with h5py.File(path, "r") as f:
        u = f["tensor"][:n_use, :, ::r, ::r]  # (n_use, 1, nx//r, ny//r)
        a = f["nu"][:n_use, ::r, ::r]          # (n_use, nx//r, ny//r)

    # Squeeze singleton time dim
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]
    elif u.ndim == 4:
        u = u[:, -1, :, :]

    N = a.shape[0]
    X = a.reshape(N, -1).astype(np.float32)
    y = u.reshape(N, -1).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test / N, random_state=seed
    )
    return X_tr[:n_train], y_tr[:n_train], X_te[:n_test], y_te[:n_test]


def _load_shallow_water_hdf5(
    path: Path, *, n_train: int, n_test: int, reduced_resolution: int = 1, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shallow Water Equations 2D (file extension ``.h5``).

    HDF5 layout (from PDEBench source)::

        Top-level keys are zero-padded trajectory IDs: "0000", "0001", …
        f[key]["data"] → (T, nx, ny, nc)   # T=101, nx=ny=128, nc=1

    Task: predict final-step height h(x, y, T) from initial h(x, y, 0).
    Channel 0 = water height.  Flatten spatial → (N, (nx//r)*(ny//r)).
    """
    import h5py
    from sklearn.model_selection import train_test_split

    r = max(1, int(reduced_resolution))
    n_use = n_train + n_test
    with h5py.File(path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if "data" in f[k])
        traj_keys = traj_keys[:n_use]

        frames = []
        for k in traj_keys:
            arr = f[k]["data"][::1, ::r, ::r, :]  # (T, nx//r, ny//r, nc)
            frames.append(arr)

    # Stack → (N, T, nx//r, ny//r, nc)
    data = np.stack(frames, axis=0).astype(np.float32)
    N, T, nx, ny, nc = data.shape

    h = data[:, :, :, :, 0]               # (N, T, nx//r, ny//r)
    X = h[:, 0, :, :].reshape(N, -1)      # IC
    y = h[:, -1, :, :].reshape(N, -1)     # final state

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test / N, random_state=seed
    )
    return X_tr[:n_train], y_tr[:n_train], X_te[:n_test], y_te[:n_test]
