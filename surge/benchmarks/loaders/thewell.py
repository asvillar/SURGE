"""TheWell dataset loader for SURGE.

TheWell (Ohana et al., NeurIPS 2024 D&B) is a large-scale collection of
physics-based PDE datasets totalling ~15 TB across 16 distinct physics
systems — MHD turbulence, Rayleigh-Bénard convection, Gray-Scott
reaction-diffusion, supernova remnants, etc.

Homepage: https://github.com/PolymathicAI/the_well
Paper:    https://arxiv.org/abs/2412.00568

Status: **Tier 4 — future, manual-trigger only**.  Not included in the
default benchmark run (``--tier 0/1/2/3``).  Run explicitly by key::

    python -m surge.benchmarks.run --benchmark thewell.gray_scott

Installation
------------
    pip install the-well            # base; installs WellDataset + well_download
    pip install the-well[benchmark] # also installs neuralop, einops, metrics

Data
----
Data is NOT bundled with the package.  Download a dataset once with::

    from the_well.utils.download import well_download
    well_download(base_path="~/.surge/data/thewell", dataset="gray_scott_reaction_diffusion", split="train")
    well_download(base_path="~/.surge/data/thewell", dataset="gray_scott_reaction_diffusion", split="valid")

or let :func:`load_thewell` handle it automatically (``download=True``).

WellDataset API (v1.2)
----------------------
``WellDataset`` is a map-style ``torch.utils.data.Dataset``.  Each item is
a dict with keys:

  ``input_fields``   — torch.Tensor  shape ``(T_in, Lx[, Ly[, Lz]], F)``
  ``output_fields``  — torch.Tensor  shape ``(T_out, Lx[, Ly[, Lz]], F)``
  ``space_grid``     — coordinate grids (optional)
  ``boundary_conditions`` — BC masks (optional)

SURGE interface
---------------
:func:`load_thewell` returns ``(X_train, y_train, X_test, y_test)`` where
each array is 2-D ``(n_samples, n_features)`` — ``input_fields`` and
``output_fields`` are each flattened across all spatial, temporal and field
dimensions.  For large grids this yields very high-dimensional arrays; the
FNO/CNN adapters are the intended models for these benchmarks.

Reference
---------
Ohana, R. et al. (2024). The Well: a Large-Scale Collection of Diverse
Physics Simulations for Machine Learning.  NeurIPS 2024 Datasets and
Benchmarks.  arXiv:2412.00568.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("surge.benchmarks.thewell")

# well_download(base_path=...) writes under <base_path>/datasets/<well_name>/data/<split>/
_DOWNLOAD_ROOT = Path.home() / ".surge" / "data" / "thewell"


def _download_root(cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir) if cache_dir is not None else _DOWNLOAD_ROOT


def _well_base_path(cache_dir: Path | str | None = None) -> Path:
    """Directory passed to WellDataset as ``well_base_path``."""
    return _download_root(cache_dir) / "datasets"

THEWELL_AVAILABLE = False
try:
    from the_well.data import WellDataset  # noqa: F401
    THEWELL_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Dataset registry
# Keys: SURGE key → TheWell dataset name (passed to WellDataset / well_download)
# ---------------------------------------------------------------------------

_DATASETS: dict[str, dict[str, Any]] = {
    "gray_scott": {
        "well_name": "gray_scott_reaction_diffusion",
        "description": "Gray-Scott reaction-diffusion 2D (~132 GB: 117 train + 15 valid)",
        "shape_hint": "(T, Lx, Ly, 2 fields) → next timestep",
        "n_steps_input": 4,
        "n_steps_output": 1,
    },
    "turbulence_2d": {
        "well_name": "turbulent_radiative_layer_2D",
        "description": "Turbulent radiative layer 2D",
        "shape_hint": "(T, 128, 384, 4 fields) → next timestep",
        "n_steps_input": 4,
        "n_steps_output": 1,
    },
    "helmholtz": {
        "well_name": "helmholtz_staircase",
        "description": "Helmholtz acoustics over a staircase (2D, ~80 GB)",
        "shape_hint": "(T, 1024, 256, 2 fields: Re/Im pressure) → next timestep",
        "n_steps_input": 4,
        "n_steps_output": 1,
    },
    "mhd": {
        "well_name": "MHD_64",
        "description": "3D MHD turbulence 64³",
        "shape_hint": "(T, 64, 64, 64, 7 fields) → next timestep",
        "n_steps_input": 4,
        "n_steps_output": 1,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_thewell_datasets() -> list[str]:
    """Return supported TheWell SURGE keys."""
    return sorted(_DATASETS)


def is_cached(key: str, *, cache_dir: Path | str | None = None) -> bool:
    """Return True if the dataset has already been downloaded."""
    if key not in _DATASETS:
        raise KeyError(f"Unknown TheWell dataset {key!r}.")
    well_name = _DATASETS[key]["well_name"]
    data_dir = _well_base_path(cache_dir) / well_name / "data"
    return (data_dir / "train").exists() or (data_dir / "valid").exists()


def download_thewell(
    key: str,
    *,
    splits: list[str] | None = None,
    cache_dir: Path | str | None = None,
) -> None:
    """
    Download a TheWell dataset using ``well_download``.

    Parameters
    ----------
    key:
        SURGE dataset key (e.g. ``"gray_scott"``).
    splits:
        Which splits to download.  Defaults to ``["train", "valid"]``.
    cache_dir:
        Override the default ``~/.surge/data/thewell/`` cache directory.
    """
    if not THEWELL_AVAILABLE:
        raise ImportError(
            "the_well is required.\n"
            "Install: pip install the-well\n"
            "Or:      pip install 'surge-ml[thewell]'"
        )
    from the_well.utils.download import well_download

    root = _download_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    well_name = _DATASETS[key]["well_name"]
    for split in (splits or ["train", "valid"]):
        _LOG.info("Downloading TheWell %s/%s → %s", well_name, split, root)
        print(f"[thewell] Downloading {well_name}/{split} …")
        well_download(base_path=str(root), dataset=well_name, split=split)


def load_thewell(
    key: str,
    *,
    n_train: int = 500,
    n_test: int = 100,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a TheWell dataset, downloading on first use.

    Uses :class:`the_well.data.WellDataset` (map-style PyTorch Dataset).
    Each sample's ``input_fields`` and ``output_fields`` tensors are
    flattened to 1-D and stacked into 2-D numpy arrays.

    Parameters
    ----------
    key:
        One of the keys from :func:`list_thewell_datasets`.
    n_train, n_test:
        Number of samples drawn from the ``train`` / ``valid`` splits.
    seed:
        Random seed for sample selection.
    cache_dir:
        Override the default ``~/.surge/data/thewell/`` cache directory.
    download:
        If True (default) and data is absent, call :func:`download_thewell`
        automatically.  Set to False to fail fast if data is missing.

    Returns
    -------
    ``(X_train, y_train, X_test, y_test)`` — 2-D float32 numpy arrays.
    """
    if not THEWELL_AVAILABLE:
        raise ImportError(
            "the_well is required for TheWell benchmarks.\n"
            "Install: pip install the-well\n"
            "Or:      pip install 'surge-ml[thewell]'"
        )
    if key not in _DATASETS:
        raise KeyError(f"Unknown TheWell dataset {key!r}. Use list_thewell_datasets().")

    from the_well.data import WellDataset

    well_root = _well_base_path(cache_dir)
    meta = _DATASETS[key]
    well_name = meta["well_name"]

    # Download train + valid splits if absent.
    for split, needed in [("train", n_train), ("valid", n_test)]:
        if needed <= 0:
            continue
        split_path = well_root / well_name / "data" / split
        if not split_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"TheWell data not found at {split_path}.\n"
                    "Run with download=True or call download_thewell(key) manually."
                )
            download_thewell(key, splits=[split], cache_dir=_download_root(cache_dir))

    rng = np.random.default_rng(seed)

    def _assert_split_hdf5_ready(split: str) -> None:
        split_path = well_root / well_name / "data" / split
        h5_files = sorted(split_path.glob("*.hdf5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files in {split_path}")
        import h5py
        bad: list[str] = []
        for path in h5_files:
            try:
                with h5py.File(path, "r"):
                    pass
            except OSError:
                bad.append(path.name)
        if bad:
            raise RuntimeError(
                f"TheWell {well_name}/{split}: incomplete download "
                f"({len(bad)} corrupt file(s), e.g. {bad[0]}). "
                "Wait for download_thewell() to finish or re-run it."
            )

    def _collect(split: str, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if n_samples <= 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
        _assert_split_hdf5_ready(split)
        ds = WellDataset(
            well_base_path=str(well_root),
            well_dataset_name=well_name,
            well_split_name=split,
            n_steps_input=meta["n_steps_input"],
            n_steps_output=meta["n_steps_output"],
            use_normalization=False,
        )
        total = len(ds)
        if total == 0:
            raise RuntimeError(f"WellDataset {well_name}/{split} has 0 samples.")
        n = min(n_samples, total)
        idx = rng.choice(total, size=n, replace=False)
        Xs, ys = [], []
        for i in idx:
            item = ds[int(i)]
            # input_fields: (T_in, Lx, ..., F) — flatten entirely.
            x_flat = item["input_fields"].numpy().ravel().astype("float32")
            y_flat = item["output_fields"].numpy().ravel().astype("float32")
            Xs.append(x_flat)
            ys.append(y_flat)
        return np.array(Xs), np.array(ys)

    X_train, y_train = _collect("train", n_train)
    X_test, y_test = _collect("valid", n_test)
    return X_train, y_train, X_test, y_test
