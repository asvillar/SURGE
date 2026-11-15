"""Refactored dataset loader for SURGE."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .preprocessing import analyze_dataset_structure, get_dataset_statistics, print_dataset_analysis
from .utils import posix_str

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False

try:  # pragma: no cover
    import h5py  # type: ignore

    H5PY_AVAILABLE = True
except Exception:  # pragma: no cover
    H5PY_AVAILABLE = False

try:  # pragma: no cover
    import xarray as xr  # type: ignore

    XARRAY_AVAILABLE = True
except Exception:  # pragma: no cover
    XARRAY_AVAILABLE = False

# PyTorch and TensorFlow are NOT imported at module load time.
# They are imported lazily inside to_dataloader() and to_tf_dataset() only when
# those methods are called. This keeps SurrogateDataset framework-agnostic:
# you can import and use it without any ML framework installed.
# Use iter_batches() for framework-agnostic numpy batching.


DataLike = Union[str, Path, pd.DataFrame]


class SurrogateDataset:
    """
    General-purpose dataset loader with automatic input/output detection.

    The refactored loader extends the legacy implementation with metadata-driven
    overrides, NetCDF support, and sampling utilities tailored to large
    scientific surrogate datasets.
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.input_columns: List[str] = []
        self.output_columns: List[str] = []
        self.output_groups: Dict[str, List[str]] = {}
        self.profile_groups: Dict[str, List[str]] = {}
        self.analysis: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}
        self.file_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: DataLike, **kwargs: Any) -> "SurrogateDataset":
        dataset = cls()
        dataset.load_from_path(path, **kwargs)
        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        input_columns: Optional[Sequence[str]] = None,
        output_columns: Optional[Sequence[str]] = None,
        analyzer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SurrogateDataset":
        dataset = cls()
        dataset.load_from_dataframe(
            df,
            input_columns=input_columns,
            output_columns=output_columns,
            analyzer_kwargs=analyzer_kwargs,
        )
        return dataset

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------
    def load_from_path(
        self,
        path: DataLike,
        *,
        format: str = "auto",
        metadata_path: Optional[Union[str, Path]] = None,
        sample: Optional[int] = None,
        sample_random_state: int = 42,
        analyzer_kwargs: Optional[Dict[str, Any]] = None,
        **reader_kwargs: Any,
    ) -> Tuple[List[str], List[str]]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        self.file_path = file_path

        if metadata_path:
            self.metadata = self._load_metadata(metadata_path)
        else:
            self.metadata = {}

        # Build reader_kwargs; for xgc format, pass sample and hints
        _reader_kwargs = dict(reader_kwargs)
        fmt_lower = (format or "").lower()
        is_xgc = fmt_lower == "xgc" or (fmt_lower in ("auto", "") and file_path.is_dir())
        if is_xgc:
            _reader_kwargs.setdefault("sample", sample)
            if analyzer_kwargs and "hints" in analyzer_kwargs:
                _reader_kwargs.update(analyzer_kwargs["hints"])

        self.df = self._read_file(file_path, format=format, **_reader_kwargs)

        # Subsample only if not already done by format-specific loader (e.g. xgc)
        if sample is not None and 0 < sample < len(self.df):
            if not is_xgc:
                self.df = self.df.sample(n=sample, random_state=sample_random_state)

        self._analyze(
            analyzer_kwargs=analyzer_kwargs,
        )
        return self.input_columns, self.output_columns

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        input_columns: Optional[Sequence[str]] = None,
        output_columns: Optional[Sequence[str]] = None,
        analyzer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        self.df = df.copy(deep=False)
        self.file_path = None

        if input_columns and output_columns:
            self.input_columns = list(input_columns)
            self.output_columns = list(output_columns)
            self.output_groups = {}
            self.profile_groups = {}
            self.analysis = None
        else:
            self._analyze(analyzer_kwargs=analyzer_kwargs)
        return self.input_columns, self.output_columns

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        base = {
            "file_path": posix_str(self.file_path) if self.file_path else "<in-memory>",
            "n_rows": len(self.df) if self.df is not None else 0,
            "n_inputs": len(self.input_columns),
            "n_outputs": len(self.output_columns),
        }
        if self.analysis is None:
            return base
        base["output_groups"] = self.output_groups
        base["profile_groups"] = self.profile_groups
        base["dataset_info"] = self.analysis.get("dataset_info")
        return base

    def describe(self) -> str:
        if self.analysis is None:
            return "Dataset not analyzed yet."
        return print_dataset_analysis(self.analysis)

    def stats(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        return get_dataset_statistics(self.df, columns=columns)

    def iter_batches(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Generator yielding (X_batch, y_batch) as numpy arrays. Framework-agnostic.

        Works with PyTorch, TensorFlow, or any framework that accepts numpy.
        No framework imports required.

        Parameters
        ----------
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle before iterating
        random_state : int, optional
            Seed for reproducible shuffling

        Yields
        ------
        X_batch : np.ndarray
            Input batch (n_batch, n_inputs)
        y_batch : np.ndarray
            Output batch (n_batch, n_outputs)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_from_path or load_from_dataframe first.")
        if not self.input_columns or not self.output_columns:
            raise ValueError("No input or output columns set.")

        import numpy as np

        X = self.df[self.input_columns].values.astype(np.float32)
        y = self.df[self.output_columns].values.astype(np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n = len(X)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "Any":
        """
        Create a PyTorch DataLoader from the dataset.

        For framework-agnostic batching, use iter_batches() instead.
        Requires PyTorch and a loaded DataFrame. Yields (X, y) batches
        suitable for training loops.

        Parameters
        ----------
        batch_size : int
            Batch size for the DataLoader
        shuffle : bool
            Whether to shuffle the data
        num_workers : int
            Number of worker processes (0 = main process only)
        device : str, optional
            Device to place tensors on ('cpu', 'cuda'). If None, keeps on CPU.
        **kwargs
            Additional arguments passed to DataLoader (e.g. pin_memory)

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding (X_batch, y_batch) tensors

        Raises
        ------
        ImportError
            If PyTorch is not installed
        ValueError
            If dataset is not loaded
        """
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch is required for to_dataloader(). Install with: pip install torch"
            )
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_from_path or load_from_dataframe first.")
        if not self.input_columns or not self.output_columns:
            raise ValueError("No input or output columns set.")

        X = self.df[self.input_columns].values.astype("float32")
        y = self.df[self.output_columns].values.astype("float32")
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        if device is not None:
            X_tensor = X_tensor.to(device)
            y_tensor = y_tensor.to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        **kwargs: Any,
    ) -> "Any":
        """
        Create a TensorFlow tf.data.Dataset from the dataset.

        Requires TensorFlow. Returns a batched, optionally shuffled dataset
        suitable for model.fit() or custom training loops.

        Parameters
        ----------
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle (uses shuffle_buffer_size)
        shuffle_buffer_size : int, optional
            Buffer size for shuffling. If None, uses dataset size.
        **kwargs
            Additional arguments passed to tf.data.Dataset.batch()

        Returns
        -------
        tf.data.Dataset
            Batched dataset yielding (X_batch, y_batch) tensors
        """
        try:
            import tensorflow as tf  # type: ignore
        except ImportError:
            raise ImportError(
                "TensorFlow is required for to_tf_dataset(). Install with: pip install tensorflow"
            )
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_from_path or load_from_dataframe first.")
        if not self.input_columns or not self.output_columns:
            raise ValueError("No input or output columns set.")

        X = self.df[self.input_columns].values.astype("float32")
        y = self.df[self.output_columns].values.astype("float32")
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            buf = shuffle_buffer_size if shuffle_buffer_size is not None else len(X)
            ds = ds.shuffle(buffer_size=buf)
        ds = ds.batch(batch_size, **kwargs)
        return ds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _analyze(self, analyzer_kwargs: Optional[Dict[str, Any]]) -> None:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        analyzer_kwargs = analyzer_kwargs or {}
        self.analysis = analyze_dataset_structure(
            self.df,
            metadata=self.metadata,
            hints=analyzer_kwargs.pop("hints", None),
            sample_size_for_stats=analyzer_kwargs.pop("sample_size_for_stats", None),
        )
        self.input_columns = self.analysis["input_variables"]
        self.output_columns = self.analysis["output_variables"]
        self.output_groups = self.analysis.get("output_groups", {})
        self.profile_groups = self.analysis.get("profile_groups", {})

        manual_inputs = self.metadata.get("inputs")
        if manual_inputs:
            self.input_columns = [
                col for col in manual_inputs if col in self.df.columns
            ]
        manual_outputs = self.metadata.get("outputs")
        if manual_outputs:
            self.output_columns = [
                col for col in manual_outputs if col in self.df.columns
            ]

    def _read_file(self, path: Path, *, format: str, **reader_kwargs: Any) -> pd.DataFrame:
        fmt = (format or "").lower()
        if fmt == "auto" or not fmt:
            if path.is_dir():
                fmt = "xgc"
            else:
                fmt = path.suffix.lower().lstrip(".")

        if fmt == "xgc":
            return self._read_xgc(path, **reader_kwargs)
        if fmt == "csv":
            return pd.read_csv(path, **reader_kwargs)
        if fmt in ("pkl", "pickle"):
            _ensure_numpy_core_alias()
            return pd.read_pickle(path)
        if fmt in ("parquet", "pq"):
            return pd.read_parquet(path, **reader_kwargs)
        if fmt in ("xlsx", "xls"):
            return pd.read_excel(path, **reader_kwargs)
        if fmt in ("json",):
            return pd.read_json(path, **reader_kwargs)
        if fmt in ("h5", "hdf5"):
            return self._read_hdf5(path, **reader_kwargs)
        if fmt in ("nc", "netcdf"):
            return self._read_netcdf(path, **reader_kwargs)
        raise ValueError(f"Unsupported dataset format '{fmt}' for file {path}")

    def _read_xgc(self, path: Path, **reader_kwargs: Any) -> pd.DataFrame:
        """Load stacked-directory .npy layout via XGCDataset (see ``surge.datasets.xgc``)."""
        from .datasets.xgc import _load_stacked_npy

        if not path.is_dir():
            raise ValueError(f"XGC format requires a directory path, got: {path}")

        set_name = reader_kwargs.get("set_name", "set1")
        sample = reader_kwargs.get("sample")
        random_state = reader_kwargs.get("random_state", 42)
        row_range = reader_kwargs.get("row_range")

        df, _input_cols, _output_cols = _load_stacked_npy(
            path,
            set_name=set_name,
            sample=sample,
            random_state=random_state,
            row_range=row_range,
        )
        return df

    def _read_hdf5(self, path: Path, **reader_kwargs: Any) -> pd.DataFrame:
        if "key" in reader_kwargs:
            return pd.read_hdf(path, **reader_kwargs)
        if H5PY_AVAILABLE:
            with h5py.File(path, "r") as handle:  # type: ignore[var-annotated]
                found: List[Any] = []
                dataset = self._find_first_2d_dataset(handle, collect=found)
                if dataset is None:
                    raise ValueError("Could not auto-detect a 2D dataset inside HDF5 file.")
                if len(found) > 1:
                    logger.warning(
                        "HDF5 file %s holds %d 2-D datasets (%s); using '%s'. "
                        "Pass key=... to choose another.",
                        path, len(found), ", ".join(d.name for d in found), dataset.name,
                    )
                limit_gb = _max_hdf5_gb()
                n_bytes = 1
                for dim in dataset.shape:
                    n_bytes *= int(dim)
                n_bytes *= int(dataset.dtype.itemsize)
                size_gb = n_bytes / 1024**3
                if size_gb > limit_gb:
                    raise ValueError(
                        f"HDF5 dataset '{dataset.name}' has shape {tuple(dataset.shape)} "
                        f"and would need {size_gb:.2f} GB in memory (limit {limit_gb:g} GB). "
                        "Tabulate a sample of the rows into a table (CSV/Parquet) first, "
                        "or raise the limit with the SURGE_MAX_HDF5_GB environment variable."
                    )
                data = dataset[:]
                columns = self._hdf5_columns(dataset)
                return pd.DataFrame(data, columns=columns)
        # Fallback to pandas (will raise helpful error if key missing)
        return pd.read_hdf(path, **reader_kwargs)

    def _find_first_2d_dataset(self, group, collect: Optional[List[Any]] = None):
        """Return the first 2-D dataset; if ``collect`` is given, append all of them."""
        first = None
        for key, item in group.items():
            if isinstance(item, h5py.Dataset) and item.ndim == 2:
                if collect is None:
                    return item
                collect.append(item)
                first = first or item
            elif isinstance(item, h5py.Group):
                found = self._find_first_2d_dataset(item, collect)
                first = first or found
                if found is not None and collect is None:
                    return found
        return first

    def _hdf5_columns(self, dataset) -> List[str]:
        if "column_names" in dataset.attrs:
            names = dataset.attrs["column_names"]
            return [
                name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
                for name in names
            ]
        return [f"col_{idx}" for idx in range(dataset.shape[1])]

    def _read_netcdf(self, path: Path, **reader_kwargs: Any) -> pd.DataFrame:
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for NetCDF support. Install with `pip install xarray`.")
        dataset = xr.open_dataset(path, **reader_kwargs)  # type: ignore[name-defined]
        try:
            df = dataset.to_dataframe().reset_index()
        finally:
            dataset.close()
        return df

    def _load_metadata(self, metadata_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        if path.suffix.lower() in (".yml", ".yaml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required to parse metadata YAML files.")
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        raise ValueError(f"Unsupported metadata format: {path.suffix}")


def _max_hdf5_gb() -> float:
    raw = os.environ.get("SURGE_MAX_HDF5_GB")
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("Ignoring invalid SURGE_MAX_HDF5_GB=%r", raw)
    return 2.0


def _ensure_numpy_core_alias() -> None:
    """
    Allow reading pickle files created with NumPy 2.x when running under NumPy 1.x.

    NumPy 2 renamed the private package ``numpy.core`` to ``numpy._core``. Pickle
    files generated with NumPy 2 therefore reference modules such as
    ``numpy._core.numeric`` which do not exist in NumPy 1.x. We alias the legacy
    modules so the pickle loader can resolve them without requiring a full
    environment upgrade (which would conflict with TensorFlow/Gpflow pins).
    """

    try:
        numpy_core = importlib.import_module("numpy.core")
    except ImportError:  # pragma: no cover - NumPy missing
        return

    sys.modules["numpy._core"] = numpy_core

    submodules = [
        "numeric",
        "multiarray",
        "fromnumeric",
        "umath",
        "_multiarray_umath",
        "_dtype_like",
        "_exceptions",
        "_multiarray_tests",
    ]
    for name in submodules:
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)

