"""Refactored dataset loader for SURGE."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .preprocessing import analyze_dataset_structure, get_dataset_statistics, print_dataset_analysis

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

        self.df = self._read_file(file_path, format=format, **reader_kwargs)
        if sample is not None and 0 < sample < len(self.df):
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
        if self.analysis is None:
            return {}
        return {
            "file_path": str(self.file_path) if self.file_path else "<in-memory>",
            "n_rows": len(self.df) if self.df is not None else 0,
            "n_inputs": len(self.input_columns),
            "n_outputs": len(self.output_columns),
            "output_groups": self.output_groups,
            "profile_groups": self.profile_groups,
            "dataset_info": self.analysis.get("dataset_info"),
        }

    def describe(self) -> str:
        if self.analysis is None:
            return "Dataset not analyzed yet."
        return print_dataset_analysis(self.analysis)

    def stats(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        return get_dataset_statistics(self.df, columns=columns)

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
            fmt = path.suffix.lower().lstrip(".")

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

    def _read_hdf5(self, path: Path, **reader_kwargs: Any) -> pd.DataFrame:
        if "key" in reader_kwargs:
            return pd.read_hdf(path, **reader_kwargs)
        if H5PY_AVAILABLE:
            with h5py.File(path, "r") as handle:  # type: ignore[var-annotated]
                dataset = self._find_first_2d_dataset(handle)
                if dataset is None:
                    raise ValueError("Could not auto-detect a 2D dataset inside HDF5 file.")
                data = dataset[:]
                columns = self._hdf5_columns(dataset)
                return pd.DataFrame(data, columns=columns)
        # Fallback to pandas (will raise helpful error if key missing)
        return pd.read_hdf(path, **reader_kwargs)

    def _find_first_2d_dataset(self, group):
        for key, item in group.items():
            if isinstance(item, h5py.Dataset) and item.ndim == 2:
                return item
            if isinstance(item, h5py.Group):
                found = self._find_first_2d_dataset(item)
                if found is not None:
                    return found
        return None

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

