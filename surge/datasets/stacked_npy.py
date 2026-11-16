"""Stacked .npy directory loader (optional format `stacked_npy` in SurrogateDataset)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from ..dataset import SurrogateDataset


def _resolve_stack_prefix(data_dir: Path, set_name: Optional[str]) -> str:
    """Return the file-name prefix of the stack to load from ``data_dir``.

    A stack is a set of files ``<prefix>_data.npy`` and ``<prefix>_target.npy``
    (optionally ``<prefix>_var_all.npy``). If ``set_name`` is given it is used
    as the prefix; otherwise the directory must hold exactly one stack.
    """
    if set_name:
        return str(set_name)
    prefixes = sorted(p.name[: -len("_data.npy")] for p in data_dir.glob("*_data.npy"))
    if len(prefixes) == 1:
        return prefixes[0]
    if not prefixes:
        raise FileNotFoundError(f"No '<name>_data.npy' file found in {data_dir}")
    raise ValueError(
        f"Several stacks found in {data_dir}: {prefixes}. Pass set_name=<prefix> to choose one."
    )


def _load_stacked_npy(
    data_dir: Union[str, Path],
    set_name: Optional[str] = None,
    sample: Optional[int] = None,
    random_state: int = 42,
    row_range: Optional[tuple[int, int]] = None,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load a stacked-.npy directory into a DataFrame.

    File naming: ``<set_name>_data.npy``, ``<set_name>_target.npy`` and an
    optional ``<set_name>_var_all.npy``. If ``set_name`` is None, the single
    ``*_data.npy`` stack found in the directory is used.

    Parameters
    ----------
    row_range : tuple of (start, end), optional
        Load rows [start:end] instead of random sample. Use for chunk-based
        evaluation (e.g. row_range=(50000, 100000) for a held-out chunk).
    """
    data_dir = Path(data_dir)
    prefix = _resolve_stack_prefix(data_dir, set_name)
    data_path = data_dir / f"{prefix}_data.npy"
    target_path = data_dir / f"{prefix}_target.npy"
    var_all_path = data_dir / f"{prefix}_var_all.npy"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {data_path}. "
            f"Check set_name (the file-name prefix before '_data.npy')."
        )
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    data = np.load(data_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")

    n_samples, n_inputs = data.shape
    _, n_outputs = target.shape

    if row_range is not None:
        start, end = row_range
        data = np.asarray(data[start:end])
        target = np.asarray(target[start:end])
    elif sample is not None and 0 < sample < n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=sample, replace=False)
        data = np.asarray(data[idx])
        target = np.asarray(target[idx])

    input_cols = [f"input_{i}" for i in range(n_inputs)]
    output_cols = [f"output_{i}" for i in range(n_outputs)]

    df_data = pd.DataFrame(data, columns=input_cols)
    df_target = pd.DataFrame(target, columns=output_cols)
    df = pd.concat([df_data, df_target], axis=1)

    if var_all_path.exists():
        var_all = np.load(var_all_path, allow_pickle=True)
        if hasattr(var_all, "tolist"):
            var_all = var_all.tolist()
        # Store in metadata for reference; fewer variable names than input
        # columns suggests each variable spans several slots (e.g. stages)
        pass  # Could add df.attrs["var_all"] = var_all if needed

    return df, input_cols, output_cols


class StackedNpyDataset(SurrogateDataset):
    """
    Stacked .npy directory layout (``format="stacked_npy"`` in ``SurrogateDataset``).

    Constructors:
    - ``from_npy``: explicit path to input and target arrays
    """

    @classmethod
    def from_npy(
        cls,
        data_path: Union[str, Path],
        target_path: Union[str, Path],
        *,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        sample: Optional[int] = None,
        random_state: int = 42,
    ) -> "StackedNpyDataset":
        """
        Load a stacked-npy dataset from generic .npy input and target files.

        Parameters
        ----------
        data_path : str or Path
            Path to input array .npy (shape N x n_inputs)
        target_path : str or Path
            Path to target array .npy (shape N x n_outputs)
        input_names : list of str, optional
            Column names for inputs. If None, use input_0, input_1, ...
        output_names : list of str, optional
            Column names for outputs. If None, use output_0, output_1, ...
        sample : int, optional
            Subsample for testing
        random_state : int
            Random seed for subsampling

        Returns
        -------
        StackedNpyDataset
            Loaded dataset
        """
        data_path = Path(data_path)
        target_path = Path(target_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        data = np.load(data_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")

        n_samples, n_inputs = data.shape
        _, n_outputs = target.shape

        if sample is not None and 0 < sample < n_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n_samples, size=sample, replace=False)
            data = np.asarray(data[idx])
            target = np.asarray(target[idx])

        input_cols = input_names or [f"input_{i}" for i in range(n_inputs)]
        output_cols = output_names or [f"output_{i}" for i in range(n_outputs)]

        if len(input_cols) != n_inputs:
            raise ValueError(
                f"input_names length {len(input_cols)} != data columns {n_inputs}"
            )
        if len(output_cols) != n_outputs:
            raise ValueError(
                f"output_names length {len(output_cols)} != target columns {n_outputs}"
            )

        df_data = pd.DataFrame(data, columns=input_cols)
        df_target = pd.DataFrame(target, columns=output_cols)
        df = pd.concat([df_data, df_target], axis=1)

        dataset = cls()
        dataset.load_from_dataframe(
            df,
            input_columns=input_cols,
            output_columns=output_cols,
        )
        dataset.file_path = data_path
        return dataset
