"""Stacked .npy directory loader (``format="stacked_npy"``)."""

from __future__ import annotations

import numpy as np

from surge.dataset import SurrogateDataset
from surge.datasets import StackedNpyDataset


def _write_stack(d, n=20, n_in=4, n_out=2):
    rng = np.random.default_rng(0)
    np.save(d / "data_nprev5_set1_data.npy", rng.normal(size=(n, n_in)))
    np.save(d / "data_nprev5_set1_target.npy", rng.normal(size=(n, n_out)))


def test_from_npy(tmp_path):
    _write_stack(tmp_path)
    ds = StackedNpyDataset.from_npy(
        tmp_path / "data_nprev5_set1_data.npy", tmp_path / "data_nprev5_set1_target.npy"
    )
    assert len(ds.df) == 20
    assert "input_3" in ds.df.columns and "output_1" in ds.df.columns


def test_surrogate_dataset_format_string(tmp_path):
    _write_stack(tmp_path)
    ds = SurrogateDataset.from_path(tmp_path, format="stacked_npy")
    assert len(ds.df) == 20
    assert ds.df.shape[1] == 6
