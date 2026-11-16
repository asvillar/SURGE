"""Stacked .npy directory loader (``format="stacked_npy"``)."""

from __future__ import annotations

import numpy as np

from surge.dataset import SurrogateDataset
from surge.datasets import StackedNpyDataset


def _write_stack(d, n=20, n_in=4, n_out=2):
    rng = np.random.default_rng(0)
    np.save(d / "stack_a_data.npy", rng.normal(size=(n, n_in)))
    np.save(d / "stack_a_target.npy", rng.normal(size=(n, n_out)))


def test_from_npy(tmp_path):
    _write_stack(tmp_path)
    ds = StackedNpyDataset.from_npy(
        tmp_path / "stack_a_data.npy", tmp_path / "stack_a_target.npy"
    )
    assert len(ds.df) == 20
    assert "input_3" in ds.df.columns and "output_1" in ds.df.columns


def test_surrogate_dataset_format_string(tmp_path):
    _write_stack(tmp_path)
    ds = SurrogateDataset.from_path(tmp_path, format="stacked_npy")
    assert len(ds.df) == 20
    assert ds.df.shape[1] == 6


def test_set_name_selects_among_stacks(tmp_path):
    _write_stack(tmp_path)
    rng = np.random.default_rng(1)
    np.save(tmp_path / "stack_b_data.npy", rng.normal(size=(7, 3)))
    np.save(tmp_path / "stack_b_target.npy", rng.normal(size=(7, 1)))
    ds = SurrogateDataset.from_path(tmp_path, format="stacked_npy", set_name="stack_b")
    assert ds.df.shape == (7, 4)


def test_multiple_stacks_without_set_name_is_an_error(tmp_path):
    import pytest

    _write_stack(tmp_path)
    np.save(tmp_path / "stack_b_data.npy", np.zeros((3, 2)))
    np.save(tmp_path / "stack_b_target.npy", np.zeros((3, 1)))
    with pytest.raises(ValueError, match="set_name"):
        SurrogateDataset.from_path(tmp_path, format="stacked_npy")
