"""Column-role authority, non-numeric handling and HDF5 size guard."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surge import SurrogateWorkflowSpec, run_surrogate_workflow
from surge.dataset import SurrogateDataset
from surge.preprocessing import analyze_dataset_structure


def _frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "extra_num": rng.normal(size=n),
            "label": ["foo", "bar"] * (n // 2),
        }
    )
    df["target"] = 0.5 * df["a"] - 0.25 * df["b"]
    return df


def _run(tmp_path: Path, df: pd.DataFrame, overrides: dict, tag: str) -> dict:
    csv = tmp_path / f"{tag}.csv"
    df.to_csv(csv, index=False)
    spec = SurrogateWorkflowSpec.from_dict(
        {
            "dataset_path": str(csv),
            "models": [{"key": "sklearn.ridge", "name": "ridge"}],
            "output_dir": str(tmp_path),
            "run_tag": tag,
            "overwrite_existing_run": True,
            "metadata_overrides": overrides,
        }
    )
    return run_surrogate_workflow(spec)


def test_both_lists_are_authoritative(tmp_path: Path) -> None:
    df = _frame()
    res = analyze_dataset_structure(
        df, hints={"inputs": ["a", "b", "c"], "outputs": ["target"]}
    )
    assert res["input_variables"] == ["a", "b", "c"]
    assert res["output_variables"] == ["target"]

    summary = _run(tmp_path, df, {"inputs": ["a", "b", "c"], "outputs": ["target"]}, "auth")
    text = json.dumps(summary, default=str)
    assert "extra_num" not in text and "label" not in text
    root = Path(summary["artifacts"]["root"])
    assert (root / "metrics.json").is_file()


def test_only_outputs_skips_text_with_warning(tmp_path: Path, caplog) -> None:
    df = _frame()
    with caplog.at_level(logging.WARNING, logger="surge.preprocessing"):
        res = analyze_dataset_structure(df, hints={"outputs": ["target"]})
    assert "label" not in res["input_variables"]
    assert {"a", "b", "c", "extra_num"} <= set(res["input_variables"])
    assert any("label" in r.getMessage() for r in caplog.records)

    summary = _run(tmp_path, df, {"outputs": ["target"]}, "outonly")
    assert (Path(summary["artifacts"]["root"]) / "metrics.json").is_file()


def test_text_column_listed_as_input_raises() -> None:
    df = _frame()
    with pytest.raises(ValueError, match=r"label.*numeric"):
        analyze_dataset_structure(df, hints={"inputs": ["a", "label"], "outputs": ["target"]})


def test_dataset_hints_end_to_end_columns() -> None:
    ds = SurrogateDataset.from_dataframe(
        _frame(),
        analyzer_kwargs={"hints": {"inputs": ["a", "b"], "outputs": ["target"]}},
    )
    assert ds.input_columns == ["a", "b"]


def test_hdf5_size_guard(tmp_path: Path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "big.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("grp/table", data=np.zeros((1000, 50)))
    monkeypatch.setenv("SURGE_MAX_HDF5_GB", "1e-9")
    with pytest.raises(ValueError, match="/grp/table"):
        SurrogateDataset.from_path(path, format="h5")


def test_hdf5_small_file_loads(tmp_path: Path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    monkeypatch.delenv("SURGE_MAX_HDF5_GB", raising=False)
    path = tmp_path / "small.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("table", data=np.ones((10, 3)))
    ds = SurrogateDataset()
    df = ds._read_hdf5(path)
    assert df.shape == (10, 3)


def test_pattern_group_does_not_reclaim_listed_input() -> None:
    n = 20
    df = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "out_0": np.arange(n, dtype=float),
            "out_1": np.arange(n, dtype=float),
            "target": np.arange(n, dtype=float),
        }
    )
    res = analyze_dataset_structure(
        df, hints={"inputs": ["x", "out_0"], "outputs": ["target"]}
    )
    assert res["input_variables"] == ["out_0", "x"]
    assert res["output_variables"] == ["target"]
    assert "out" not in res["output_groups"]


def test_declared_output_group_survives_authoritative_mode() -> None:
    df = _frame()
    df["g_0"] = df["a"] * 2
    df["g_1"] = df["b"] * 2
    res = analyze_dataset_structure(
        df,
        metadata={
            "inputs": ["a", "b"],
            "outputs": ["target"],
            "output_groups": {"g": ["g_0", "g_1"]},
        },
    )
    assert {"target", "g_0", "g_1"} <= set(res["output_variables"])
    assert res["output_groups"]["g"] == ["g_0", "g_1"]


def test_text_column_in_declared_output_group_raises() -> None:
    df = _frame()
    with pytest.raises(ValueError, match=r"label.*numeric"):
        analyze_dataset_structure(
            df,
            metadata={"output_groups": {"targets": ["target", "label"]}},
            hints={"inputs": ["a"], "outputs": ["target"]},
        )
