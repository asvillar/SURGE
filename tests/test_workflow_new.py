"""Tests for the unified SURGE workflow API.

Exercises the current ``SurrogateDataset`` + ``SurrogateEngine`` surface.
End-to-end ``run_surrogate_workflow`` coverage lives in
``tests/test_e2e_release_smoke.py``; here we stay at the unit level so
these tests are fast and don't touch disk beyond ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from surge.dataset import SurrogateDataset
from surge.engine import EngineRunConfig, ModelSpec, SurrogateEngine


def _make_dummy_dataframe(n_rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    inputs = {
        "input_a": rng.normal(size=n_rows),
        "input_b": rng.normal(size=n_rows),
        "input_c": rng.normal(size=n_rows),
    }
    outputs = {
        "output_x": inputs["input_a"] * 0.5 + rng.normal(scale=0.1, size=n_rows),
        "output_y": inputs["input_b"] * -0.25 + rng.normal(scale=0.05, size=n_rows),
    }
    return pd.DataFrame({**inputs, **outputs})


class TestSurrogateDataset:
    """Unit tests for the refactored SurrogateDataset API."""

    def test_load_from_dataframe_auto_detection(self) -> None:
        df = _make_dummy_dataframe()
        dataset = SurrogateDataset.from_dataframe(df)
        summary = dataset.summary()

        assert summary["n_rows"] == len(df)
        assert set(dataset.input_columns).issubset(df.columns)
        assert set(dataset.output_columns).issubset(df.columns)
        assert dataset.df is not None
        assert not dataset.df.isna().any().any()

    def test_load_from_path_with_metadata(self, tmp_path: Path) -> None:
        df = _make_dummy_dataframe()
        data_path = tmp_path / "dummy.pkl"
        df.to_pickle(data_path)

        metadata_path = tmp_path / "meta.yaml"
        metadata_path.write_text(
            "inputs:\n  - input_a\n  - input_b\noutputs:\n  - output_x\n",
            encoding="utf-8",
        )

        dataset = SurrogateDataset.from_path(data_path, metadata_path=metadata_path)
        assert dataset.input_columns == ["input_a", "input_b"]
        assert dataset.output_columns == ["output_x"]
        assert dataset.df is not None
        assert len(dataset.df) == len(df)


class TestSurrogateEngine:
    """Focused tests for the SurrogateEngine data prep + training surface."""

    def test_prepare_standardizes_splits(self) -> None:
        df = _make_dummy_dataframe(120)
        config = EngineRunConfig(
            test_fraction=0.2,
            val_fraction=0.2,
            standardize_inputs=True,
            standardize_outputs=True,
            random_state=0,
        )
        engine = SurrogateEngine(run_config=config)
        engine.configure_dataframe(df, ["input_a", "input_b", "input_c"], ["output_x", "output_y"])
        engine.prepare()

        proc = engine.get_processed_splits()
        assert proc.X_train.shape == (72, 3)
        assert proc.y_train.shape == (72, 2)
        assert np.allclose(proc.X_train.mean(axis=0), 0.0, atol=1e-7)
        assert np.allclose(proc.X_train.std(axis=0), 1.0, atol=1e-6)

        scalers = engine.scalers
        assert scalers.input_scaler is not None
        assert scalers.output_scaler is not None

    def test_run_trains_registered_model(self) -> None:
        df = _make_dummy_dataframe(160)
        dataset = SurrogateDataset.from_dataframe(df)

        engine = SurrogateEngine(run_config=EngineRunConfig(test_fraction=0.2, val_fraction=0.2))
        engine.configure_from_dataset(dataset)

        spec = ModelSpec(
            key="sklearn.random_forest",
            name="rf_unit_engine",
            params={"n_estimators": 25, "max_depth": 6},
        )
        results = engine.run([spec])

        assert len(results) == 1
        result = results[0]
        assert "r2" in result.val_metrics
        assert "train" in result.predictions
        assert result.predictions["train"]["y_pred"].shape == result.predictions["train"]["y_true"].shape
        assert engine.results  # run() should record results internally

