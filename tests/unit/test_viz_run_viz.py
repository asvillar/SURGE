"""Tests for surge.viz.run_viz helpers."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from surge.viz import run_viz


def test_viz_hpo_writes_figure_with_auto_metric(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "hpo").mkdir(parents=True)
    trials = [{"number": i, "value": 0.8 + 0.01 * i} for i in range(6)]
    (run_dir / "hpo" / "mlp_hpo.json").write_text(
        json.dumps({"trials": trials, "metric": "val_r2"}))
    out = tmp_path / "out"
    out.mkdir()
    paths = run_viz.viz_hpo(run_dir, out)
    assert paths == [str(out / "hpo_convergence.png")]
    assert (out / "hpo_convergence.png").exists()


def test_viz_hpo_logs_real_errors(tmp_path, caplog):
    run_dir = tmp_path / "run"
    (run_dir / "hpo").mkdir(parents=True)
    (run_dir / "hpo" / "mlp_hpo.json").write_text("not json")
    with caplog.at_level(logging.WARNING, logger=run_viz.LOG.name):
        assert run_viz.viz_hpo(run_dir, tmp_path) == []
    assert any("HPO convergence" in r.message for r in caplog.records)


def test_model_labels_fall_back_to_model_name():
    assert run_viz._model_short_name("my_model") == "My Model"


def test_train_ranges_builds_from_parquet_predictions(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.random(20), "x2": rng.random(20),
                       "y": rng.random(20)})
    data = tmp_path / "data.csv"
    df.to_csv(data, index=False)
    run_dir = tmp_path / "run"
    (run_dir / "predictions").mkdir(parents=True)
    pd.DataFrame({"index": np.arange(10)}).to_parquet(
        run_dir / "predictions" / "m_train.parquet")
    spec = SimpleNamespace(dataset_format="csv", metadata_path=None,
                           metadata_overrides={"input_columns": ["x1", "x2"],
                                               "output_columns": ["y"]},
                           analyzer={}, seed=0, sample_rows=None,
                           input_columns=None, n_inputs=None)
    ranges = run_viz._load_or_build_train_ranges(run_dir, spec, data, {}, None, None)
    assert ranges is not None
    assert ranges["inputs"]["columns"] == ["x1", "x2"]
    assert (run_dir / "train_data_ranges.json").exists()


def test_train_ranges_without_predictions_returns_none(tmp_path):
    (tmp_path / "predictions").mkdir()
    spec = SimpleNamespace()
    assert run_viz._load_or_build_train_ranges(
        tmp_path, spec, tmp_path / "d.csv", {}, None, None) is None
