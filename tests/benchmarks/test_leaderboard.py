"""Tests for surge.benchmarks.leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surge.benchmarks.leaderboard import (
    _default_models_for,
    format_leaderboard_table,
    log_leaderboard_to_mlflow,
    run_leaderboard,
)
from surge.benchmarks.registry import benchmark_info


# ---------------------------------------------------------------------------
# Model compatibility helpers
# ---------------------------------------------------------------------------


def test_default_models_for_regression():
    models = _default_models_for("regression")
    assert "sklearn.random_forest" in models
    assert "sklearn.mlp" in models
    assert "sklearn.gradient_boosting_regressor" in models
    # Classification models must not bleed in
    assert "sklearn.random_forest_classifier" not in models


def test_default_models_for_classification():
    models = _default_models_for("classification")
    assert "sklearn.random_forest_classifier" in models
    assert "sklearn.gradient_boosting_classifier" in models
    assert "sklearn.logistic_regression" in models
    assert "sklearn.random_forest" not in models


# ---------------------------------------------------------------------------
# run_leaderboard — functional smoke (Tier 0, fast)
# ---------------------------------------------------------------------------


def test_run_leaderboard_regression(tmp_path):
    results = run_leaderboard(
        ["synthetic.regression_1d"],
        model_keys=["sklearn.random_forest", "sklearn.mlp"],
        seed=42,
        save_root=tmp_path,
    )
    assert "synthetic.regression_1d" in results
    rl = results["synthetic.regression_1d"]
    assert len(rl) == 2
    model_keys_seen = {r.model_key for r in rl}
    assert "sklearn.random_forest" in model_keys_seen
    assert "sklearn.mlp" in model_keys_seen
    # All should have numeric metrics
    for r in rl:
        assert "test_r2" in r.metrics
        assert "test_rmse" in r.metrics
        assert "runtime_s" in r.metrics


def test_run_leaderboard_classification(tmp_path):
    results = run_leaderboard(
        ["synthetic.classification_binary"],
        model_keys=["sklearn.random_forest_classifier", "sklearn.logistic_regression"],
        seed=42,
        save_root=tmp_path,
    )
    rl = results["synthetic.classification_binary"]
    assert len(rl) == 2
    for r in rl:
        assert "test_accuracy" in r.metrics
        assert "test_f1_macro" in r.metrics


def test_run_leaderboard_autosave(tmp_path):
    run_leaderboard(
        ["synthetic.regression_1d"],
        model_keys=["sklearn.random_forest"],
        seed=0,
        save_root=tmp_path,
    )
    saved = list(tmp_path.glob("synthetic.regression_1d/*/result.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["model_key"] == "sklearn.random_forest"


def test_run_leaderboard_no_save(tmp_path):
    run_leaderboard(
        ["synthetic.regression_1d"],
        model_keys=["sklearn.random_forest"],
        seed=0,
        save_root=None,
    )
    assert list(tmp_path.rglob("result.json")) == []


def test_run_leaderboard_unknown_model_skips():
    """An unknown model should be skipped gracefully, not raise."""
    results = run_leaderboard(
        ["synthetic.regression_1d"],
        model_keys=["sklearn.random_forest", "nonexistent.model_xyz"],
        seed=42,
        save_root=None,
    )
    rl = results["synthetic.regression_1d"]
    model_keys = {r.model_key for r in rl}
    assert "sklearn.random_forest" in model_keys
    unknown = [r for r in rl if r.model_key == "nonexistent.model_xyz"]
    assert len(unknown) == 1
    assert unknown[0].extra.get("status") == "skipped"
    assert "not in MODEL_REGISTRY" in unknown[0].extra.get("skip_reason", "")


def test_run_leaderboard_multiple_benchmarks(tmp_path):
    results = run_leaderboard(
        ["synthetic.regression_1d", "synthetic.classification_binary"],
        model_keys=None,  # use defaults
        seed=42,
        save_root=tmp_path,
    )
    assert "synthetic.regression_1d" in results
    assert "synthetic.classification_binary" in results
    assert len(results["synthetic.regression_1d"]) >= 2
    assert len(results["synthetic.classification_binary"]) >= 2


# ---------------------------------------------------------------------------
# format_leaderboard_table
# ---------------------------------------------------------------------------


def test_format_table_regression():
    results = run_leaderboard(
        ["synthetic.regression_1d"],
        model_keys=["sklearn.random_forest", "sklearn.mlp"],
        seed=42,
        save_root=None,
    )
    table = format_leaderboard_table("synthetic.regression_1d", results["synthetic.regression_1d"])
    assert "synthetic.regression_1d" in table
    assert "sklearn.random_forest" in table
    assert "sklearn.mlp" in table
    assert "test_r2" in table
    assert "test_rmse" in table
    assert "*" in table              # best-value marker
    assert "PASS" in table or "FAIL" in table


def test_format_table_classification():
    results = run_leaderboard(
        ["synthetic.classification_binary"],
        model_keys=["sklearn.random_forest_classifier", "sklearn.logistic_regression"],
        seed=42,
        save_root=None,
    )
    table = format_leaderboard_table(
        "synthetic.classification_binary",
        results["synthetic.classification_binary"],
    )
    assert "test_accuracy" in table
    assert "test_f1_macro" in table
    # Best marker appears exactly once per column
    lines = [l for l in table.splitlines() if "sklearn." in l]
    assert len(lines) == 2  # one row per model


def test_format_table_empty():
    table = format_leaderboard_table("tabular.iris", [])
    assert "no results" in table.lower()


# ---------------------------------------------------------------------------
# CLI --leaderboard and --compare-models
# ---------------------------------------------------------------------------


def test_cli_leaderboard_single_benchmark(capsys, tmp_path):
    from surge.benchmarks.run import main

    code = main([
        "--leaderboard",
        "--benchmark", "synthetic.regression_1d",
        "--no-save",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "synthetic.regression_1d" in out
    assert "sklearn.random_forest" in out
    assert "*" in out


def test_cli_leaderboard_tier0(capsys, tmp_path):
    from surge.benchmarks.run import main

    code = main([
        "--leaderboard",
        "--all-benchmarks",
        "--tier", "0",
        "--no-save",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "synthetic.regression_1d" in out
    assert "synthetic.classification_binary" in out


def test_cli_compare_models(capsys, tmp_path):
    from surge.benchmarks.run import main

    code = main([
        "--benchmark", "synthetic.classification_binary",
        "--compare-models", "sklearn.random_forest_classifier,sklearn.logistic_regression",
        "--no-save",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "sklearn.random_forest_classifier" in out
    assert "sklearn.logistic_regression" in out
    assert "test_accuracy" in out


def test_cli_leaderboard_saves_results(tmp_path):
    from surge.benchmarks.run import main

    code = main([
        "--leaderboard",
        "--benchmark", "synthetic.regression_1d",
        "--save-dir", str(tmp_path),
    ])
    assert code == 0
    saved = list(tmp_path.rglob("result.json"))
    # one result per *available* model: torch-backed entries in the
    # default leaderboard list are transparently skipped without torch
    import importlib.util

    expected = 2 if importlib.util.find_spec("torch") is not None else 1
    assert len(saved) >= expected


# ---------------------------------------------------------------------------
# MLflow leaderboard logging
# ---------------------------------------------------------------------------


def test_log_leaderboard_to_mlflow(tmp_path):
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE

    if not MLFLOW_AVAILABLE:
        pytest.skip("mlflow not installed")

    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # file store is deprecated in mlflow>=3.14
    results = run_leaderboard(
        ["synthetic.regression_1d"],
        model_keys=["sklearn.random_forest", "sklearn.mlp"],
        seed=42,
        save_root=tmp_path / "benchmark_reports",
    )
    ok = log_leaderboard_to_mlflow(
        results,
        experiment_name="test_leaderboard",
        tracking_uri=tracking_uri,
        save_tables=True,
        tables_dir=tmp_path / "tables",
    )
    assert ok is True

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("test_leaderboard")
    assert exp is not None

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 2

    # Every run must have benchmark_key tag and test_r2 metric
    for run in runs:
        assert run.data.tags["benchmark_key"] == "synthetic.regression_1d"
        assert "test_r2" in run.data.metrics

    model_keys_logged = {r.data.tags["model_key"] for r in runs}
    assert "sklearn.random_forest" in model_keys_logged
    assert "sklearn.mlp" in model_keys_logged


def test_cli_leaderboard_mlflow(tmp_path):
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE

    if not MLFLOW_AVAILABLE:
        pytest.skip("mlflow not installed")

    import mlflow
    from surge.benchmarks.run import main

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # file store is deprecated in mlflow>=3.14
    code = main([
        "--leaderboard",
        "--benchmark", "synthetic.classification_binary",
        "--no-save",
        "--mlflow",
        "--mlflow-experiment", "test_lb_cli",
        "--mlflow-tracking-uri", tracking_uri,
    ])
    assert code == 0

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("test_lb_cli")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    # Should have one run per compatible classification model
    assert len(runs) >= 2
    for run in runs:
        assert "test_accuracy" in run.data.metrics
        assert run.data.tags["benchmark_key"] == "synthetic.classification_binary"
