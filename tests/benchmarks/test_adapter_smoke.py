"""Adapter and benchmark smoke tests.

Covers:
- sklearn.gradient_boosting_regressor
- pytorch.residual_mlp
- pytorch.mlp_classifier
- synthetic.multioutput_2d benchmark
- surge.viz.benchmark module
"""

from __future__ import annotations

import numpy as np
import pytest

from surge.benchmarks.registry import list_benchmarks, run_benchmark
from surge.model.registry import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_regression_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = X[:, 0] * 2 + X[:, 1] - 0.5 + 0.1 * rng.standard_normal(100)
    return X[:80], y[:80], X[80:], y[80:]


@pytest.fixture()
def small_multioutput_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    A = np.array([[1, -1], [0.5, 0.5], [-1, 0], [0, 1]])
    Y = X @ A + 0.05 * rng.standard_normal((100, 2))
    return X[:80], Y[:80], X[80:], Y[80:]


@pytest.fixture()
def small_classification_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:90], y[:90], X[90:], y[90:]


# ---------------------------------------------------------------------------
# sklearn.gradient_boosting_regressor
# ---------------------------------------------------------------------------


def test_gbr_registered():
    assert "sklearn.gradient_boosting_regressor" in MODEL_REGISTRY


def test_gbr_fit_predict(small_regression_data):
    X_tr, y_tr, X_te, y_te = small_regression_data
    adapter = MODEL_REGISTRY.create("sklearn.gradient_boosting_regressor")
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_gbr_benchmark():
    result = run_benchmark("synthetic.regression_1d", model_key="sklearn.gradient_boosting_regressor")
    assert result.benchmark_key == "synthetic.regression_1d"
    assert result.model_key == "sklearn.gradient_boosting_regressor"
    assert "test_r2" in result.metrics
    assert result.passed


# ---------------------------------------------------------------------------
# pytorch.residual_mlp
# ---------------------------------------------------------------------------


@pytest.fixture()
def pytorch_available():
    try:
        from surge.model.pytorch import PYTORCH_AVAILABLE
        return PYTORCH_AVAILABLE
    except Exception:
        return False


def test_residual_mlp_registered(pytorch_available):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    assert "pytorch.residual_mlp" in MODEL_REGISTRY


def test_residual_mlp_fit_predict(pytorch_available, small_regression_data):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    X_tr, y_tr, X_te, y_te = small_regression_data
    adapter = MODEL_REGISTRY.create("pytorch.residual_mlp", n_epochs=5)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_residual_mlp_multioutput(pytorch_available, small_multioutput_data):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    X_tr, y_tr, X_te, y_te = small_multioutput_data
    adapter = MODEL_REGISTRY.create("pytorch.residual_mlp", n_epochs=5)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te), 2)
    assert np.isfinite(preds).all()


def test_residual_mlp_benchmark(pytorch_available):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    result = run_benchmark(
        "synthetic.regression_1d",
        model_key="pytorch.residual_mlp",
    )
    assert result.model_key == "pytorch.residual_mlp"
    assert "test_r2" in result.metrics
    assert np.isfinite(result.metrics["test_r2"])


# ---------------------------------------------------------------------------
# pytorch.mlp_classifier
# ---------------------------------------------------------------------------


def test_mlp_classifier_registered(pytorch_available):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    assert "pytorch.mlp_classifier" in MODEL_REGISTRY


def test_mlp_classifier_fit_predict(pytorch_available, small_classification_data):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    X_tr, y_tr, X_te, y_te = small_classification_data
    adapter = MODEL_REGISTRY.create("pytorch.mlp_classifier", n_epochs=5)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_mlp_classifier_predict_proba(pytorch_available, small_classification_data):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    X_tr, y_tr, X_te, y_te = small_classification_data
    adapter = MODEL_REGISTRY.create("pytorch.mlp_classifier", n_epochs=5)
    adapter.fit(X_tr, y_tr)
    proba = adapter.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_mlp_classifier_benchmark(pytorch_available):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    result = run_benchmark("tabular.iris", model_key="pytorch.mlp_classifier")
    assert result.model_key == "pytorch.mlp_classifier"
    assert "test_accuracy" in result.metrics
    assert np.isfinite(result.metrics["test_accuracy"])


# ---------------------------------------------------------------------------
# synthetic.multioutput_2d benchmark
# ---------------------------------------------------------------------------


def test_multioutput_2d_in_registry():
    assert "synthetic.multioutput_2d" in list_benchmarks()


def test_multioutput_2d_default_model():
    result = run_benchmark("synthetic.multioutput_2d")
    assert result.benchmark_key == "synthetic.multioutput_2d"
    assert "test_r2" in result.metrics
    assert result.metrics["test_r2"] > 0.5


def test_multioutput_2d_with_gbr():
    result = run_benchmark("synthetic.multioutput_2d", model_key="sklearn.gradient_boosting_regressor")
    assert result.passed


def test_multioutput_2d_with_residual_mlp(pytorch_available):
    if not pytorch_available:
        pytest.skip("PyTorch not available")
    result = run_benchmark("synthetic.multioutput_2d", model_key="pytorch.residual_mlp")
    assert result.benchmark_key == "synthetic.multioutput_2d"
    assert "test_r2" in result.metrics


# ---------------------------------------------------------------------------
# surge.viz.benchmark
# ---------------------------------------------------------------------------


def test_viz_imports():
    from surge.viz.benchmark import (
        load_benchmark_results,
        plot_benchmark_leaderboard,
        plot_metric_table,
        plot_multi_benchmark_dashboard,
    )
    assert callable(plot_benchmark_leaderboard)
    assert callable(plot_metric_table)
    assert callable(plot_multi_benchmark_dashboard)
    assert callable(load_benchmark_results)


def test_viz_leaderboard_bar_chart(tmp_path):
    from surge.benchmarks.base import BenchmarkResult
    from surge.viz.benchmark import plot_benchmark_leaderboard

    results = [
        BenchmarkResult(
            benchmark_key="tabular.iris",
            model_key="sklearn.random_forest_classifier",
            tier="1",
            task_type="classification",
            metrics={"test_accuracy": 0.97, "test_f1_macro": 0.97},
            passed=True,
        ),
        BenchmarkResult(
            benchmark_key="tabular.iris",
            model_key="sklearn.logistic_regression",
            tier="1",
            task_type="classification",
            metrics={"test_accuracy": 0.93, "test_f1_macro": 0.92},
            passed=True,
        ),
    ]
    save_path = tmp_path / "leaderboard.png"
    fig = plot_benchmark_leaderboard(results, metric="test_accuracy", save_path=save_path)
    assert save_path.exists()
    assert save_path.with_suffix(".pdf").exists()


def test_viz_metric_table(tmp_path):
    from surge.benchmarks.base import BenchmarkResult
    from surge.viz.benchmark import plot_metric_table

    results = [
        BenchmarkResult(
            benchmark_key="tabular.iris",
            model_key="rf",
            tier="1",
            task_type="classification",
            metrics={"test_accuracy": 0.97, "test_f1_macro": 0.97},
            passed=True,
        ),
        BenchmarkResult(
            benchmark_key="tabular.iris",
            model_key="lr",
            tier="1",
            task_type="classification",
            metrics={"test_accuracy": 0.91, "test_f1_macro": 0.90},
            passed=False,
        ),
    ]
    save_path = tmp_path / "table.png"
    fig = plot_metric_table(results, save_path=save_path)
    assert save_path.exists()


def test_viz_dashboard_multi(tmp_path):
    from surge.benchmarks.base import BenchmarkResult
    from surge.viz.benchmark import plot_multi_benchmark_dashboard

    r1 = BenchmarkResult(
        benchmark_key="tabular.iris",
        model_key="rf",
        tier="1",
        task_type="classification",
        metrics={"test_accuracy": 0.97},
        passed=True,
    )
    r2 = BenchmarkResult(
        benchmark_key="tabular.diabetes",
        model_key="rf",
        tier="1",
        task_type="regression",
        metrics={"test_r2": 0.52},
        passed=True,
    )
    save_path = tmp_path / "dashboard.png"
    fig = plot_multi_benchmark_dashboard(
        {"tabular.iris": [r1], "tabular.diabetes": [r2]},
        save_path=save_path,
    )
    assert save_path.exists()


def test_load_benchmark_results(tmp_path):
    from surge.benchmarks.base import BenchmarkResult
    from surge.viz.benchmark import load_benchmark_results

    r = BenchmarkResult(
        benchmark_key="synthetic.regression_1d",
        model_key="sklearn.random_forest",
        tier="0",
        task_type="regression",
        metrics={"test_r2": 0.99},
        passed=True,
    )
    r.save(root=tmp_path)
    loaded = load_benchmark_results(tmp_path)
    assert "synthetic.regression_1d" in loaded
    assert loaded["synthetic.regression_1d"][0].metrics["test_r2"] == pytest.approx(0.99)
