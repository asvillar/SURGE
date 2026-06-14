"""Tests for XGBoost adapters and the HPO module."""

from __future__ import annotations

import numpy as np
import pytest

from surge.benchmarks.registry import run_benchmark
from surge.model.registry import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def reg_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] * 2 - X[:, 1] + 0.1 * rng.standard_normal(100)
    return X[:80], y[:80], X[80:], y[80:]


@pytest.fixture()
def clf_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:90], y[:90], X[90:], y[90:]


@pytest.fixture()
def multioutput_data():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 4))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])
    return X[:80], Y[:80], X[80:], Y[80:]


# ---------------------------------------------------------------------------
# XGBoost availability guard
# ---------------------------------------------------------------------------


@pytest.fixture()
def xgb_available():
    try:
        from surge.model.backends.xgboost import XGBOOST_AVAILABLE
        return XGBOOST_AVAILABLE
    except Exception:
        return False


# ===========================================================================
# XGBoost Regressor
# ===========================================================================


def test_xgbr_registered(xgb_available):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    assert "xgboost.xgbregressor" in MODEL_REGISTRY


def test_xgbr_fit_predict(xgb_available, reg_data):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    X_tr, y_tr, X_te, y_te = reg_data
    adapter = MODEL_REGISTRY.create("xgboost.xgbregressor", n_estimators=10)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_xgbr_multioutput(xgb_available, multioutput_data):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    X_tr, y_tr, X_te, y_te = multioutput_data
    adapter = MODEL_REGISTRY.create("xgboost.xgbregressor", n_estimators=10)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te), 2)


def test_xgbr_benchmark(xgb_available):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    result = run_benchmark("synthetic.regression_1d", model_key="xgboost.xgbregressor")
    assert result.model_key == "xgboost.xgbregressor"
    assert "test_r2" in result.metrics
    assert result.passed


def test_xgbr_multioutput_benchmark(xgb_available):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    result = run_benchmark("synthetic.multioutput_2d", model_key="xgboost.xgbregressor")
    assert "test_r2" in result.metrics
    assert result.passed


# ===========================================================================
# XGBoost Classifier
# ===========================================================================


def test_xgbc_registered(xgb_available):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    assert "xgboost.xgbclassifier" in MODEL_REGISTRY


def test_xgbc_fit_predict(xgb_available, clf_data):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    X_tr, y_tr, X_te, y_te = clf_data
    adapter = MODEL_REGISTRY.create("xgboost.xgbclassifier", n_estimators=10)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_xgbc_predict_proba(xgb_available, clf_data):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    X_tr, y_tr, X_te, y_te = clf_data
    adapter = MODEL_REGISTRY.create("xgboost.xgbclassifier", n_estimators=10)
    adapter.fit(X_tr, y_tr)
    proba = adapter.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_xgbc_benchmark(xgb_available):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    result = run_benchmark("tabular.iris", model_key="xgboost.xgbclassifier")
    assert result.model_key == "xgboost.xgbclassifier"
    assert "test_accuracy" in result.metrics
    assert result.passed


def test_xgbc_aliases(xgb_available):
    if not xgb_available:
        pytest.skip("XGBoost not available")
    a1 = MODEL_REGISTRY.create("xgbr", n_estimators=5)
    a2 = MODEL_REGISTRY.create("xgbc", n_estimators=5)
    assert a1.name == "xgboost.xgbregressor"
    assert a2.name == "xgboost.xgbclassifier"


# ===========================================================================
# HPO module
# ===========================================================================


def test_list_hpo_models():
    from surge.benchmarks.hpo import list_hpo_models
    models = list_hpo_models()
    assert "xgboost.xgbregressor" in models
    assert "xgboost.xgbclassifier" in models
    assert "sklearn.random_forest" in models
    assert "pytorch.residual_mlp" in models
    assert "tabpfn.regressor" in models
    assert "tabpfn.classifier" in models


def test_suggest_params_xgbr():
    import optuna
    from surge.benchmarks.hpo import suggest_params

    # Use a real study + trial so distributions are consistent.
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params("xgboost.xgbregressor", trial)
    assert "n_estimators" in params
    assert "learning_rate" in params
    assert "max_depth" in params
    assert 50 <= params["n_estimators"] <= 600
    assert 2 <= params["max_depth"] <= 10


def test_suggest_params_unknown_raises():
    import optuna
    from surge.benchmarks.hpo import suggest_params

    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(KeyError, match="No HPO search space"):
        suggest_params("nonexistent.model", trial)


def test_suggest_params_decodes_categorical_values():
    import optuna
    from surge.benchmarks.hpo import suggest_params

    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params("pytorch.residual_mlp", trial)
    assert isinstance(params["hidden_layers"], list)
    assert all(isinstance(v, int) for v in params["hidden_layers"])


def test_all_hpo_search_spaces_suggest_params():
    import optuna
    from surge.benchmarks.hpo import list_hpo_models, suggest_params

    for model_key in list_hpo_models():
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params(model_key, trial)
        assert isinstance(params, dict)
        assert params, f"{model_key} returned no params"
        for value in params.values():
            assert not isinstance(value, range)


def test_run_benchmark_hpo_rf(tmp_path):
    """Fast HPO smoke test: 5 trials, RF on synthetic.regression_1d."""
    from surge.benchmarks.hpo import run_benchmark_hpo

    result, best_params = run_benchmark_hpo(
        "synthetic.regression_1d",
        "sklearn.random_forest",
        n_trials=5,
        seed=0,
        save_root=tmp_path,
    )
    assert result is not None
    assert "test_r2" in result.metrics
    assert np.isfinite(result.metrics["test_r2"])
    assert "hpo_n_trials" in result.extra
    assert result.extra["hpo_n_trials"] == 5
    assert "n_estimators" in best_params or "max_depth" in best_params
    # auto-saved
    saved = list(tmp_path.rglob("result.json"))
    assert len(saved) == 1


def test_run_benchmark_hpo_sklearn_mlp(tmp_path):
    """Fast non-tree HPO smoke test: sklearn MLP on synthetic regression."""
    from surge.benchmarks.hpo import run_benchmark_hpo

    result, best_params = run_benchmark_hpo(
        "synthetic.regression_1d",
        "sklearn.mlp",
        n_trials=2,
        seed=3,
        save_root=None,
    )
    assert result is not None
    assert "test_r2" in result.metrics
    assert np.isfinite(result.metrics["test_r2"])
    assert "hidden_layer_sizes" in best_params
    assert isinstance(best_params["hidden_layer_sizes"], tuple)


def test_run_benchmark_hpo_classification(tmp_path):
    """Fast HPO smoke test: 5 trials, GBC on synthetic.classification_binary."""
    from surge.benchmarks.hpo import run_benchmark_hpo

    result, best_params = run_benchmark_hpo(
        "synthetic.classification_binary",
        "sklearn.gradient_boosting_classifier",
        n_trials=5,
        seed=0,
        save_root=tmp_path,
    )
    assert result is not None
    assert "test_accuracy" in result.metrics
    assert result.extra["hpo_metric"] == "test_accuracy"
    assert result.extra["hpo_direction"] == "maximize"


def test_run_benchmark_hpo_xgbr(xgb_available, tmp_path):
    """XGBoost HPO smoke test."""
    if not xgb_available:
        pytest.skip("XGBoost not available")
    from surge.benchmarks.hpo import run_benchmark_hpo

    result, best_params = run_benchmark_hpo(
        "synthetic.regression_1d",
        "xgboost.xgbregressor",
        n_trials=5,
        seed=1,
        save_root=None,
    )
    assert result is not None
    assert "test_r2" in result.metrics
    assert "learning_rate" in best_params or "n_estimators" in best_params


def test_run_benchmark_hpo_custom_metric(tmp_path):
    """HPO with explicit metric and direction."""
    from surge.benchmarks.hpo import run_benchmark_hpo

    result, best_params = run_benchmark_hpo(
        "synthetic.regression_1d",
        "sklearn.random_forest",
        n_trials=3,
        seed=2,
        metric="test_rmse",
        direction="minimize",
        save_root=None,
    )
    assert result is not None
    assert result.extra["hpo_metric"] == "test_rmse"
    assert result.extra["hpo_direction"] == "minimize"


def test_hpo_print_summary(capsys):
    """print_hpo_summary should not raise and should include expected sections."""
    from surge.benchmarks.base import BenchmarkResult
    from surge.benchmarks.hpo import print_hpo_summary

    result = BenchmarkResult(
        benchmark_key="synthetic.regression_1d",
        model_key="sklearn.random_forest",
        tier="0",
        task_type="regression",
        metrics={"test_r2": 0.99, "test_rmse": 0.05, "runtime_s": 0.3},
        passed=True,
    )
    print_hpo_summary(
        result,
        {"n_estimators": 200, "max_depth": 10},
        benchmark_key="synthetic.regression_1d",
        model_key="sklearn.random_forest",
        n_trials=5,
        metric="test_r2",
    )
    out = capsys.readouterr().out
    assert "HPO Summary" in out
    assert "n_estimators" in out
    assert "test_r2" in out
    assert "PASS" in out


# ===========================================================================
# CLI integration
# ===========================================================================


def test_cli_list_hpo_models(capsys):
    from surge.benchmarks.run import main

    rc = main(["--list-hpo-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "xgboost.xgbregressor" in out
    assert "sklearn.random_forest" in out


def test_cli_hpo_flag(tmp_path):
    from surge.benchmarks.run import main

    rc = main([
        "--benchmark", "synthetic.regression_1d",
        "--model", "sklearn.random_forest",
        "--hpo", "--hpo-trials", "3",
        "--save-dir", str(tmp_path),
    ])
    assert rc == 0
    saved = list(tmp_path.rglob("result.json"))
    assert len(saved) == 1


def test_cli_hpo_requires_benchmark(capsys):
    from surge.benchmarks.run import main

    rc = main(["--model", "sklearn.random_forest", "--hpo", "--hpo-trials", "2"])
    assert rc == 1


def test_cli_hpo_requires_model(capsys):
    from surge.benchmarks.run import main

    rc = main(["--benchmark", "synthetic.regression_1d", "--hpo", "--hpo-trials", "2"])
    assert rc == 1
