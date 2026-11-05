"""SOLPS-NN exhaust benchmarks: plumbing, metrics, gate, listing (no network, no real data)."""

from __future__ import annotations

import numpy as np
import pytest

from surge.benchmarks import leaderboard as lb
from surge.benchmarks.tasks import RELATIVE_L2_PROTOCOL, solpsnn_scalar_metrics

SCALARS = "plasma.solpsnn_exhaust_scalars"
FIELD = "plasma.solpsnn_te_field"


def _synthetic(noise=0.0, n_tr=120, n_te=40):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (n_tr + n_te, 8))
    W = rng.normal(size=(8, 3))
    y = X @ W * 0.5 + np.array([5.0, 6.0, 20.0]) + noise * rng.normal(size=(n_tr + n_te, 3))
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def test_scalar_metrics_perfect_and_ten_percent():
    y = np.abs(np.random.default_rng(1).normal(size=(50, 3))) + 1.0
    names = ["psol", "pwmxap", "fnixap"]
    m = solpsnn_scalar_metrics(y, y, names)
    assert m["test_median_rel_err"] == 0.0 and m["test_r2_log10"] == pytest.approx(1.0)
    m = solpsnn_scalar_metrics(y, 1.1 * y, names)
    assert m["test_median_rel_err"] == pytest.approx(0.10)
    for n in names:
        assert m[f"test_median_rel_err_{n}"] == pytest.approx(0.10)
        assert f"test_r2_log10_{n}" in m


def test_scalar_metrics_gate_is_max_over_targets():
    y = np.ones((10, 3))
    p = y * np.array([1.0, 1.05, 1.3])
    m = solpsnn_scalar_metrics(y, p, ["a", "b", "c"])
    assert m["test_median_rel_err"] == pytest.approx(0.30)


def test_pooled_field_mode():
    y = np.full((5, 7), 2.0)
    m = solpsnn_scalar_metrics(y, y * 1.1)
    assert m["test_median_rel_err"] == pytest.approx(0.10)


def test_run_leaderboard_inverts_and_gates(monkeypatch):
    split = _synthetic()
    monkeypatch.setattr(lb, "_load_solpsnn_exhaust_scalars", lambda: split)
    res = lb.run_leaderboard([SCALARS], model_keys=["sklearn.ridge"], save_root=None)[SCALARS][0]
    m = res.metrics
    # noiseless linear log10 targets: ridge is near-perfect in log space,
    # so physical-unit errors are tiny and the gate passes
    assert m["test_r2"] > 0.99
    assert m["test_median_rel_err"] < 0.05
    for n in ("psol", "pwmxap", "fnixap"):
        assert f"test_median_rel_err_{n}" in m and f"test_r2_log10_{n}" in m
    assert res.passed is True
    assert res.extra["n_train"] == 120 and res.extra["n_test"] == 40


def test_run_leaderboard_fails_gate_on_noise(monkeypatch):
    split = _synthetic(noise=3.0)
    monkeypatch.setattr(lb, "_load_solpsnn_exhaust_scalars", lambda: split)
    res = lb.run_leaderboard([SCALARS], model_keys=["sklearn.ridge"], save_root=None)[SCALARS][0]
    assert res.metrics["test_median_rel_err"] > 0.20
    assert res.passed is False


def test_inverse_hook_applied_to_both_arrays():
    f = lb._TARGET_INVERSE[SCALARS]
    np.testing.assert_allclose(f(np.array([1.0, 2.0])), [10.0, 100.0])
    assert FIELD in lb._TARGET_INVERSE


def test_gate_and_protocols():
    assert lb._GATE_THRESHOLDS[SCALARS] == ("test_median_rel_err", 0.20)
    assert lb._GATE_THRESHOLDS[FIELD] == ("test_median_rel_err", 0.10)
    assert lb._check_pass(SCALARS, {"test_median_rel_err": 0.1}) is True
    assert lb._check_pass(SCALARS, {"test_median_rel_err": 0.3}) is False
    assert RELATIVE_L2_PROTOCOL[FIELD] == "global"


def test_field_model_list_excludes_trees():
    models = lb._default_models_for("regression", FIELD)
    assert models == ["pytorch.mlp", "pytorch.residual_mlp", "sklearn.ridge"]
    assert not any(t in m for m in models for t in ("forest", "gradient_boosting", "xgboost", "lgbm", "catboost"))


def test_bench_model_defaults():
    d = lb.BENCH_MODEL_DEFAULTS
    assert d[SCALARS]["pytorch.residual_mlp"]["hidden_layers"] == [1000] * 8
    assert d[SCALARS]["kumo.tabular_regressor"] == {"target_transform": None}
    assert d[FIELD]["pytorch.residual_mlp"]["batch_size"] == 64


def test_registry_listing_and_aliases():
    from surge.benchmarks.registry import benchmark_info, list_benchmarks, resolve_benchmark_key as resolve_key

    keys = {b if isinstance(b, str) else b.get("key", b) for b in list_benchmarks()}
    flat = " ".join(map(str, list_benchmarks()))
    assert SCALARS in flat and FIELD in flat
    for alias, key in (("solpsnn", SCALARS), ("solps_scalars", SCALARS), ("solps_te", FIELD)):
        assert resolve_key(alias) == key
    assert benchmark_info(SCALARS)["task_type"] == "regression"
    del keys


def test_status_and_report_formatting():
    from surge.benchmarks import status
    from surge.report import leaderboard as rep

    r = {"metrics": {"test_median_rel_err": 0.123, "test_r2": 0.99}, "passed": True,
         "model_key": "sklearn.ridge", "_ts": "2026-11-04T00:00:00Z"}
    assert "median rel. err. 12.3 %" in status._fmt_result(r)
    assert status._is_lower_better("test_median_rel_err")
    assert rep._is_lower_better("test_median_rel_err")
    assert rep._LABELS["test_median_rel_err"] == "median rel. err."
    assert rep._fmt_value("test_median_rel_err", {"mean": 0.123, "std": 0.0}) == "12.3 %"
    assert rep.primary_metric_key({"m": {"metrics": {"test_median_rel_err": 1, "test_r2": 1}}}) == "test_median_rel_err"
