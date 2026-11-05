"""Regression tests for PR #18 review findings (lorenz gate, sweep defaults,
lorenz cache validation, --no-predictions in sweeps)."""

from __future__ import annotations

import pytest

import numpy as np

from surge.benchmarks import leaderboard as lb
from surge.model.registry import MODEL_REGISTRY


def test_check_pass_missing_gate_metric_is_not_pass():
    metrics = {"test_nrmse": 0.01}
    assert lb._check_pass("sequence.lorenz63", metrics) is False
    assert metrics == {"test_nrmse": 0.01}  # pure: no mutation
    assert lb.gate_metric_key("sequence.lorenz63") == "test_vpt_median"
    # ungated benchmarks keep the permissive default
    assert lb._check_pass("not.a.benchmark", {}) is True


def test_run_with_adapter_lorenz_emits_rollout_metrics():
    adapter = MODEL_REGISTRY.create("sklearn.ridge")
    res = lb._run_with_adapter("sequence.lorenz63", adapter, seed=0)
    for k in ("test_vpt_median", "test_vpt_p25", "test_vpt_p75",
              "test_climate_error", "test_nrmse"):
        assert k in res.metrics, k
    assert "gate_metric_missing" not in res.metrics
    assert res.extra["rollout_lyapunov_times"] == 10.0


def test_load_lorenz63_regenerates_stale_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lb.paths, "bench_data_root", lambda: tmp_path)
    cache = tmp_path / "sequence" / "lorenz63.npz"
    cache.parent.mkdir(parents=True)
    np.savez_compressed(cache, X=np.zeros((10, 60)), y=np.zeros((10, 60)))
    X, y = lb._load_lorenz63()
    assert X.ndim == 2 and X.shape[1] == 3 and y.shape == X.shape
    assert "stale shape" in capsys.readouterr().err
    assert np.load(cache)["X"].shape == X.shape


def test_run_leaderboard_merges_bench_model_defaults(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # pytorch.residual_mlp is unregistered on the no-torch job
    captured: list[dict] = []

    class _Stub:
        name = "pytorch.residual_mlp"

    def _create(key, **kw):
        captured.append(kw)
        return _Stub()

    monkeypatch.setattr(MODEL_REGISTRY, "create", _create)
    monkeypatch.setattr(lb, "_run_with_adapter", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_task_mismatch", lambda *a, **k: None)
    lb.run_leaderboard(["plasma.qlknn10d"], model_keys=["pytorch.residual_mlp"],
                       save_root=tmp_path)
    want = lb.BENCH_MODEL_DEFAULTS["plasma.qlknn10d"]["pytorch.residual_mlp"]
    assert captured, "adapter was never constructed"
    kw = captured[0]
    assert kw["hidden_layers"] == want["hidden_layers"]
    assert kw["batch_size"] == want["batch_size"]
    assert kw["n_epochs"] == want["n_epochs"]  # epoch cap must not override


def test_run_leaderboard_save_predictions_flag(tmp_path):
    off = tmp_path / "off"
    on = tmp_path / "on"
    lb.run_leaderboard(["synthetic.regression_1d"], model_keys=["sklearn.ridge"],
                       save_root=off, save_predictions=False)
    lb.run_leaderboard(["synthetic.regression_1d"], model_keys=["sklearn.ridge"],
                       save_root=on, save_predictions=True)
    assert not list(off.rglob("predictions.npz"))
    assert list(on.rglob("predictions.npz"))
