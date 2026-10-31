"""Third-pass review fixes: rel-L2 on adapter paths, diverged rollouts, histogram, clips."""
from __future__ import annotations

import numpy as np
import pytest

from surge.benchmarks.tasks import relative_l2_metrics


class _Perfect:
    """Stub adapter that predicts y = 2 X exactly."""
    name = "stub.perfect"

    def fit(self, X, y, **kw):
        return self

    def predict(self, X):
        return 2.0 * np.asarray(X)


def _fake(n=60, d=8):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    return X, 2.0 * X


def test_relative_l2_protocol_table():
    y = np.array([[3.0, 4.0], [0.0, 1.0]])
    p = y + 1.0
    g = relative_l2_metrics("pde.burgers_1d", y, p)
    assert g["test_relative_l2"] == pytest.approx(2.0 / np.sqrt(26), rel=1e-6)
    assert relative_l2_metrics("thewell.gray_scott", y, p) == g
    m = relative_l2_metrics("plasma.constellaration_modb", y, p)
    per = np.array([np.sqrt(2) / 5, np.sqrt(2) / 1])
    assert m["test_relative_l2"] == pytest.approx(np.median(per))
    assert m["test_relative_l2_mean"] == pytest.approx(per.mean())
    assert relative_l2_metrics("tabular.iris", y, p) == {}


def test_leaderboard_adapter_path_emits_rel_l2_and_passes(monkeypatch):
    from surge.benchmarks import leaderboard

    monkeypatch.setattr(leaderboard, "_load_dataset", lambda key: _fake())
    res = leaderboard._run_with_adapter("pde.burgers_1d", _Perfect(), seed=0)
    assert res.metrics["test_relative_l2"] == 0.0
    assert res.passed is True


@pytest.mark.parametrize("key", ["thewell.gray_scott", "plasma.constellaration_modb",
                                 "plasma.constellaration_surfaces"])
def test_leaderboard_adapter_path_gated_field_keys(monkeypatch, key):
    from surge.benchmarks import leaderboard

    monkeypatch.setattr(leaderboard, "_load_dataset", lambda k: _fake())
    res = leaderboard._run_with_adapter(key, _Perfect(), seed=0)
    assert res.metrics["test_relative_l2"] == 0.0
    assert res.passed is True


def test_hpo_adapter_data_path_emits_rel_l2_and_passes():
    from surge.benchmarks.hpo import _run_with_adapter_data

    X, y = _fake()
    res = _run_with_adapter_data("pde.burgers_1d", _Perfect(), X[:40], X[40:], y[:40], y[40:],
                                 task_type="regression", tier="1", seed=0)
    assert res.metrics["test_relative_l2"] == 0.0
    assert res.passed is True


def test_lorenz_diverged_trajectory_does_not_abort_batch(monkeypatch):
    import surge.inference.rollout as ro
    from surge.benchmarks.tasks import lorenz_rollout_metrics

    class Stub:
        def __init__(self):
            self.calls = 0

        def predict(self, x):
            x = np.asarray(x)
            if not np.all(np.isfinite(x)):
                raise ValueError("non-finite input")
            self.calls += 1
            out = x.copy()
            if self.calls == 4:
                out[7] = np.nan
            return out

    seen = []
    real = ro.valid_prediction_time

    def spy(errors, *a, **k):
        seen.append(np.asarray(errors).copy())
        return real(errors, *a, **k)

    monkeypatch.setattr(ro, "valid_prediction_time", spy)
    res = lorenz_rollout_metrics(Stub(), seed=0, climate_norm=8.0, n_eval_ics=10,
                                 rollout_lyapunov_times=0.5)
    assert "test_vpt_median" in res["metrics"]
    # IC 7 went non-finite at rollout step index 3 (4th model application) and
    # stays non-finite; its VPT is the first crossing, 4 * dt_lyapunov.
    assert np.isnan(seen[7][3:]).all() and np.isfinite(seen[7][:3]).all()
    assert np.isfinite(seen[6]).all()
    assert real(seen[7], 0.4, 1.0, horizon_lyapunov=9.0) == 4.0
    # the saved median curve stays finite across all steps despite IC 7 diverging
    curve = np.asarray(res["extra"]["median_error_curve"])
    assert curve.shape[0] == len(seen[7]) and np.isfinite(curve).all()


def test_climate_histogram_error_nonfinite_policy():
    from surge.inference.rollout import climate_histogram_error

    rng = np.random.default_rng(1)
    t = rng.normal(size=4000)
    p = rng.normal(size=4000) + 0.1
    # finite inputs: identical to the density-based formula
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    edges = np.linspace(lo, hi, 51)
    th, _ = np.histogram(t, bins=edges, density=True)
    ph, _ = np.histogram(p, bins=edges, density=True)
    ref = np.sum(np.abs(ph - th)) / (np.sum(np.abs(th)) + 1e-12)
    assert climate_histogram_error(t, p) == pytest.approx(ref, rel=1e-9, abs=1e-12)
    assert climate_histogram_error(t, np.full_like(t, np.nan)) == 1.0
    half = p.copy()
    half[::2] = np.nan
    assert climate_histogram_error(t, half) > climate_histogram_error(t, p[1::2]) + 0.2


def test_clip_lookup_per_run_dir_and_flat_only_when_allowed(tmp_path):
    from surge.report import demo

    anim = tmp_path / "anim"
    (anim / "runA").mkdir(parents=True)
    (anim / "runA" / "hpo_evolution_dark.webm").write_bytes(b"A")
    (anim / "hpo_evolution_dark.webm").write_bytes(b"FLAT")
    rd = tmp_path / "runs" / "runA"
    other = tmp_path / "runs" / "runB"
    assert demo._clip(rd, "hpo_evolution", "dark", anim) == b"A"
    assert demo._clip(other, "hpo_evolution", "dark", anim, allow_flat=True) == b"FLAT"
    # multiple runs: runB must not borrow the flat clip -> falls to render (None: no run dir)
    assert demo._clip(other, "hpo_evolution", "dark", anim, allow_flat=False) is None
