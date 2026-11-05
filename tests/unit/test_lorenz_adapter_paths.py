"""Adapter-path Lorenz-63: HPO trial VPT, prediction capture, timing, demo clips."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from surge.benchmarks.leaderboard import _check_pass, gate_metric_key


def test_check_pass_is_pure_and_gate_key_exposed():
    metrics = {"test_nrmse": 0.1}
    assert gate_metric_key("sequence.lorenz63") == "test_vpt_median"
    assert gate_metric_key("no.such.benchmark") is None
    assert _check_pass("sequence.lorenz63", metrics) is False
    assert metrics == {"test_nrmse": 0.1}  # no mutation


def test_adapter_lorenz_saves_predictions_and_vpt(tmp_path):
    from surge.benchmarks.leaderboard import _run_with_adapter
    from surge.model.registry import MODEL_REGISTRY

    adapter = MODEL_REGISTRY.create("sklearn.ridge")
    res = _run_with_adapter("sequence.lorenz63", adapter, seed=0)
    assert "test_vpt_median" in res.metrics
    assert all(isinstance(v, (int, float)) for v in res.metrics.values())
    assert res.metrics["runtime_s"] > 0
    out = res.save(root=tmp_path)
    assert list(Path(tmp_path).rglob("predictions.npz"))
    assert out is not None


def test_hpo_trial_runner_emits_vpt():
    from sklearn.model_selection import train_test_split

    from surge.benchmarks.hpo import _run_with_adapter_data
    from surge.benchmarks.tasks import _generate_lorenz_trajectories
    from surge.model.registry import MODEL_REGISTRY

    X, y = _generate_lorenz_trajectories(n_trajectories=300, dt=0.01, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    res = _run_with_adapter_data(
        "sequence.lorenz63", MODEL_REGISTRY.create("sklearn.ridge"),
        Xtr, Xte, ytr, yte, task_type="regression", tier="0", seed=0)
    assert res is not None
    assert "test_vpt_median" in res.metrics
    assert "gate_metric_missing" not in res.extra
    assert all(isinstance(v, (int, float)) for v in res.metrics.values())


def test_clip_returns_bytes_and_cleans_tempdir(monkeypatch):
    from surge.report import demo
    from surge.viz import animate

    def fake_frames(run_dir, out, mode):
        Path(out).mkdir(parents=True, exist_ok=True)

    def fake_video(frames, out):
        Path(out).with_suffix(".webm").write_bytes(b"WEBM")

    monkeypatch.setattr(animate, "hpo_evolution_frames", fake_frames)
    monkeypatch.setattr(animate, "frames_to_video", fake_video)
    tmpdir = Path(tempfile.gettempdir())
    before = {p.name for p in tmpdir.glob("surge_hpo_evolution_*")}
    data = demo._clip(Path("."), "hpo_evolution", "dark", None)
    after = {p.name for p in tmpdir.glob("surge_hpo_evolution_*")}
    assert data == b"WEBM"
    assert after == before

    def boom(frames, out):
        raise RuntimeError("ffmpeg missing")

    monkeypatch.setattr(animate, "frames_to_video", boom)
    assert demo._clip(Path("."), "hpo_evolution", "dark", None) is None
    assert {p.name for p in tmpdir.glob("surge_hpo_evolution_*")} == before
