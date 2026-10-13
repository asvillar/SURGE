"""Tests for surge.viz.animate (synthetic run dir, tiny frame counts)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from surge.viz import animate

N_TRIALS = 12


@pytest.fixture()
def run_dir(tmp_path):
    rng = np.random.default_rng(0)
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    lines = []
    for i in range(N_TRIALS):
        params = {"n_blocks": 2, "hidden_layers_0_log": 5.0 + rng.random(),
                  "hidden_layers_1_log": 5.0 + rng.random(),
                  "learning_rate": float(10 ** rng.uniform(-4, -2)),
                  "batch_size": 128, "dropout_rate": float(rng.random() * 0.3)}
        lines.append(json.dumps({"trial": i, "value": 0.9 + 0.005 * i + 0.01 * rng.random(),
                                 "metric": "val_r2", "params": params}))
        hist = [{"epoch": e, "train_loss": 1.0 / (e + 1) + 0.01 * i,
                 "val_loss": 1.2 / (e + 1) + 0.02} for e in range(1, 15)]
        (run / f"hpo_trial_{i:04d}_training_history.json").write_text(json.dumps(hist))
    (run / "hpo_trials_manifest.jsonl").write_text("\n".join(lines))
    (run / "metrics.json").write_text(json.dumps(
        {"m": {"test": {"r2": 0.97, "rmse": 0.8}, "timings": {}}}))
    import pandas as pd
    y = rng.normal(size=300)
    pd.DataFrame({"y_true_a": y, "y_pred_a": y + 0.1 * rng.normal(size=300)}
                 ).to_parquet(run / "predictions" / "m_test.parquet")
    return run


@pytest.fixture()
def reports(tmp_path):
    base = tmp_path / "reports" / "bench.x"
    for i, (m, r2) in enumerate([("a.one", 0.5), ("b.two", 0.95), ("c.three", 0.92)]):
        d = base / f"2026-01-0{i + 1}T00:00:00Z"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(
            {"model_key": m, "timestamp": f"2026-01-0{i + 1}T00:00:00Z",
             "metrics": {"test_r2": r2}}))
    return tmp_path / "reports"


def _sizes(paths):
    return {Image.open(p).size for p in paths}


def test_hpo_evolution(run_dir, tmp_path):
    paths = animate.hpo_evolution_frames(run_dir, tmp_path / "o", "dark", n_frames=5)
    assert len(paths) == 5 and len(list((tmp_path / "o").glob("f_*.png"))) == 5
    assert len(_sizes(paths)) == 1


def test_mission_control(run_dir, tmp_path):
    paths = animate.mission_control_frames(run_dir, tmp_path / "o", "light", n_frames=6)
    assert len(paths) == 6 and len(_sizes(paths)) == 1


def test_leaderboard(reports, tmp_path):
    paths = animate.leaderboard_frames(reports, "bench.x", tmp_path / "o", "dark",
                                       threshold=0.9)
    assert len(paths) == 3 + 8 and len(_sizes(paths)) == 1


def test_training(run_dir, tmp_path):
    paths = animate.training_frames(run_dir / "hpo_trial_0003_training_history.json",
                                    tmp_path / "o", "light", n_frames=6)
    assert len(paths) == 6 and len(_sizes(paths)) == 1


def test_n_shown_endpoints():
    assert animate._n_shown(0, 60, 200) == 1
    assert animate._n_shown(59, 60, 200) == 200


@pytest.mark.skipif(animate._ffmpeg() is None, reason="ffmpeg missing")
def test_video_and_gif(run_dir, tmp_path):
    animate.training_frames(run_dir / "hpo_trial_0003_training_history.json",
                            tmp_path / "f", "dark", n_frames=4)
    mp4, webm = animate.frames_to_video(tmp_path / "f", tmp_path / "v" / "clip", fps=4)
    assert mp4.stat().st_size > 0 and webm.stat().st_size > 0
    gif = animate.frames_to_gif(tmp_path / "f", tmp_path / "v" / "clip", fps=4)
    assert gif.stat().st_size > 0
