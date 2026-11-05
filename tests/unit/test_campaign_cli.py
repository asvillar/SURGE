"""Run-directory figures, ``surge viz`` / ``surge monitor`` / ``surge data`` (no network)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surge.cli import main
from surge.viz.campaign import (
    detect_hpo_model,
    detect_models,
    hpo_figure,
    mission_control_figure,
    render_run,
)

REPO = Path(__file__).resolve().parents[2]
MODELS = ("mlp_a", "rf_b")
HPO_MODEL = "mlp_a"


@pytest.fixture()
def run_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "run"
    (d / "predictions").mkdir(parents=True)
    metrics = {m: {"train": {"r2": 0.99}, "val": {"r2": 0.95},
                   "test": {"r2": 0.93 + 0.01 * i, "rmse": 1.5, "mae": 1.0},
                   "timings": {"train_seconds": 3.2}}
               for i, m in enumerate(MODELS)}
    (d / "metrics.json").write_text(json.dumps(metrics))
    values = [0.80, 0.90, 0.85, 0.95]
    rows = []
    for n, v in enumerate(values):
        rows.append({"trial": n, "value": v, "metric": "val_r2",
                     "is_best_yet": v >= max(values[: n + 1]),
                     "params": {"learning_rate": 10 ** rng.uniform(-4, -2),
                                "dropout_rate": float(rng.uniform(0, 0.3))}})
        hist = [{"epoch": e, "train_loss": 1.0 / e + 0.01 * n,
                 "val_loss": 1.2 / e + 0.02 * n} for e in range(1, 9)]
        (d / f"hpo_trial_{n:04d}_training_history.json").write_text(json.dumps(hist))
    (d / "hpo_trials_manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    for m in MODELS:
        y = rng.uniform(0, 10, 400)
        for split in ("train", "test"):
            pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(0, 0.3, 400)}
                         ).to_parquet(d / "predictions" / f"{m}_{split}.parquet")
        (d / f"model_card_{m}.json").write_text(json.dumps({"model_name": m}))
    (d / "spec.yaml").write_text("run_tag: synthetic\n")
    (d / "run.log").write_text(
        "2026-01-01 10:00:00 SURGE workflow started.\n"
        "2026-01-01 10:02:30 workflow finished.\n")
    return d


def test_detect_models_and_hpo_model(run_dir, tmp_path):
    assert detect_models(run_dir) == list(MODELS)
    assert detect_hpo_model(run_dir) == HPO_MODEL
    assert detect_models(tmp_path) == []
    assert detect_hpo_model(tmp_path) is None


def test_render_run_writes_four_pngs(run_dir, tmp_path):
    out = tmp_path / "figs"
    written = render_run(run_dir, out_dir=out)
    assert set(written) == {"mission_control_light", "hpo_convergence_light",
                            "training_curves_light", "parity_light"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 1000


def test_render_run_default_dir_only_and_missing_hpo(run_dir):
    written = render_run(run_dir, only=["parity"])
    assert list(written) == ["parity_light"]
    assert written["parity_light"].parent == run_dir / "figures"
    (run_dir / "hpo_trials_manifest.jsonl").unlink()
    assert hpo_figure(run_dir) is None
    assert mission_control_figure(run_dir) is None
    assert "mission_control_light" not in render_run(run_dir, modes=("light",))


def test_surge_viz_cli(run_dir, tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["viz", str(run_dir), "--out", str(out), "--modes", "light",
                 "--only", "mission_control", "parity", "--model", HPO_MODEL]) == 0
    assert (out / "mission_control_light.png").exists()
    assert (out / "parity_light.png").exists()
    assert "mission_control_light.png" in capsys.readouterr().out


def test_surge_viz_empty_dir_exits_2(tmp_path, capsys):
    assert main(["viz", str(tmp_path)]) == 2
    assert "metrics.json" in capsys.readouterr().err


def test_surge_monitor_finished(run_dir, capsys):
    assert main(["monitor", str(run_dir)]) == 0
    out = capsys.readouterr().out
    for n in range(4):
        assert f"trial   {n}" in out
    assert "best: trial 3" in out
    assert "best so far" in out
    assert "2m30s" in out
    assert "test:" in out and "r2=0.93" in out and "rf_b" in out


def test_surge_monitor_shows_trial_in_progress(run_dir, capsys):
    (run_dir / "metrics.json").unlink()
    hist = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4}]
    (run_dir / "hpo_trial_0004_training_history.json").write_text(json.dumps(hist))
    assert main(["monitor", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "RUNNING" in out and "trial   4  (running)" in out


def test_surge_monitor_watch_exits_when_finished(run_dir, capsys):
    assert main(["monitor", str(run_dir), "--watch", "0.01"]) == 0
    assert capsys.readouterr().out.count("SURGE monitor") == 1


def _fake_xy(n=6, nx=3):
    X = np.arange(n * nx, dtype=float).reshape(n, nx)
    return X, X.sum(axis=1)


def test_export_benchmark_named_columns(tmp_path, monkeypatch):
    from surge.benchmarks import leaderboard
    from surge.benchmarks.export import export_benchmark

    monkeypatch.setattr(leaderboard, "_load_dataset", lambda key: _fake_xy(5, 10))
    df = export_benchmark("plasma.qlknn_transport", tmp_path / "q.csv")
    assert list(df.columns)[:3] == ["Ati", "Ate", "Ane"]
    assert df.columns[-1] == "efeITG" and len(df) == 5
    assert list(pd.read_csv(tmp_path / "q.csv").columns) == list(df.columns)
    with pytest.raises(ValueError):
        export_benchmark("plasma.qlknn_transport", tmp_path / "q.csv", split="test")


def test_export_benchmark_fixed_split_and_generic_names(tmp_path, monkeypatch):
    from surge.benchmarks import leaderboard
    from surge.benchmarks.export import export_benchmark

    def fake_fixed():
        Xa, ya = _fake_xy(4, 3)
        Xb, yb = _fake_xy(2, 3)
        return Xa, ya, Xb, yb

    monkeypatch.setitem(leaderboard._FIXED_SPLIT_LOADERS, "plasma.qlknn_transport",
                        "_fake_fixed_loader")
    monkeypatch.setattr(leaderboard, "_fake_fixed_loader", fake_fixed, raising=False)
    df = export_benchmark("plasma.qlknn_transport", tmp_path / "f.parquet")
    assert list(df.columns) == ["x0", "x1", "x2", "efeITG", "split"]
    assert df["split"].tolist() == ["train"] * 4 + ["test"] * 2
    test_only = export_benchmark("plasma.qlknn_transport", tmp_path / "t.csv",
                                 split="test")
    assert set(test_only["split"]) == {"test"} and len(test_only) == 2


def test_surge_data_export_cli_round_trip(tmp_path, monkeypatch, capsys):
    from surge.benchmarks import leaderboard

    monkeypatch.setattr(leaderboard, "_load_dataset", lambda key: _fake_xy(5, 10))
    out = tmp_path / "q.csv"
    assert main(["data", "export", "qlknn", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "shape=5 x 11" in printed and "efeITG" in printed
    assert pd.read_csv(out).shape == (5, 11)
    assert main(["data", "export", "no.such_benchmark", str(out)]) == 2


def test_surge_data_list(capsys):
    assert main(["data", "list"]) == 0
    out = capsys.readouterr().out
    assert "plasma.qlknn_transport" in out and "vision.cifar10" in out


@pytest.mark.skipif(not (REPO / "runs" / "qlknn_hpo_40").is_dir(),
                    reason="runs/qlknn_hpo_40 not present")
def test_gallery_still_builds_mission_control(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "examples" / "viz_theme_gallery.py"),
         "--only", "mission_control", "--hpo-run", str(REPO / "runs" / "qlknn_hpo_40"),
         "--modes", "light", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "mission_control_light.png").exists()
