"""Tests for surge.viz.training (Phase 1 benchmark plan)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surge.viz.training import (
    compare_training_curves,
    load_training_history,
    plot_loss_curve,
    plot_lr_schedule,
    plot_training_dashboard,
)


def _sample_history():
    return [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.55, "train_rmse_scaled": 0.7, "val_rmse_scaled": 0.72, "lr": 1e-3},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.35, "train_rmse_scaled": 0.55, "val_rmse_scaled": 0.59, "lr": 1e-3},
        {"epoch": 3, "train_loss": 0.2, "val_loss": 0.28, "train_rmse_scaled": 0.45, "val_rmse_scaled": 0.52, "lr": 5e-4},
    ]


def test_load_training_history_from_list():
    h = _sample_history()
    out = load_training_history(h)
    assert len(out) == 3
    assert out[0]["epoch"] == 1


def test_load_training_history_jsonl(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    lines = "\n".join(json.dumps(r) for r in _sample_history())
    p.write_text(lines + "\n", encoding="utf-8")
    out = load_training_history(p)
    assert len(out) == 3


def test_load_training_history_json_array(tmp_path: Path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps(_sample_history()), encoding="utf-8")
    out = load_training_history(p)
    assert len(out) == 3


def test_plot_loss_curve_returns_figure(tmp_path: Path):
    fig = plot_loss_curve(_sample_history(), save_path=tmp_path / "loss.png")
    assert fig is not None
    assert (tmp_path / "loss.png").is_file()


def test_plot_lr_schedule(tmp_path: Path):
    fig = plot_lr_schedule(_sample_history(), save_path=tmp_path / "lr.png")
    assert fig is not None
    assert (tmp_path / "lr.png").is_file()


def test_plot_training_dashboard(tmp_path: Path):
    fig = plot_training_dashboard(
        _sample_history(),
        model_name="test_mlp",
        save_path=tmp_path / "dash.png",
    )
    assert fig is not None
    assert (tmp_path / "dash.png").is_file()


def test_compare_training_curves(tmp_path: Path):
    h1 = _sample_history()
    h2 = [{"epoch": i + 1, "val_loss": 0.5 - i * 0.1} for i in range(3)]
    fig = compare_training_curves(
        {"run_a": h1, "run_b": h2},
        metric="val_loss",
        save_path=tmp_path / "cmp.png",
    )
    assert fig is not None
    assert (tmp_path / "cmp.png").is_file()


def test_plot_loss_curve_empty_raises():
    with pytest.raises(ValueError, match="Empty"):
        plot_loss_curve([])
