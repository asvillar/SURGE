"""Tests for surge.viz.classification (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from surge.viz.classification import (
    plot_classification_dashboard,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def test_plot_roc_binary_1d(tmp_path: Path):
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.35, 0.6, 0.85])
    fig = plot_roc_curve(y_true, y_prob, save_path=tmp_path / "roc.png")
    assert fig is not None
    assert (tmp_path / "roc.png").is_file()


def test_plot_pr_and_confusion_multiclass(tmp_path: Path):
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=80)
    y_prob = rng.random((80, 3))
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1)
    fig = plot_precision_recall_curve(y_true, y_prob, save_path=tmp_path / "pr.png")
    assert (tmp_path / "pr.png").is_file()
    fig2 = plot_confusion_matrix(
        y_true, y_pred, labels=["a", "b", "c"], save_path=tmp_path / "cm.png"
    )
    assert (tmp_path / "cm.png").is_file()


def test_classification_dashboard_binary(tmp_path: Path):
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.55, 0.7, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)
    fig = plot_classification_dashboard(
        y_true,
        y_pred,
        y_prob,
        model_name="test",
        save_path=tmp_path / "dash.png",
    )
    assert fig is not None
    assert (tmp_path / "dash.png").is_file()
    assert (tmp_path / "dash.pdf").is_file()
