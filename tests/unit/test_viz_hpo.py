"""Tests for surge.viz.hpo (markers + outlier-robust y-limits)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from surge.viz.hpo import plot_hpo_convergence, robust_hpo_ylim


def _synthetic_history(n=150, seed=0, outlier_idx=(24, 100)):
    """150-trial 'campaign' clustered around 0.9-0.98 val_r2, with two
    outlier trials that diverged near zero — the shape that used to blow
    out the y-axis of the convergence plot.
    """
    rng = np.random.default_rng(seed)
    values = 0.9 + 0.08 * rng.random(n)
    for idx in outlier_idx:
        values[idx] = 0.0
    trials = [{"number": i, "value": float(v)} for i, v in enumerate(values)]
    return values, {"trials": trials, "direction": "maximize", "metric": "val_r2"}


# --------------------------------------------------------- robust_hpo_ylim

def test_robust_hpo_ylim_excludes_outliers_for_large_campaign():
    values, _ = _synthetic_history()
    lo, hi, pad, masks = robust_hpo_ylim([values], higher_is_better=True)
    assert lo is not None and hi is not None and pad is not None
    # the y-limits must sit well above the diverged trials' near-zero
    # values — a couple of divergences among 150 healthy trials can no
    # longer collapse the axis to [0, 1]
    assert lo > 0.5
    assert 0.9 <= hi <= 1.0
    # the healthy cluster's median is comfortably inside the rendered band
    (mask,) = masks
    assert lo <= np.median(values[~mask]) <= hi


def test_robust_hpo_ylim_flags_the_injected_outliers():
    """The percentile method flags the bottom slice of the pool (~5% for
    maximize), so it always catches genuine divergences — it just isn't
    required to flag *only* them.
    """
    values, _ = _synthetic_history()
    _, _, _, (mask,) = robust_hpo_ylim([values], higher_is_better=True)
    flagged = set(np.nonzero(mask)[0].tolist())
    assert {24, 100} <= flagged
    # a bounded slice, not the whole cluster
    assert 2 <= mask.sum() <= 20


def test_robust_hpo_ylim_small_campaign_is_passthrough():
    """<=10 trials per series: no robustification, matches old autoscale."""
    values = np.array([0.9, 0.92, 0.0, 0.95, 0.93, 0.91, 0.5, 0.96, 0.94, 0.97])
    lo, hi, pad, (mask,) = robust_hpo_ylim([values], higher_is_better=True)
    assert lo is None and hi is None and pad is None
    assert not mask.any()


def test_robust_hpo_ylim_pooling_does_not_misflag_a_low_but_valid_series():
    """Two 10-trial series (20 pooled points) must still be treated as a
    small campaign — a genuinely low-scoring series (e.g. a weaker model)
    must not get misread as 'diverged' just because another series padded
    the pool past a naive pooled-count threshold.
    """
    mlp = np.array([0.89, 0.90, 0.92, 0.96, 0.95, 0.93, 0.925, 0.938, 0.922, 0.97])
    rf = np.array([0.72, 0.73, 0.68, 0.75, 0.75, 0.71, 0.695, 0.74, 0.72, 0.749])
    lo, hi, pad, masks = robust_hpo_ylim([mlp, rf], higher_is_better=True)
    assert lo is None and hi is None
    assert not any(m.any() for m in masks)


# ------------------------------------------------------ plot_hpo_convergence

def test_plot_hpo_convergence_excludes_outliers_and_saves(tmp_path: Path):
    values, data = _synthetic_history()
    f = tmp_path / "model_hpo.json"
    f.write_text(json.dumps(data))

    fig, ax, best_values = plot_hpo_convergence(
        [f], metric="r2", method_names=["model"], save_path=tmp_path / "out.png")

    assert fig is not None
    assert (tmp_path / "out.png").is_file()

    lo, hi = ax.get_ylim()
    # the diverged trials (value 0.0) must fall outside the rendered range
    assert lo > 0.1
    # the healthy cluster's median must be comfortably inside the range
    assert lo <= np.median(values) <= hi

    # best value must come from the healthy cluster, not a diverged trial
    assert best_values["model"]["value"] > 0.5


def test_plot_hpo_convergence_small_campaign_autoscales(tmp_path: Path):
    """<=10 trials: no percentile clipping, mirrors legacy behaviour."""
    values = [0.9, 0.92, 0.0, 0.95, 0.93, 0.91, 0.5, 0.96, 0.94, 0.97]
    trials = [{"number": i, "value": v} for i, v in enumerate(values)]
    data = {"trials": trials, "direction": "maximize", "metric": "val_r2"}
    f = tmp_path / "small_hpo.json"
    f.write_text(json.dumps(data))

    fig, ax, best_values = plot_hpo_convergence(
        [f], metric="r2", method_names=["small"], save_path=tmp_path / "out.png")

    assert fig is not None
    assert (tmp_path / "out.png").is_file()
    # autoscale must include the low trial (0.0) — no clipping for small n
    lo, _hi = ax.get_ylim()
    assert lo <= 0.1
    assert best_values["small"]["value"] == pytest.approx(0.97)
