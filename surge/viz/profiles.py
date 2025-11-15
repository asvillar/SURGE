"""Visualization utilities focused on multi-output scientific profiles."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:  # pragma: no cover
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


def compute_profile_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    output_groups: Mapping[str, Sequence[str]],
) -> Mapping[str, Mapping[str, float]]:
    """Return per-profile metrics (RMSE, MAE, R2) for grouped outputs."""
    metrics = {}
    for group_name, columns in output_groups.items():
        if not columns:
            continue
        true_vals = y_true[columns].to_numpy()
        pred_vals = y_pred[columns].to_numpy()
        mse = mean_squared_error(true_vals, pred_vals, multioutput="uniform_average")
        metrics[group_name] = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(true_vals, pred_vals, multioutput="uniform_average")),
            "r2": float(r2_score(true_vals, pred_vals, multioutput="uniform_average")),
        }
    return metrics


def plot_profile_band(
    radius: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    label: str,
    color: str = "C0",
    ax=None,
    fill_alpha: float = 0.15,
    line_alpha: float = 0.9,
):
    """Plot ground truth vs prediction bands for a single profile."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")
    ax = ax or plt.gca()
    radius = np.asarray(radius)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ax.plot(radius, y_true, color=color, linewidth=2.0, alpha=line_alpha, label=f"{label} (GT)")
    ax.plot(radius, y_pred, color=color, linestyle="--", linewidth=2.0, alpha=line_alpha, label=f"{label} (Pred)")
    ax.fill_between(radius, y_true, y_pred, color=color, alpha=fill_alpha)
    ax.set_xlabel("Radius / normalized coordinate")
    ax.set_ylabel(label)
    ax.grid(True, linestyle="--", alpha=0.3)
    return ax


def plot_density_scatter(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    bins: int = 120,
    ax=None,
    cmap: str = "viridis",
    title: Optional[str] = None,
):
    """Plot GT vs prediction density for a flattened profile."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")
    ax = ax or plt.gca()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    hist, xedges, yedges = np.histogram2d(y_true, y_pred, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Count")
    ax.plot(extent[:2], extent[:2], "k--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    return ax


