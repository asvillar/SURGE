"""Training dynamics plots from SURGE neural backends.

Reads from:
  - a list of per-epoch dicts (``adapter.training_history``), or
  - ``Path`` to ``training_log.jsonl`` (one JSON object per line), or
  - ``Path`` to ``training_history_*.json`` (JSON array).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MPL_AVAILABLE = False
    plt = None
    Figure = Any


HistoryEntry = Mapping[str, Any]
HistorySource = Sequence[HistoryEntry] | str | Path


def load_training_history(source: HistorySource) -> list[dict[str, Any]]:
    """Normalize inputs into a list of epoch dicts."""
    if isinstance(source, (list, tuple)):
        return [dict(row) for row in source]
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Training history not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        raise ValueError(f"Expected JSON array in {path}")
    raise ValueError(f"Unsupported training history format: {path}")


def plot_loss_curve(
    history: HistorySource,
    *,
    metrics: list[str] | None = None,
    log_scale: bool = False,
    title: str = "Training history",
    save_path: Path | str | None = None,
    ax=None,
) -> Any:
    """Plot train / val loss vs epoch; optional extra metric lines."""
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_loss_curve")
    hist = load_training_history(history)
    if not hist:
        raise ValueError("Empty training history")
    epochs = [int(h["epoch"]) for h in hist]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, [h["train_loss"] for h in hist], label="train_loss", color="C0")
    if any(h.get("val_loss") is not None for h in hist):
        ax.plot(
            epochs,
            [h.get("val_loss") if h.get("val_loss") is not None else np.nan for h in hist],
            label="val_loss",
            color="C1",
        )
    for m in metrics or []:
        if m in {"train_loss", "val_loss"}:
            continue
        ys = []
        for h in hist:
            v = h.get(m)
            ys.append(float(v) if v is not None else np.nan)
        if any(np.isfinite(ys)):
            ax.plot(epochs, ys, label=m, linestyle="--")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / metric")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig = ax.figure
    if save_path is not None:
        _save_figure(fig, Path(save_path))
    return fig


def plot_lr_schedule(
    history: HistorySource,
    *,
    title: str = "Learning rate",
    save_path: Path | str | None = None,
    ax=None,
) -> Any:
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_lr_schedule")
    hist = load_training_history(history)
    if not hist or not any("lr" in h for h in hist):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No lr logged", ha="center", va="center")
            ax.set_axis_off()
        fig = ax.figure
        if save_path is not None:
            _save_figure(fig, Path(save_path))
        return fig
    epochs = [int(h["epoch"]) for h in hist]
    lrs = [float(h.get("lr", np.nan)) for h in hist]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, lrs, color="C2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig = ax.figure
    if save_path is not None:
        _save_figure(fig, Path(save_path))
    return fig


def plot_training_dashboard(
    history: HistorySource,
    *,
    model_name: str = "",
    save_path: Path | str | None = None,
) -> Any:
    """Multi-panel: loss, primary val metric (rmse if present), LR."""
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_training_dashboard")
    hist = load_training_history(history)
    if not hist:
        raise ValueError("Empty training history")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    suptitle = "Training dashboard"
    if model_name:
        suptitle = f"{suptitle} — {model_name}"
    fig.suptitle(suptitle)

    epochs = [int(h["epoch"]) for h in hist]
    axes[0].plot(epochs, [h["train_loss"] for h in hist], label="train_loss")
    if any(h.get("val_loss") is not None for h in hist):
        axes[0].plot(
            epochs,
            [h.get("val_loss") if h.get("val_loss") is not None else np.nan for h in hist],
            label="val_loss",
        )
    axes[0].set_ylabel("loss (MSE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Prefer RMSE in scaled space when available
    if any("val_rmse_scaled" in h and h.get("val_rmse_scaled") is not None for h in hist):
        axes[1].plot(
            epochs,
            [h.get("train_rmse_scaled", np.nan) for h in hist],
            label="train_rmse_scaled",
        )
        axes[1].plot(
            epochs,
            [h.get("val_rmse_scaled") if h.get("val_rmse_scaled") is not None else np.nan for h in hist],
            label="val_rmse_scaled",
        )
        axes[1].set_ylabel("RMSE (scaled)")
    elif any(h.get("val_r2") is not None for h in hist):
        axes[1].plot(epochs, [h.get("val_r2", np.nan) for h in hist], label="val_r2")
        axes[1].set_ylabel("R²")
    else:
        axes[1].text(0.5, 0.5, "No val metric in history", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if any("lr" in h for h in hist):
        axes[2].plot(epochs, [float(h.get("lr", np.nan)) for h in hist], color="C2")
        axes[2].set_ylabel("lr")
    else:
        axes[2].text(0.5, 0.5, "No lr logged", ha="center", va="center")
        axes[2].set_axis_off()
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        _save_figure(fig, Path(save_path))
    return fig


def compare_training_curves(
    histories: Mapping[str, HistorySource],
    *,
    metric: str = "val_loss",
    log_scale: bool = False,
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Any:
    """Overlay one curve per named run (e.g. benchmark comparison)."""
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for compare_training_curves")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, src in histories.items():
        hist = load_training_history(src)
        epochs = [int(h["epoch"]) for h in hist]
        ys = [h.get(metric) for h in hist]
        ys_f = [float(y) if y is not None else np.nan for y in ys]
        ax.plot(epochs, ys_f, label=name)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Training curves ({metric})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        _save_figure(fig, Path(save_path))
    return fig


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "load_training_history",
    "plot_loss_curve",
    "plot_lr_schedule",
    "plot_training_dashboard",
    "compare_training_curves",
]
