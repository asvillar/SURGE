"""
Plotting utilities for SURGE model training history.

Quick usage::

    from surge.model.plot_training import plot_training_history

    model.fit(X_train, y_train)
    plot_training_history(model)                        # show
    plot_training_history(model, save_path="loss.png")  # save

    # During training — tail the log file and plot the partial history:
    plot_training_history(log_file="logs/training.jsonl", save_path="live.png")
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Optional, Union

__all__ = ["plot_training_history", "load_training_history"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_training_history(
    source: Union[str, pathlib.Path, list[dict]],
    *,
    run_index: int = -1,
) -> list[dict]:
    """Load training history from a model, list, or JSONL log file.

    Parameters
    ----------
    source:
        One of:
        - A fitted model with a ``training_history`` attribute.
        - A ``list[dict]`` already in memory.
        - A path (``str`` or ``pathlib.Path``) to a JSONL file written by
          :class:`~surge.model.backends._progress.ProgressList`.
    run_index:
        When loading from a multi-run JSONL file, which run to load.
        ``-1`` (default) returns the **last** run.  ``0`` returns the first.

    Returns
    -------
    list[dict]
        Epoch records, each a dict with at least ``{"epoch": int, "train_loss": float}``.
    """
    if hasattr(source, "training_history"):
        return list(source.training_history)  # type: ignore[union-attr]

    if isinstance(source, list):
        return source

    path = pathlib.Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")

    raw = path.read_text(encoding="utf-8").splitlines()
    records: list[dict] = [json.loads(line) for line in raw if line.strip()]

    # Split into runs separated by {"__run_start__": true} sentinels.
    runs: list[list[dict]] = []
    current: list[dict] = []
    for rec in records:
        if rec.get("__run_start__"):
            if current:
                runs.append(current)
            current = []
        else:
            current.append(rec)
    if current:
        runs.append(current)

    if not runs:
        return records  # No sentinel — return everything as one run.

    if run_index >= len(runs) or run_index < -len(runs):
        raise IndexError(
            f"run_index={run_index} out of range; file contains {len(runs)} run(s)."
        )
    return runs[run_index]


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_training_history(
    source: Any = None,
    *,
    log_file: Optional[Union[str, pathlib.Path]] = None,
    run_index: int = -1,
    save_path: Optional[Union[str, pathlib.Path]] = None,
    title: Optional[str] = None,
    show: bool = True,
    figsize: tuple[float, float] = (7, 4),
    smoothing: float = 0.0,
) -> Any:
    """Plot training (and optional validation) loss curves.

    Parameters
    ----------
    source:
        Fitted model **or** ``list[dict]`` history.  Pass ``None`` and use
        ``log_file`` to load from disk.
    log_file:
        Path to a JSONL log file (alternative to ``source``).
    run_index:
        Which run to display from a multi-run log file (default: last).
    save_path:
        If given, save the figure here (PNG/PDF/SVG; format inferred from extension).
        If ``None`` and ``show=True``, the figure is displayed interactively.
    title:
        Figure title.  Auto-generated from model class name if not given.
    show:
        Call ``plt.show()`` after plotting (default ``True``).
    figsize:
        ``(width, height)`` in inches.
    smoothing:
        Exponential moving-average smoothing factor in ``[0, 1)``.
        ``0`` (default) → no smoothing.  ``0.8`` → heavy smoothing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  pip install matplotlib"
        ) from exc

    # --- Resolve source ---
    if source is None and log_file is not None:
        source = log_file
    if source is None:
        raise ValueError("Provide a model, history list, or log_file=...")

    history = load_training_history(source, run_index=run_index)

    if not history:
        raise ValueError("Training history is empty — was the model trained?")

    # --- Extract series ---
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history if "val_loss" in r]
    has_val = len(val_loss) == len(epochs)

    def _smooth(values: list[float], alpha: float) -> list[float]:
        if alpha <= 0:
            return values
        out, s = [], values[0]
        for v in values:
            s = alpha * s + (1 - alpha) * v
            out.append(s)
        return out

    train_plot = _smooth(train_loss, smoothing)
    val_plot = _smooth(val_loss, smoothing) if has_val else []

    # --- Auto title ---
    if title is None:
        if hasattr(source, "__class__") and hasattr(source, "training_history"):
            title = f"{type(source).__name__} — Training loss"
        elif log_file is not None:
            title = f"Training loss — {pathlib.Path(log_file).stem}"
        else:
            title = "Training loss"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_loss, color="#cccccc", linewidth=0.8, alpha=0.6)
    ax.plot(
        epochs, train_plot,
        label="train loss", color="#1f77b4", linewidth=1.8,
    )
    if has_val:
        ax.plot(epochs, val_loss, color="#ddaaaa", linewidth=0.8, alpha=0.6)
        ax.plot(
            epochs, val_plot,
            label="val loss", color="#d62728", linewidth=1.8, linestyle="--",
        )

    # Mark early-stop epoch if present
    early_stop_epochs = [r["epoch"] for r in history if r.get("early_stop")]
    if early_stop_epochs:
        ax.axvline(
            early_stop_epochs[-1], color="gray", linestyle=":",
            linewidth=1.2, label=f"early stop (ep {early_stop_epochs[-1]})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_training_histories(
    histories: dict[str, Any],
    *,
    metric: str = "train_loss",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    title: Optional[str] = None,
    show: bool = True,
    figsize: tuple[float, float] = (8, 5),
    smoothing: float = 0.0,
) -> Any:
    """Compare training curves for multiple models on one plot.

    Parameters
    ----------
    histories:
        Mapping from model label (str) to model / history-list / log-file path.
    metric:
        Which key in the epoch record to plot.  Defaults to ``"train_loss"``.
    save_path, title, show, figsize, smoothing:
        Same as :func:`plot_training_history`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required.  pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=figsize)

    for label, src in histories.items():
        try:
            hist = load_training_history(src)
            epochs = [r["epoch"] for r in hist]
            values = [r.get(metric, float("nan")) for r in hist]
        except Exception as exc:
            print(f"[warn] could not load history for {label!r}: {exc}")
            continue

        def _smooth(vals: list[float], alpha: float) -> list[float]:
            if alpha <= 0:
                return vals
            out, s = [], vals[0]
            for v in vals:
                s = alpha * s + (1 - alpha) * v
                out.append(s)
            return out

        smooth_vals = _smooth(values, smoothing)
        ax.plot(epochs, values, linewidth=0.7, alpha=0.4)
        ax.plot(epochs, smooth_vals, linewidth=1.8, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"Training comparison — {metric}")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
