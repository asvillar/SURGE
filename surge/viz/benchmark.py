"""Benchmark comparison visualizations (benchmark plan §3.3).

Reads from :class:`~surge.benchmarks.base.BenchmarkResult` objects or
``benchmark_reports/*/result.json`` files on disk.

All functions follow the same save contract as the rest of ``surge/viz/``:
- *save_path*: if provided, saves PNG + PDF side-by-side.
- Returns the ``matplotlib.figure.Figure`` for interactive use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.figure import Figure

    MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MPL_AVAILABLE = False
    plt = None
    Figure = Any


def _ensure_mpl() -> None:
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for benchmark plots")


def _save_figure(fig: Any, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if save_path.suffix.lower() == ".png":
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")


# Metrics where lower is better (affects coloring and axis direction).
_LOWER_IS_BETTER: frozenset[str] = frozenset({"runtime_s", "test_rmse"})

# Human-readable metric labels for axis titles.
_METRIC_LABELS: dict[str, str] = {
    "test_accuracy": "Accuracy",
    "test_f1_macro": "F1 (macro)",
    "test_auroc": "AUROC",
    "test_r2": "R²",
    "test_rmse": "RMSE",
    "runtime_s": "Runtime (s)",
}

# Color scheme: green for pass, coral for fail.
_PASS_COLOR = "#4CAF50"
_FAIL_COLOR = "#EF5350"
_BEST_COLOR = "#1565C0"   # highlighted best bar
_BASE_COLOR = "#90A4AE"   # other bars


def plot_benchmark_leaderboard(
    results: list[Any],
    *,
    metric: str = "test_r2",
    title: str | None = None,
    baseline_model: str | None = None,
    highlight_best: bool = True,
    save_path: Path | None = None,
    ax: Any = None,
) -> Any:
    """
    Horizontal bar chart: one bar per model, sorted by *metric*.

    Parameters
    ----------
    results:
        List of :class:`~surge.benchmarks.base.BenchmarkResult` objects
        (all must share the same ``benchmark_key``).
    metric:
        Metric column to plot (must be present in ``result.metrics``).
    title:
        Plot title.  Defaults to ``"<benchmark_key> — <metric>"``.
    baseline_model:
        If set, the bar for this model key is drawn in a distinct colour.
    highlight_best:
        Mark the best bar with a star annotation.
    save_path:
        Write PNG + PDF to this path.
    ax:
        Draw into existing axes (for subplot composition).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    if not results:
        raise ValueError("results list is empty")

    lower_better = metric in _LOWER_IS_BETTER
    sorted_results = sorted(
        results,
        key=lambda r: r.metrics.get(metric, float("-inf")),
        reverse=not lower_better,
    )

    model_names = [r.model_key for r in sorted_results]
    values = [r.metrics.get(metric, float("nan")) for r in sorted_results]
    passed = [r.passed for r in sorted_results]

    best_idx = (
        np.nanargmin(values) if lower_better else np.nanargmax(values)
    )

    bk = sorted_results[0].benchmark_key
    metric_label = _METRIC_LABELS.get(metric, metric)
    if title is None:
        title = f"{bk} — {metric_label}"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(model_names) + 1.2)))
    else:
        fig = ax.get_figure()

    colors = []
    for i, (mn, p) in enumerate(zip(model_names, passed)):
        if mn == baseline_model:
            colors.append("#FF9800")  # orange for baseline
        elif highlight_best and i == best_idx:
            colors.append(_BEST_COLOR)
        elif not p:
            colors.append(_FAIL_COLOR)
        else:
            colors.append(_BASE_COLOR)

    bars = ax.barh(range(len(model_names)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel(metric_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # Annotate values on bars.
    for i, (bar, v) in enumerate(zip(bars, values)):
        if np.isnan(v):
            continue
        label = f"{v:.4f}" if metric != "runtime_s" else f"{v:.2f}s"
        star = " ★" if highlight_best and i == best_idx else ""
        ax.text(
            bar.get_width() + 0.001 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            bar.get_y() + bar.get_height() / 2,
            f"{label}{star}",
            va="center",
            ha="left",
            fontsize=8,
            color=_BEST_COLOR if (highlight_best and i == best_idx) else "black",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    direction = "↓ lower is better" if lower_better else "↑ higher is better"
    ax.set_xlabel(f"{metric_label}  ({direction})", fontsize=9)

    if own_fig:
        fig.tight_layout()
        _save_figure(fig, save_path)
    return fig


def plot_metric_table(
    results: list[Any],
    *,
    metrics: list[str] | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> Any:
    """
    Styled matplotlib table — models as rows, metrics as columns.

    Best value per column is bold + blue; worst is italic + red.
    A PASS / FAIL column is included.

    Parameters
    ----------
    results:
        List of :class:`~surge.benchmarks.base.BenchmarkResult` objects.
    metrics:
        Subset of metric keys to show.  Defaults to all numeric metrics
        found across the results (in a preferred display order).
    title:
        Table title.
    save_path:
        Write PNG + PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    if not results:
        raise ValueError("results list is empty")

    _METRIC_ORDER = [
        "test_accuracy", "test_f1_macro", "test_auroc",
        "test_r2", "test_rmse", "runtime_s",
    ]

    if metrics is None:
        seen: set[str] = set()
        metrics = []
        for mk in _METRIC_ORDER:
            if any(mk in r.metrics for r in results):
                metrics.append(mk)
                seen.add(mk)
        for r in results:
            for mk in sorted(r.metrics):
                if mk not in seen:
                    metrics.append(mk)
                    seen.add(mk)

    model_names = [r.model_key for r in results]
    bk = results[0].benchmark_key
    if title is None:
        title = f"Metric Table — {bk}"

    # Build data matrix.
    data: list[list[str]] = []
    float_matrix: list[list[float | None]] = []
    for r in results:
        row_f = [r.metrics.get(m) for m in metrics]
        float_matrix.append(row_f)
        row_s = []
        for m, v in zip(metrics, row_f):
            if v is None:
                row_s.append("—")
            elif m == "runtime_s":
                row_s.append(f"{v:.2f}s")
            else:
                row_s.append(f"{v:.4f}")
        row_s.append("✓" if r.passed else "✗")
        data.append(row_s)

    col_labels = [_METRIC_LABELS.get(m, m) for m in metrics] + ["Pass"]

    # Find best/worst per column.
    best_per_col: list[int | None] = []
    worst_per_col: list[int | None] = []
    for ci, mk in enumerate(metrics):
        col = [float_matrix[ri][ci] for ri in range(len(results))]
        numeric = [(ri, v) for ri, v in enumerate(col) if v is not None]
        if not numeric:
            best_per_col.append(None)
            worst_per_col.append(None)
            continue
        if mk in _LOWER_IS_BETTER:
            best_per_col.append(min(numeric, key=lambda x: x[1])[0])
            worst_per_col.append(max(numeric, key=lambda x: x[1])[0])
        else:
            best_per_col.append(max(numeric, key=lambda x: x[1])[0])
            worst_per_col.append(min(numeric, key=lambda x: x[1])[0])
    best_per_col.append(None)
    worst_per_col.append(None)

    n_rows = len(results)
    n_cols = len(col_labels)
    fig_h = max(2.2, 0.45 * n_rows + 1.2)
    fig_w = max(5, 1.5 * n_cols + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        rowLabels=model_names,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(n_cols)))

    # Style header.
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")

    # Style row labels.
    for i in range(n_rows):
        cell = table[(i + 1, -1)]
        cell.set_facecolor("#F5F5F5")
        cell.set_text_props(fontweight="bold")

    # Highlight best (blue bold) and worst (red italic) per column.
    for ci in range(n_cols):
        bi = best_per_col[ci]
        wi = worst_per_col[ci]
        for ri in range(n_rows):
            cell = table[(ri + 1, ci)]
            if bi is not None and ri == bi:
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(color=_BEST_COLOR, fontweight="bold")
            elif wi is not None and ri == wi and wi != bi:
                cell.set_facecolor("#FFF3E0")
                cell.set_text_props(color="#BF360C", style="italic")

    # Pass/fail column.
    pass_col = n_cols - 1
    for ri, r in enumerate(results):
        cell = table[(ri + 1, pass_col)]
        if r.passed:
            cell.set_facecolor("#E8F5E9")
            cell.set_text_props(color="#2E7D32", fontweight="bold")
        else:
            cell.set_facecolor("#FFEBEE")
            cell.set_text_props(color="#C62828", fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_multi_benchmark_dashboard(
    results_by_benchmark: dict[str, list[Any]],
    *,
    metric: str | None = None,
    save_path: Path | None = None,
) -> Any:
    """
    Multi-panel figure: one leaderboard bar chart per benchmark.

    Parameters
    ----------
    results_by_benchmark:
        ``{benchmark_key: [BenchmarkResult, ...]}`` dict as returned by
        :func:`~surge.benchmarks.leaderboard.run_leaderboard`.
    metric:
        Which metric to plot.  If ``None``, auto-selects the primary metric
        per benchmark (``test_r2`` for regression, ``test_accuracy`` for
        classification).
    save_path:
        Write PNG + PDF.
    """
    _ensure_mpl()
    if not results_by_benchmark:
        raise ValueError("results_by_benchmark is empty")

    bks = [k for k, v in results_by_benchmark.items() if v]
    n = len(bks)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    for i, bk in enumerate(bks):
        rl = results_by_benchmark[bk]
        if not rl:
            axes_flat[i].axis("off")
            continue
        # Auto-select metric.
        m = metric
        if m is None:
            task = rl[0].task_type
            m = "test_accuracy" if task == "classification" else "test_r2"
        # Fall back to first available metric.
        if not any(m in r.metrics for r in rl):
            m = next(
                (k for r in rl for k in r.metrics if isinstance(r.metrics[k], float)),
                None,
            )
        if m is None:
            axes_flat[i].axis("off")
            continue
        try:
            plot_benchmark_leaderboard(rl, metric=m, ax=axes_flat[i])
        except Exception:
            axes_flat[i].axis("off")

    # Hide unused axes.
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("SURGE Benchmark Leaderboard", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def load_benchmark_results(root: Path) -> dict[str, list[Any]]:
    """
    Load all saved :class:`~surge.benchmarks.base.BenchmarkResult` objects
    from a ``benchmark_reports/`` directory tree.

    Parameters
    ----------
    root:
        Root path (e.g. ``Path("benchmark_reports")``).

    Returns
    -------
    ``{benchmark_key: [BenchmarkResult, ...]}``
    """
    from surge.benchmarks.base import BenchmarkResult

    out: dict[str, list[BenchmarkResult]] = {}
    for result_file in sorted(Path(root).rglob("result.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            r = BenchmarkResult(
                benchmark_key=data["benchmark_key"],
                tier=data.get("tier", "?"),
                task_type=data.get("task_type", "unknown"),
                metrics=data.get("metrics", {}),
                passed=data.get("passed", False),
                model_key=data.get("model_key", ""),
                message=data.get("message", ""),
                extra=data.get("extra", {}),
                timestamp=data.get("timestamp", ""),
                surge_version=data.get("surge_version", ""),
            )
            out.setdefault(r.benchmark_key, []).append(r)
        except Exception:
            continue
    return out
