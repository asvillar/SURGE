"""HPO (Hyperparameter Optimization) visualization functions for SURGE.

Provides functions to create publication-quality plots for hyperparameter
optimization convergence analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def robust_hpo_ylim(
    values_by_series: List[np.ndarray],
    higher_is_better: bool,
    pad_frac: float = 0.08,
    small_campaign_n: int = 10,
) -> Tuple[Optional[float], Optional[float], Optional[float], List[np.ndarray]]:
    """Shared robust-y-limit + diverged-trial logic for HPO score panels.

    One or more series share a single y-axis (per-trial score vs. trial
    number). A handful of diverged trials (e.g. val_r2 ~ 0 amid 200 healthy
    trials) must not collapse the axis, so the limits come from the
    5th-100th percentile (maximize) or 0th-95th percentile (minimize) of
    the *pooled, finite* values across all series, padded a little.

    "Small campaign" is judged per series (the largest one), not by the
    pooled trial count — a two-model overlay of two 10-trial runs is still
    small even though pooled they total 20 points; otherwise a merely
    low-scoring trial in one series gets misread as "diverged" once
    another series pads the pool past the threshold.

    Returns ``(lo, hi, pad, diverged_masks)``. ``lo``/``hi``/``pad`` are
    ``None`` when every series is a small campaign (caller should leave
    the axis on plain autoscale). ``diverged_masks`` is a list, one
    boolean mask per input series, flagging finite values that fall
    outside ``[lo, hi]`` (all-``False`` when limits are ``None``).
    """
    arrays = [np.asarray(v, dtype=float) for v in values_by_series]
    max_n = max((a.size for a in arrays), default=0)
    if max_n <= small_campaign_n:
        return None, None, None, [np.zeros_like(a, dtype=bool) for a in arrays]

    pooled = np.concatenate(arrays) if arrays else np.array([])
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        return None, None, None, [np.zeros_like(a, dtype=bool) for a in arrays]

    lo, hi = np.percentile(finite, [5, 100] if higher_is_better else [0, 95])
    span = hi - lo
    pad = span * pad_frac if span > 0 else max(abs(hi), 1.0) * pad_frac
    masks = []
    for a in arrays:
        finite_a = np.isfinite(a)
        masks.append(finite_a & (a < lo) if higher_is_better
                     else finite_a & (a > hi))
    return float(lo), float(hi), float(pad), masks


def plot_hpo_convergence(
    hpo_files: Union[str, Path, List[Union[str, Path]]],
    metric: str = "r2",
    method_names: Optional[List[str]] = None,
    reference_value: Optional[float] = None,
    reference_label: str = "Reference",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    show_best_markers: bool = True,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
) -> Tuple[object, object, Dict[str, Dict[str, float]]]:
    """
    Plot HPO convergence curves matching publication style.
    
    Creates line plots showing optimization progress over iterations,
    with support for multiple optimization methods (Random, TPE, BoTorch).
    
    Parameters
    ----------
    hpo_files : str, Path, or list
        Path(s) to HPO JSON file(s). If multiple files, assumes different methods.
        Or can be list of paths for multiple methods.
    metric : str, default="r2"
        Metric to plot: "r2", "rmse", "mse". Note: HPO files use "val_rmse" by default.
        Will auto-detect from file if possible.
    method_names : list, optional
        Names for each method (e.g., ["Random", "Optuna TPE", "Optuna BoTorch"]).
        If None, tries to infer from filenames or uses generic names.
    reference_value : float, optional
        Reference line value (e.g., baseline R² = 0.96).
    reference_label : str, default="Reference"
        Label for reference line.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    show_best_markers : bool, default=True
        Whether to mark best points with stars.
    colors : dict, optional
        Dictionary mapping method names to colors.
        Default: {'Random': 'green', 'TPE': 'red', 'BoTorch': 'blue'}
    linestyles : dict, optional
        Dictionary mapping method names to the running-best step line style.
        Default: {'Random': '--', 'TPE': '--', 'BoTorch': '--'}
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    best_values : dict
        Dictionary of best values per method: {method: {'value': float, 'iteration': int}}
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    import json
    
    # Handle single file or list of files
    if isinstance(hpo_files, (str, Path)):
        hpo_files = [hpo_files]
    
    # Load HPO data
    hpo_data_list = []
    for hpo_file in hpo_files:
        hpo_path = Path(hpo_file)
        if not hpo_path.exists():
            raise FileNotFoundError(f"HPO file not found: {hpo_path}")
        
        with open(hpo_path, 'r') as f:
            data = json.load(f)
        hpo_data_list.append(data)
    
    # Determine method names
    if method_names is None:
        method_names = []
        for i, hpo_file in enumerate(hpo_files):
            filename = Path(hpo_file).stem
            # Try to infer from filename
            if 'random' in filename.lower():
                method_names.append('Random')
            elif 'tpe' in filename.lower() or 'optuna' in filename.lower():
                method_names.append('Optuna TPE')
            elif 'botorch' in filename.lower():
                method_names.append('Optuna BoTorch')
            else:
                method_names.append(f'Method {i+1}')
    elif len(method_names) != len(hpo_data_list):
        raise ValueError(f"Number of method names ({len(method_names)}) must match number of files ({len(hpo_data_list)})")
    
    # Default colors and styles
    if colors is None:
        colors = {
            'Random': 'green',
            'Optuna TPE': 'red',
            'Optuna BoTorch': 'blue',
            'TPE': 'red',
            'BoTorch': 'blue'
        }
    if linestyles is None:
        linestyles = {name: '--' for name in method_names}
    
    # Determine metric direction
    if metric.lower() == "r2":
        higher_is_better = True
        metric_key = "r2"
    elif metric.lower() in ("rmse", "mse"):
        higher_is_better = False
        # HPO files typically store "val_rmse" in the "value" field
        metric_key = "value"
    else:
        higher_is_better = False
        metric_key = metric
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    best_values = {}

    # ---- pass 1: extract per-method series ----------------------------
    series_data = []
    for method_name, hpo_data in zip(method_names, hpo_data_list):
        trials = hpo_data.get('trials', [])

        if not trials:
            continue

        iterations = []
        values = []

        for trial in trials:
            trial_value = trial.get('value')
            if trial_value is not None:
                iterations.append(trial['number'])
                values.append(float(trial_value))

        if not values:
            continue

        iterations = np.array(iterations)
        values = np.array(values)

        # Sort by iteration
        sort_idx = np.argsort(iterations)
        iterations_sorted = iterations[sort_idx]
        values_sorted = values[sort_idx]

        running_best = (np.maximum if higher_is_better else np.minimum) \
            .accumulate(values_sorted)

        color = colors.get(method_name, f'C{method_names.index(method_name)}')
        series_data.append({
            'name': method_name,
            'iterations': iterations_sorted,
            'values': values_sorted,
            'running_best': running_best,
            'color': color,
        })

    # ---- robust y-limits + diverged-trial masks (shared helper) -------
    outlier_lo, outlier_hi, outlier_pad, diverged_masks = robust_hpo_ylim(
        [s['values'] for s in series_data], higher_is_better)
    if outlier_lo is not None:
        ax.set_ylim(outlier_lo - outlier_pad, outlier_hi + outlier_pad)

    # ---- pass 2: markers, running-best step, star + axes annotation ---
    n_diverged_total = 0
    for i, s in enumerate(series_data):
        method_name = s['name']
        color = s['color']
        iterations_sorted = s['iterations']
        values_sorted = s['values']
        running_best = s['running_best']
        linestyle = linestyles.get(method_name, '--')

        diverged = diverged_masks[i]
        n_diverged_total += int(diverged.sum())

        # per-trial markers (in-range trials only)
        ax.scatter(
            iterations_sorted[~diverged], values_sorted[~diverged],
            color=color, s=22, alpha=0.75, edgecolors='none',
            label=method_name, zorder=2,
        )
        # diverged trials: clamp to the axis edge as small muted triangles
        if diverged.any():
            clamp_y = outlier_lo if higher_is_better else outlier_hi
            marker = 'v' if higher_is_better else '^'
            ax.scatter(
                iterations_sorted[diverged],
                np.full(int(diverged.sum()), clamp_y),
                marker=marker, color='gray', s=28, alpha=0.8,
                zorder=2, label=None,
            )

        # running best: dashed step line
        ax.step(
            iterations_sorted, running_best, where='post',
            color=color, linestyle=linestyle, linewidth=1.8,
            alpha=0.9, zorder=3, label=None,
        )

        # Find best value
        if higher_is_better:
            best_idx = np.argmax(values_sorted)
        else:
            best_idx = np.argmin(values_sorted)
        best_value = values_sorted[best_idx]
        best_iteration = iterations_sorted[best_idx]

        # Store best values
        best_values[method_name] = {
            'value': float(best_value),
            'iteration': int(best_iteration)
        }

        # Mark best point
        if show_best_markers:
            ax.plot(
                best_iteration, best_value,
                marker='*',
                markersize=15,
                color='yellow',
                markeredgecolor='black',
                markeredgewidth=1,
                zorder=5,
                label=None
            )
            # Annotation lives in axes coordinates (stacked per method,
            # upper-left) so it never covers the title or the trial cloud.
            label = (f'{method_name}: R² = {best_value:.4f}'
                      if metric.lower() == "r2"
                      else f'{method_name}: {best_value:.4f}')
            ax.text(
                0.02, 0.97 - i * 0.08, label,
                transform=ax.transAxes, fontsize=8.5,
                ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=color, alpha=0.9),
            )

    if n_diverged_total:
        # legend proxy (not ax.text) so the muted label sits in the
        # reserved legend box instead of overlapping the clamped
        # triangles it describes
        ax.scatter(
            [], [], marker='v' if higher_is_better else '^',
            color='gray', s=28, label=f'{n_diverged_total} diverged',
        )

    # Add reference line
    if reference_value is not None:
        ax.axhline(
            reference_value,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f'{reference_label} (R²={reference_value:.2f})' if metric.lower() == "r2" else f'{reference_label} ({reference_value:.2f})'
        )
    
    # Set labels
    if metric.lower() == "r2":
        ylabel = "R² Score"
        title = "Hyperparameter Optimization - RFR (P_e)"
    elif metric.lower() == "rmse":
        ylabel = "RMSE"
        title = "Hyperparameter Optimization - Validation RMSE"
    else:
        ylabel = metric.upper()
        title = f"Hyperparameter Optimization - {metric.upper()}"
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='lower right' if higher_is_better else 'upper right',
              fontsize=10)
    
    plt.tight_layout()
    
    # Set axis limits
    ax.set_xlim(left=0)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ HPO convergence plot saved to: {save_path}")
    
    return fig, ax, best_values


def plot_hpo_comparison(
    hpo_files_dict: Dict[str, Union[str, Path]],
    metric: str = "r2",
    reference_value: Optional[float] = None,
    reference_label: str = "Reference",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Tuple[object, object, Dict[str, Dict[str, float]]]:
    """
    Plot comparison of multiple HPO methods from different files.
    
    Convenience wrapper around plot_hpo_convergence that accepts a dictionary
    mapping method names to file paths.
    
    Parameters
    ----------
    hpo_files_dict : dict
        Dictionary mapping method names to HPO file paths.
        Format: {'Random': 'path1.json', 'TPE': 'path2.json', ...}
    metric : str, default="r2"
        Metric to plot.
    reference_value : float, optional
        Reference line value.
    reference_label : str, default="Reference"
        Label for reference line.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    best_values : dict
        Dictionary of best values per method.
    """
    hpo_files = list(hpo_files_dict.values())
    method_names = list(hpo_files_dict.keys())
    
    return plot_hpo_convergence(
        hpo_files=hpo_files,
        metric=metric,
        method_names=method_names,
        reference_value=reference_value,
        reference_label=reference_label,
        figsize=figsize,
        save_path=save_path,
        dpi=dpi,
    )

