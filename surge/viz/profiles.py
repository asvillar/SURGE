"""Visualization utilities focused on multi-output scientific profiles."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
from pathlib import Path

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


def plot_profile_comparison_with_inset(
    radius: Sequence[float],
    y_true: Sequence[float],
    predictions: Dict[str, Sequence[float]],
    *,
    case_label: Optional[str] = None,
    case_metadata: Optional[Dict[str, Union[str, float]]] = None,
    mse_values: Optional[Dict[str, float]] = None,
    category: Optional[str] = None,  # "Good", "Average", "Poor"
    inset_region: Optional[Tuple[float, float]] = None,  # (x_min, x_max) for inset
    inset_zoom: Optional[float] = None,  # Zoom factor label for inset
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    xlabel: str = r"$\rho$ [-]",
    ylabel: Optional[str] = None,
    units: Optional[str] = None,
    output_name: Optional[str] = None,
    line_styles: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
) -> Tuple[object, object]:
    """
    Plot profile comparison with optional inset for detailed view.
    
    Creates publication-quality profile plots with main view and optional
    inset zoom, matching the style of Reference vs Surrogate profile comparisons.
    
    Parameters
    ----------
    radius : Sequence[float]
        Radius or coordinate array (x-axis values).
    y_true : Sequence[float]
        Ground truth profile values.
    predictions : dict
        Dictionary mapping model names to predicted profiles.
        Format: {'RFR': y_pred_rfr, 'MLP': y_pred_mlp, ...}
    case_label : str, optional
        Case label/identifier (e.g., "case 9476").
    case_metadata : dict, optional
        Dictionary of case parameters to display (e.g., {'N_φ': 8, 'n_e0': 4.9e19}).
    mse_values : dict, optional
        Dictionary mapping model names to MSE values.
        Format: {'RFR': 1.2e-8, 'MLP': 8.6e-8}.
    category : str, optional
        Category label: "Good", "Average", or "Poor".
    inset_region : tuple, optional
        Tuple (x_min, x_max) defining inset region. If None, no inset created.
    inset_zoom : float, optional
        Zoom label for inset (e.g., 6.55e-04).
    figsize : tuple, default=(8, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    xlabel : str, default=r"$\rho$ [-]"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Auto-generated if None.
    units : str, optional
        Units for y-axis (e.g., "[a.u.]").
    output_name : str, optional
        Output name (e.g., "P_e", "P_H").
    line_styles : dict, optional
        Dictionary mapping model names to line styles.
        Default: {'GT': '-', 'RFR': '--', 'MLP': ':'}
    colors : dict, optional
        Dictionary mapping model names to colors.
        Default: {'GT': 'k', 'RFR': 'r', 'MLP': 'b'}
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Main axes object.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")
    
    from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    radius = np.asarray(radius)
    y_true = np.asarray(y_true)
    
    # Default line styles and colors
    if line_styles is None:
        line_styles = {'GT': '-', 'RFR': '--', 'MLP': ':', 'Reference': '-'}
    if colors is None:
        colors = {'GT': 'k', 'RFR': 'r', 'MLP': 'b', 'Reference': 'k'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ground truth
    gt_label = 'Reference' if 'Reference' in line_styles else 'GT'
    ax.plot(
        radius, y_true,
        color=colors.get('GT', colors.get('Reference', 'k')),
        linestyle=line_styles.get('GT', line_styles.get('Reference', '-')),
        linewidth=2.0,
        label=gt_label,
        alpha=0.9
    )
    
    # Plot predictions
    for model_name, y_pred in predictions.items():
        y_pred_arr = np.asarray(y_pred)
        model_label = f'Surrogate ({model_name})'
        ax.plot(
            radius, y_pred_arr,
            color=colors.get(model_name, f'C{list(predictions.keys()).index(model_name) + 1}'),
            linestyle=line_styles.get(model_name, '--'),
            linewidth=2.0,
            label=model_label,
            alpha=0.9
        )
    
    # Build title
    title_parts = []
    if category:
        title_parts.append(category)
    if case_label:
        title_parts.append(f"({case_label})")
    if output_name:
        title_parts.append(output_name)
    if units:
        title_parts.append(units)
    
    if title_parts:
        title = ": ".join(title_parts)
    else:
        title = "Profile Comparison"
    
    # Add metadata to title or text box
    if case_metadata:
        metadata_str = ", ".join([f"${k} = {v}$" if isinstance(v, (int, float)) else f"{k} = {v}" 
                                  for k, v in case_metadata.items()])
        title = f"{title}\n{metadata_str}"
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Add MSE values if provided
    if mse_values:
        mse_text = ", ".join([f"$MSE_{{{model}}} = {mse:.2e}$" 
                              for model, mse in mse_values.items()])
        ax.text(0.98, 0.02, mse_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel is None:
        ylabel_base = output_name or "Output"
        if units:
            ylabel_base = f"{ylabel_base} {units}"
        ylabel = ylabel_base
    ax.set_ylabel(ylabel, fontsize=11)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Create inset if requested
    ax_inset = None
    if inset_region is not None:
        x_min, x_max = inset_region
        
        # Create inset axes
        ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper right',
                             bbox_to_anchor=(0.02, 0.98, 1, 1), bbox_transform=ax.transAxes,
                             borderpad=2)
        
        # Plot inset data
        mask = (radius >= x_min) & (radius <= x_max)
        if np.any(mask):
            radius_inset = radius[mask]
            y_true_inset = y_true[mask]
            
            ax_inset.plot(
                radius_inset, y_true_inset,
                color=colors.get('GT', colors.get('Reference', 'k')),
                linestyle=line_styles.get('GT', line_styles.get('Reference', '-')),
                linewidth=1.5,
                alpha=0.9
            )
            
            for model_name, y_pred in predictions.items():
                y_pred_arr = np.asarray(y_pred)
                y_pred_inset = y_pred_arr[mask]
                ax_inset.plot(
                    radius_inset, y_pred_inset,
                    color=colors.get(model_name, f'C{list(predictions.keys()).index(model_name) + 1}'),
                    linestyle=line_styles.get(model_name, '--'),
                    linewidth=1.5,
                    alpha=0.9
                )
            
            # Set inset limits with padding
            y_min_inset = min(np.min(y_true_inset), 
                             min([np.min(np.asarray(y_pred)[mask]) for y_pred in predictions.values()]))
            y_max_inset = max(np.max(y_true_inset),
                             max([np.max(np.asarray(y_pred)[mask]) for y_pred in predictions.values()]))
            y_range_inset = y_max_inset - y_min_inset
            padding = y_range_inset * 0.1
            
            ax_inset.set_xlim(x_min, x_max)
            ax_inset.set_ylim(y_min_inset - padding, y_max_inset + padding)
            
            # Inset labels
            if inset_zoom is not None:
                ax_inset.text(0.5, 0.95, f"{inset_zoom:.2e}", 
                             transform=ax_inset.transAxes,
                             fontsize=8, verticalalignment='top', horizontalalignment='center',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax_inset.grid(True, linestyle='--', alpha=0.3)
            ax_inset.tick_params(labelsize=7)
            
            # Mark inset region on main plot with rectangle
            y_min_main = min(np.min(y_true), min([np.min(np.asarray(y_pred)) for y_pred in predictions.values()]))
            y_max_main = max(np.max(y_true), max([np.max(np.asarray(y_pred)) for y_pred in predictions.values()]))
            
            rect = Rectangle((x_min, y_min_main), x_max - x_min, y_max_main - y_min_main,
                           linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.5)
            ax.add_patch(rect)
            
            # Add arrow from rectangle to inset (optional, can be complex)
            # Using a simple annotation
            ax.annotate('', xy=(x_max, y_max_main), xytext=(0.98, 0.98),
                       xycoords='data', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))
    
    # Legend
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Profile comparison with inset saved to: {save_path}")
    
    return fig, ax


