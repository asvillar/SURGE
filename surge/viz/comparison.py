"""Comparison visualization functions for SURGE model results.

Provides functions to create publication-quality comparison plots including:
- Inference comparison grids (model×dataset×output)
- MSE comparison plots (log-log scale)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_inference_comparison_grid(
    results_dict: Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    output_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    bins: int = 100,
    cmap: str = 'plasma_r',
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    units: Optional[Dict[str, str]] = None,
    show_colorbar: bool = True,
    xlabel_prefix: str = "Reference",
    ylabel_prefix: str = "Surrogate",
) -> Tuple[Optional[object], Optional[object], Dict[str, Dict[str, float]]]:
    """
    Create a grid of inference comparison plots matching publication style.
    
    Creates a grid layout comparing multiple models across train/test sets
    and different outputs. Similar to the Reference vs Surrogate comparison plots.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary structure:
        {
            'Pe': {
                'RFR': {'train': (y_true, y_pred), 'test': (y_true, y_pred)},
                'MLP': {'train': (y_true, y_pred), 'test': (y_true, y_pred)}
            },
            'PII': {...}
        }
    output_names : list, optional
        List of output names in order. If None, uses keys from results_dict.
    model_names : list, optional
        List of model names in order. If None, uses keys from first output.
    bins : int, default=100
        Number of bins for 2D histogram.
    cmap : str, default='plasma_r'
        Colormap for density heatmap.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    units : dict, optional
        Dictionary mapping output names to unit strings (e.g., {'Pe': '[a.u.]'})
    show_colorbar : bool, default=True
        Whether to show colorbar on each subplot.
    xlabel_prefix : str, default="Reference"
        Prefix for x-axis label (ground truth source).
    ylabel_prefix : str, default="Surrogate"
        Prefix for y-axis label (prediction source, will include model name).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : np.ndarray
        Array of axes objects.
    results : dict
        Dictionary of R² scores: {output: {model: {dataset: r2}}}
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    if not results_dict:
        raise ValueError("results_dict cannot be empty")
    
    # Extract output and model names
    if output_names is None:
        output_names = list(results_dict.keys())
    if model_names is None:
        first_output = output_names[0]
        model_names = list(results_dict[first_output].keys())
    
    # Determine layout: rows = outputs, columns = models × datasets
    datasets = ['train', 'test']
    n_outputs = len(output_names)
    n_cols = len(model_names) * len(datasets)
    n_rows = n_outputs
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Store results
    r2_results = {}
    
    # Plot each combination
    plot_idx = 0
    subplot_labels = 'abcdefghijklmnopqrstuvwxyz'
    
    for row_idx, output_name in enumerate(output_names):
        if output_name not in results_dict:
            continue
            
        output_data = results_dict[output_name]
        output_unit = units.get(output_name, '') if units else ''
        
        # Set consistent axis limits for this output across all subplots
        all_values = []
        for model_name in model_names:
            if model_name not in output_data:
                continue
            for dataset in datasets:
                if dataset not in output_data[model_name]:
                    continue
                y_true, y_pred = output_data[model_name][dataset]
                all_values.extend([y_true.flatten(), y_pred.flatten()])
        
        if all_values:
            all_values = np.concatenate(all_values)
            value_range = [np.min(all_values), np.max(all_values)]
            # Add small padding
            padding = (value_range[1] - value_range[0]) * 0.05
            axis_lim = [max(0, value_range[0] - padding), value_range[1] + padding]
        else:
            axis_lim = [0, 1]
        
        for col_idx, model_name in enumerate(model_names):
            if model_name not in output_data:
                continue
                
            for dataset_idx, dataset in enumerate(datasets):
                if dataset not in output_data[model_name]:
                    continue
                
                ax = axes[row_idx, col_idx * len(datasets) + dataset_idx]
                y_true, y_pred = output_data[model_name][dataset]
                
                # Ensure 1D arrays
                y_true = np.asarray(y_true).flatten()
                y_pred = np.asarray(y_pred).flatten()
                
                # Calculate R²
                r2 = r2_score(y_true, y_pred)
                
                # Store results
                if output_name not in r2_results:
                    r2_results[output_name] = {}
                if model_name not in r2_results[output_name]:
                    r2_results[output_name][model_name] = {}
                r2_results[output_name][model_name][dataset] = float(r2)
                
                # Create 2D histogram for density heatmap
                hist, xedges, yedges = np.histogram2d(
                    y_true, y_pred, bins=bins, range=[axis_lim, axis_lim]
                )
                
                # Plot density heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(
                    hist.T,
                    origin='lower',
                    extent=extent,
                    aspect='auto',
                    cmap=cmap,
                    interpolation='nearest',
                    alpha=0.8
                )
                
                # Add diagonal reference line
                ax.plot(axis_lim, axis_lim, 'k--', linewidth=1.5, alpha=0.6, label='y=x')
                
                # Add R² score text
                ax.text(
                    0.05, 0.95,
                    f'R² = {r2:.2f}',
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                
                # Set axis labels
                if row_idx == n_rows - 1:
                    # Bottom row: show x-axis labels
                    dataset_label = "Training Set" if dataset == 'train' else "Test Set"
                    ax.set_xlabel(f'{xlabel_prefix} {output_unit}'.strip(), fontsize=10)
                else:
                    ax.set_xlabel('')
                    ax.tick_params(labelbottom=False)
                
                if col_idx == 0 and dataset_idx == 0:
                    # First column: show y-axis labels
                    model_label = f'{ylabel_prefix}({model_name})'
                    ax.set_ylabel(f'{model_label} {output_unit}'.strip(), fontsize=10)
                
                # Set title for top row (only on first column of each output)
                if row_idx == 0 and col_idx == 0 and dataset_idx == 0:
                    title = f"({subplot_labels[plot_idx]}) {output_name} {output_unit}".strip()
                    ax.set_title(title, fontsize=11, fontweight='bold')
                elif row_idx == 0:
                    # Other columns in top row
                    title = f"({subplot_labels[plot_idx]}) {output_name} {output_unit}".strip()
                    ax.set_title(title, fontsize=11, fontweight='bold')
                
                # Set axis limits
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                
                # Add colorbar (only on last column subplots)
                if show_colorbar and col_idx == len(model_names) - 1 and dataset_idx == len(datasets) - 1:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(f'Number of data points ({dataset})', fontsize=9)
                
                # Grid
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                plot_idx += 1
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Inference comparison grid saved to: {save_path}")
    
    return fig, axes, r2_results


def plot_mse_comparison(
    mse_model1: np.ndarray,
    mse_model2: np.ndarray,
    output_names: Optional[List[str]] = None,
    model1_name: str = "RFR",
    model2_name: str = "MLP",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    show_trend: bool = True,
    trend_alpha: float = 0.7,
    scatter_alpha: float = 0.6,
    units: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[object], Optional[object], Dict[str, float]]:
    """
    Create log-log MSE comparison plots matching publication style.
    
    Creates scatter plots comparing MSE between two models, with optional
    trend curves. Supports multiple outputs as subplots.
    
    Parameters
    ----------
    mse_model1 : np.ndarray
        MSE values for model 1. Shape: (n_samples,) or (n_outputs, n_samples).
    mse_model2 : np.ndarray
        MSE values for model 2. Shape should match mse_model1.
    output_names : list, optional
        Names for outputs if multi-output. Length should match n_outputs.
    model1_name : str, default="RFR"
        Name of first model (x-axis).
    model2_name : str, default="MLP"
        Name of second model (y-axis).
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    show_trend : bool, default=True
        Whether to show trend curve with markers.
    trend_alpha : float, default=0.7
        Transparency for trend curve.
    scatter_alpha : float, default=0.6
        Transparency for scatter points.
    units : dict, optional
        Dictionary mapping output names to unit strings.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : np.ndarray or matplotlib.axes.Axes
        Axes object(s).
    correlations : dict
        Dictionary of correlation coefficients per output.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    mse_model1 = np.asarray(mse_model1)
    mse_model2 = np.asarray(mse_model2)
    
    # Handle multi-output case
    if mse_model1.ndim > 1:
        n_outputs = mse_model1.shape[0]
        is_multi_output = True
    else:
        n_outputs = 1
        is_multi_output = False
        mse_model1 = mse_model1.reshape(1, -1)
        mse_model2 = mse_model2.reshape(1, -1)
    
    if output_names is None:
        output_names = [f'Output {i+1}' for i in range(n_outputs)]
    
    # Calculate figure size
    if figsize is None:
        if n_outputs == 1:
            figsize = (6, 6)
        else:
            figsize = (6 * n_outputs, 5)
    
    # Create subplots
    if n_outputs == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_outputs, figsize=figsize, squeeze=False)
        axes = axes.flatten()
    
    correlations = {}
    subplot_labels = 'abcdefghijklmnop'
    
    for idx, (ax, output_name) in enumerate(zip(axes, output_names)):
        mse1 = mse_model1[idx, :]
        mse2 = mse_model2[idx, :]
        
        # Remove zeros/NaN for log scale
        valid_mask = (mse1 > 0) & (mse2 > 0) & np.isfinite(mse1) & np.isfinite(mse2)
        mse1_valid = mse1[valid_mask]
        mse2_valid = mse2[valid_mask]
        
        if len(mse1_valid) == 0:
            continue
        
        # Scatter plot
        ax.scatter(
            mse1_valid, mse2_valid,
            alpha=scatter_alpha,
            color='orange',
            s=20,
            edgecolors='none'
        )
        
        # Diagonal reference line (y=x)
        min_val = min(np.min(mse1_valid), np.min(mse2_valid))
        max_val = max(np.max(mse1_valid), np.max(mse2_valid))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.6)
        
        # Trend curve (if requested)
        if show_trend and len(mse1_valid) > 10:
            # Sort by x values
            sort_idx = np.argsort(mse1_valid)
            x_sorted = mse1_valid[sort_idx]
            y_sorted = mse2_valid[sort_idx]
            
            # Bin and average for smoother curve
            n_bins = min(20, len(x_sorted) // 5)
            if n_bins > 1:
                bin_edges = np.logspace(
                    np.log10(np.min(x_sorted)), 
                    np.log10(np.max(x_sorted)), 
                    n_bins + 1
                )
                bin_centers = []
                bin_means = []
                for i in range(len(bin_edges) - 1):
                    mask = (x_sorted >= bin_edges[i]) & (x_sorted < bin_edges[i + 1])
                    if np.any(mask):
                        bin_centers.append(np.mean(x_sorted[mask]))
                        bin_means.append(np.mean(y_sorted[mask]))
                
                if len(bin_centers) > 2:
                    # Plot trend curve
                    ax.plot(
                        bin_centers, bin_means,
                        'b-', linewidth=2, alpha=trend_alpha, label='Trend'
                    )
                    # Add markers at select points
                    n_markers = min(3, len(bin_centers))
                    marker_indices = np.linspace(0, len(bin_centers) - 1, n_markers, dtype=int)
                    ax.plot(
                        [bin_centers[i] for i in marker_indices],
                        [bin_means[i] for i in marker_indices],
                        'cs', markersize=8, alpha=trend_alpha, label='Trend points'
                    )
        
        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Calculate correlation
        if len(mse1_valid) > 1:
            correlation = np.corrcoef(np.log10(mse1_valid), np.log10(mse2_valid))[0, 1]
            correlations[output_name] = float(correlation)
        
        # Labels
        unit_str = units.get(output_name, '') if units else ''
        title = f"({subplot_labels[idx]}) {output_name} {unit_str}".strip()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'MSE_{model1_name}', fontsize=11)
        ax.set_ylabel(f'MSE_{model2_name}', fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        
        # Set equal aspect for log scale
        ax.set_aspect('auto')
    
    if show_trend and len(mse1_valid) > 10:
        fig.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ MSE comparison plot saved to: {save_path}")
    
    return fig, axes if n_outputs > 1 else axes[0], correlations

