"""Data analysis visualization functions for SURGE.

Provides functions to create comprehensive data analysis plots including:
- Input/output distributions (violin plots)
- Profile mean ± std deviation
- Signal-to-noise ratio analysis
- Correlation heatmaps
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    SEABORN_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
        SEABORN_AVAILABLE = False
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        SEABORN_AVAILABLE = False


def plot_input_distributions(
    df: pd.DataFrame,
    input_columns: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    normalize: bool = True,
    max_inputs: int = 10,
) -> Tuple[object, object]:
    """
    Plot violin plots for input variable distributions matching publication style.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing input columns.
    input_columns : list, optional
        List of input column names. If None, uses all numeric columns.
    figsize : tuple, default=(12, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    normalize : bool, default=True
        Whether to normalize values to [0, 1] range for display.
    max_inputs : int, default=10
        Maximum number of inputs to plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    if input_columns is None:
        input_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if len(input_columns) > max_inputs:
        input_columns = input_columns[:max_inputs]
    
    # Prepare data
    plot_data = []
    plot_labels = []
    plot_ranges = []
    
    for col in input_columns:
        values = df[col].dropna().values
        
        if normalize:
            # Normalize to [0, 1]
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
            range_str = f"{min_val:.1e}-{max_val:.1e}"
        else:
            range_str = f"{np.min(values):.1e}-{np.max(values):.1e}"
        
        plot_data.append(values)
        plot_labels.append(f"{col}")
        plot_ranges.append(range_str)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot violin plots
    if SEABORN_AVAILABLE:
        # Use seaborn for better violin plots
        plot_df = pd.DataFrame({label: data for label, data in zip(plot_labels, plot_data)})
        plot_df_melted = plot_df.melt(var_name='Variable', value_name='Normalized Value')
        
        violin_parts = ax.violinplot(
            plot_data,
            positions=range(len(plot_labels)),
            showmeans=True,
            showmedians=True
        )
        
        # Customize violin plots
        for pc in violin_parts['bodies']:
            pc.set_facecolor('C0')
            pc.set_alpha(0.7)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in violin_parts:
                violin_parts[partname].set_color('black')
                violin_parts[partname].set_linewidth(1.5)
    else:
        # Use matplotlib violin plot
        violin_parts = ax.violinplot(
            plot_data,
            positions=range(len(plot_labels)),
            showmeans=True,
            showmedians=True
        )
        
        for pc in violin_parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
    
    # Set labels
    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Normalized Values' if normalize else 'Values', fontsize=11)
    ax.set_title('Input Variable Distributions', fontsize=12, fontweight='bold')
    
    # Add range labels
    for i, (label, range_str) in enumerate(zip(plot_labels, plot_ranges)):
        ax.text(i, ax.get_ylim()[1] * 0.95, range_str, 
               ha='center', va='top', fontsize=8, alpha=0.7)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Input distributions plot saved to: {save_path}")
    
    return fig, ax


def plot_output_distributions(
    df: pd.DataFrame,
    output_columns: Optional[Sequence[str]] = None,
    sample_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    max_outputs: int = 10,
    ylabel: str = "Output value",
    title: str = "Output distributions",
) -> Tuple[object, object]:
    """
    Plot violin plots for output profile distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing output columns.
    output_columns : list, optional
        List of output column names (e.g., ['y_0', 'y_50', ...]).
        If None, uses all numeric columns.
    sample_indices : list, optional
        Indices to sample (e.g., [0, 50, 100, ...]). If None, uses all columns.
    figsize : tuple, default=(14, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    max_outputs : int, default=10
        Maximum number of outputs to plot.
    ylabel : str, default="Output value"
        Y-axis label (include units if known).
    title : str, default="Output distributions"
        Axes title.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    if output_columns is None:
        output_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Sample columns if specified
    if sample_indices is not None:
        output_columns = [output_columns[i] for i in sample_indices if i < len(output_columns)]
    
    if len(output_columns) > max_outputs:
        output_columns = output_columns[:max_outputs]
    
    # Prepare data
    plot_data = []
    plot_labels = []
    
    for col in output_columns:
        values = df[col].dropna().values
        plot_data.append(values)
        plot_labels.append(col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot violin plots
    violin_parts = ax.violinplot(
        plot_data,
        positions=range(len(plot_labels)),
        showmeans=True,
        showmedians=True
    )
    
    # Customize
    for pc in violin_parts['bodies']:
        pc.set_facecolor('C1')
        pc.set_alpha(0.7)
    
    # Set labels
    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Output distributions plot saved to: {save_path}")
    
    return fig, ax


def plot_profile_mean_std(
    df: pd.DataFrame,
    output_columns: Sequence[str],
    output_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    output_name: str = "output",
    units: str = "",
) -> Tuple[object, object]:
    """
    Plot mean ± standard deviation for output profiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing output profile columns.
    output_columns : list
        List of output column names (in order).
    output_indices : list, optional
        Indices for x-axis (e.g., [0, 50, 100, ...]). If None, uses [0, 1, 2, ...].
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    output_name : str, default="output"
        Name for output variable (for labels).
    units : str, default=""
        Units for output (e.g. "[m/s]"); omitted when empty.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    if output_indices is None:
        output_indices = range(len(output_columns))
    
    # Extract data
    profile_data = df[output_columns].values
    
    # Calculate mean and std
    mean_profile = np.mean(profile_data, axis=0)
    std_profile = np.std(profile_data, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean
    ax.plot(output_indices, mean_profile, 'g-', linewidth=2, label=f'<{output_name}>', alpha=0.9)
    
    # Plot std band
    ax.fill_between(
        output_indices,
        mean_profile - std_profile,
        mean_profile + std_profile,
        color='green',
        alpha=0.15,
        label='±σ'
    )
    
    # Labels
    ax.set_xlabel('Output Index', fontsize=11)
    ax.set_ylabel(f'{output_name} {units}'.strip(), fontsize=11)
    ax.set_title(f'Output Profile: Mean ± Standard Deviation', fontsize=12, fontweight='bold')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Profile mean±std plot saved to: {save_path}")
    
    return fig, ax


def plot_signal_to_noise(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    output_columns: Optional[Sequence[str]] = None,
    sample_output_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    max_outputs: int = 10,
) -> Tuple[object, object, Dict[str, float]]:
    """
    Plot signal-to-noise ratio (SNR = μ/σ) for inputs and outputs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing input and output columns.
    input_columns : list
        List of input column names.
    output_columns : list, optional
        List of output column names. If None, excludes input columns.
    sample_output_indices : list, optional
        Indices to sample for outputs (e.g., [0, 50, 100, ...]).
    figsize : tuple, default=(14, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    max_outputs : int, default=10
        Maximum number of outputs to plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    snr_dict : dict
        Dictionary of SNR values: {name: snr}
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Calculate SNR for inputs
    input_snr = {}
    for col in input_columns:
        values = df[col].dropna().values
        mean_val = np.mean(np.abs(values))
        std_val = np.std(values)
        if std_val > 0:
            snr = mean_val / std_val
        else:
            snr = 0.0
        input_snr[col] = float(snr)
    
    # Calculate SNR for outputs
    output_snr = {}
    if output_columns is not None:
        if sample_output_indices is not None:
            output_cols = [output_columns[i] for i in sample_output_indices if i < len(output_columns)]
        else:
            output_cols = output_columns
        
        if len(output_cols) > max_outputs:
            # Sample evenly
            step = len(output_cols) // max_outputs
            output_cols = [output_cols[i] for i in range(0, len(output_cols), step)][:max_outputs]
        
        for col in output_cols:
            values = df[col].dropna().values
            mean_val = np.mean(np.abs(values))
            std_val = np.std(values)
            if std_val > 0:
                snr = mean_val / std_val
            else:
                snr = 0.0
            output_snr[col] = float(snr)
    
    # Combine for plotting
    all_labels = list(input_snr.keys()) + list(output_snr.keys())
    all_snr = [input_snr[label] for label in input_snr.keys()] + [output_snr[label] for label in output_snr.keys()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    colors = ['steelblue'] * len(input_snr) + ['orange'] * len(output_snr)
    bars = ax.bar(range(len(all_labels)), all_snr, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Set labels
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Signal-to-Noise Ratio (μ/σ)', fontsize=11)
    ax.set_title('Signal-to-Noise Analysis', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, snr) in enumerate(zip(bars, all_snr)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{snr:.1f}',
               ha='center', va='bottom', fontsize=8)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Signal-to-noise plot saved to: {save_path}")
    
    # Combine SNR dictionary
    snr_dict = {**input_snr, **output_snr}
    
    return fig, ax, snr_dict


def plot_correlation_heatmap(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    sample_output_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    cmap: str = 'RdBu_r',
    max_outputs: int = 400,
) -> Tuple[object, object, pd.DataFrame]:
    """
    Plot correlation heatmap between inputs and outputs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing input and output columns.
    input_columns : list
        List of input column names.
    output_columns : list
        List of output column names.
    sample_output_indices : list, optional
        Indices to sample for outputs. If None, uses all or samples if too many.
    figsize : tuple, default=(12, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    cmap : str, default='RdBu_r'
        Colormap for heatmap.
    max_outputs : int, default=400
        Maximum number of outputs to include (for performance).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    correlation_matrix : pd.DataFrame
        Correlation matrix DataFrame.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Sample outputs if needed
    if sample_output_indices is not None:
        output_cols = [output_columns[i] for i in sample_output_indices if i < len(output_columns)]
    elif len(output_columns) > max_outputs:
        # Sample evenly
        step = len(output_columns) // max_outputs
        output_cols = [output_columns[i] for i in range(0, len(output_columns), step)][:max_outputs]
    else:
        output_cols = output_columns
    
    # Calculate correlation matrix
    all_cols = list(input_columns) + list(output_cols)
    correlation_matrix = df[all_cols].corr()
    
    # Extract input-output correlation submatrix
    input_output_corr = correlation_matrix.loc[input_columns, output_cols]
    
    # Create output index labels
    if sample_output_indices is not None:
        output_labels = [f'{output_cols[0].split(",")[0] if "," in output_cols[0] else output_cols[0].split("_")[0]}_{i}' 
                        for i in sample_output_indices[:len(output_cols)]]
    else:
        output_labels = output_cols
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        input_output_corr.values,
        aspect='auto',
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation='nearest'
    )
    
    # Set ticks and labels
    ax.set_xticks(range(len(output_cols)))
    ax.set_yticks(range(len(input_columns)))
    
    # Sample labels for readability
    n_label_samples = min(10, len(output_cols))
    step = max(1, len(output_cols) // n_label_samples)
    ax.set_xticklabels([output_cols[i] if i % step == 0 else '' for i in range(len(output_cols))],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(input_columns, fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation (r)', rotation=270, labelpad=20, fontsize=10)
    
    # Labels
    ax.set_xlabel('Output Index', fontsize=11)
    ax.set_ylabel('Input Variables', fontsize=11)
    ax.set_title('Input-Output Correlations', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Correlation heatmap saved to: {save_path}")
    
    return fig, ax, input_output_corr


def plot_strongest_correlation(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    standardize: bool = True,
) -> Tuple[object, object, Dict[str, Union[str, float]]]:
    """
    Plot scatter plot of strongest input-output correlation relationship.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing input and output columns.
    input_columns : list
        List of input column names.
    output_columns : list
        List of output column names.
    figsize : tuple, default=(8, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figures.
    standardize : bool, default=True
        Whether to standardize (z-score) values before plotting.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    correlation_info : dict
        Dictionary with strongest correlation info: {'input': str, 'output': str, 'r': float}
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    try:
        from scipy.stats import pearsonr
    except ImportError:
        # Fallback to numpy correlation if scipy not available
        def pearsonr(x, y):
            r = np.corrcoef(x, y)[0, 1]
            return r, 0.0  # p-value not needed
    
    # Calculate all correlations
    correlations = []
    
    for input_col in input_columns:
        input_vals = df[input_col].dropna().values
        if len(input_vals) == 0:
            continue
        
        for output_col in output_columns:
            output_vals = df[output_col].dropna().values
            if len(output_vals) == 0:
                continue
            
            # Align lengths
            min_len = min(len(input_vals), len(output_vals))
            input_vals_aligned = input_vals[:min_len]
            output_vals_aligned = output_vals[:min_len]
            
            # Remove any remaining NaN
            valid_mask = np.isfinite(input_vals_aligned) & np.isfinite(output_vals_aligned)
            if np.sum(valid_mask) < 2:
                continue
            
            input_vals_valid = input_vals_aligned[valid_mask]
            output_vals_valid = output_vals_aligned[valid_mask]
            
            # Calculate correlation
            r, _ = pearsonr(input_vals_valid, output_vals_valid)
            if not np.isnan(r):
                correlations.append({
                    'input': input_col,
                    'output': output_col,
                    'r': float(r),
                    'input_vals': input_vals_valid,
                    'output_vals': output_vals_valid
                })
    
    if not correlations:
        raise ValueError("No valid correlations found")
    
    # Find strongest correlation
    strongest = max(correlations, key=lambda x: abs(x['r']))
    
    # Standardize if requested
    if standardize:
        input_vals_plot = (strongest['input_vals'] - np.mean(strongest['input_vals'])) / np.std(strongest['input_vals'])
        output_vals_plot = (strongest['output_vals'] - np.mean(strongest['output_vals'])) / np.std(strongest['output_vals'])
    else:
        input_vals_plot = strongest['input_vals']
        output_vals_plot = strongest['output_vals']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(
        input_vals_plot, output_vals_plot,
        alpha=0.6,
        color='purple',
        s=20,
        edgecolors='none'
    )
    
    # Regression line
    z = np.polyfit(input_vals_plot, output_vals_plot, 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.min(input_vals_plot), np.max(input_vals_plot), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8, label='Regression')
    
    # Labels
    input_label = f"{strongest['input']}"
    output_label = f"{strongest['output']}"
    if standardize:
        input_label += " (standardized)"
        output_label += " (standardized)"
    
    ax.set_xlabel(input_label, fontsize=11)
    ax.set_ylabel(output_label, fontsize=11)
    
    # Title with correlation
    ax.set_title(f'Strongest Input-Output Relationship', fontsize=12, fontweight='bold')
    
    # Add correlation text
    ax.text(0.05, 0.95, f'$r = {strongest["r"]:.3f}$',
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Strongest correlation plot saved to: {save_path}")
    
    correlation_info = {
        'input': strongest['input'],
        'output': strongest['output'],
        'r': strongest['r']
    }
    
    return fig, ax, correlation_info


def plot_data_analysis_suite(
    dataset,
    input_columns: Optional[Sequence[str]] = None,
    output_columns: Optional[Sequence[str]] = None,
    output_groups: Optional[Dict[str, Sequence[str]]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    max_outputs: int = 10,
) -> Dict[str, Union[object, Dict[str, float], pd.DataFrame]]:
    """
    Create a complete suite of data analysis plots.
    
    Generates all analysis plots in a single call and saves them to a directory.
    
    Parameters
    ----------
    dataset : SurrogateDataset or pd.DataFrame
        Dataset object or DataFrame containing the data.
    input_columns : list, optional
        List of input column names. If None, uses dataset.input_columns.
    output_columns : list, optional
        List of output column names. If None, uses dataset.output_columns.
    output_groups : dict, optional
        Dictionary mapping group names to output column lists.
    save_dir : str or Path, optional
        Directory to save plots. If None, plots are displayed only.
    dpi : int, default=150
        Resolution for saved figures.
    max_outputs : int, default=10
        Maximum number of outputs to plot.
        
    Returns
    -------
    results : dict
        Dictionary containing all figure objects and computed metrics.
    """
    from surge.dataset import SurrogateDataset
    
    # Extract DataFrame
    if isinstance(dataset, SurrogateDataset):
        df = dataset.df
        if input_columns is None:
            input_columns = dataset.input_columns
        if output_columns is None:
            output_columns = dataset.output_columns
    elif isinstance(dataset, pd.DataFrame):
        df = dataset
        if input_columns is None or output_columns is None:
            raise ValueError("input_columns and output_columns must be provided when using DataFrame")
    else:
        raise TypeError("dataset must be SurrogateDataset or pd.DataFrame")
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Input distributions
    fig1, ax1 = plot_input_distributions(
        df, input_columns=input_columns,
        save_path=save_dir / 'input_distributions.png' if save_dir else None,
        dpi=dpi
    )
    results['input_distributions'] = (fig1, ax1)
    
    # 2. Output distributions
    if output_columns:
        sample_indices = None
        if len(output_columns) > max_outputs:
            step = len(output_columns) // max_outputs
            sample_indices = list(range(0, len(output_columns), step))[:max_outputs]
        
        fig2, ax2 = plot_output_distributions(
            df, output_columns=output_columns, sample_indices=sample_indices,
            save_path=save_dir / 'output_distributions.png' if save_dir else None,
            dpi=dpi, max_outputs=max_outputs
        )
        results['output_distributions'] = (fig2, ax2)
    
    # 3. Profile mean ± std
    if output_columns:
        fig3, ax3 = plot_profile_mean_std(
            df, output_columns=output_columns,
            save_path=save_dir / 'profile_mean_std.png' if save_dir else None,
            dpi=dpi
        )
        results['profile_mean_std'] = (fig3, ax3)
    
    # 4. Signal-to-noise
    sample_output_indices = None
    if output_columns and len(output_columns) > max_outputs:
        step = len(output_columns) // max_outputs
        sample_output_indices = list(range(0, len(output_columns), step))[:max_outputs]
    
    fig4, ax4, snr_dict = plot_signal_to_noise(
        df, input_columns=input_columns, output_columns=output_columns,
        sample_output_indices=sample_output_indices,
        save_path=save_dir / 'signal_to_noise.png' if save_dir else None,
        dpi=dpi, max_outputs=max_outputs
    )
    results['signal_to_noise'] = (fig4, ax4, snr_dict)
    
    # 5. Correlation heatmap
    if output_columns:
        fig5, ax5, corr_matrix = plot_correlation_heatmap(
            df, input_columns=input_columns, output_columns=output_columns,
            sample_output_indices=sample_output_indices,
            save_path=save_dir / 'correlation_heatmap.png' if save_dir else None,
            dpi=dpi, max_outputs=max_outputs
        )
        results['correlation_heatmap'] = (fig5, ax5, corr_matrix)
    
    # 6. Strongest correlation
    if output_columns:
        fig6, ax6, corr_info = plot_strongest_correlation(
            df, input_columns=input_columns, output_columns=output_columns,
            save_path=save_dir / 'strongest_correlation.png' if save_dir else None,
            dpi=dpi
        )
        results['strongest_correlation'] = (fig6, ax6, corr_info)
    
    return results

