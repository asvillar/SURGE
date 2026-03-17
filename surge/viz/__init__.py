"""Visualization helpers for SURGE model results and data analysis.

This module provides publication-quality visualization functions for:
- Model comparison and inference analysis
- Profile comparisons with detailed insets
- HPO convergence analysis
- Comprehensive data analysis (distributions, SNR, correlations)
"""

from .analysis import (
    plot_correlation_heatmap,
    plot_data_analysis_suite,
    plot_input_distributions,
    plot_output_distributions,
    plot_profile_mean_std,
    plot_signal_to_noise,
    plot_strongest_correlation,
)
from .comparison import plot_inference_comparison_grid, plot_mse_comparison
from .run_viz import load_predictions, viz_run
from .hpo import plot_hpo_comparison, plot_hpo_convergence
from .profiles import (
    compute_profile_metrics,
    plot_density_scatter,
    plot_profile_band,
    plot_profile_comparison_with_inset,
)

__all__ = [
    # Run viz
    "load_predictions",
    "viz_run",
    # Profile plots
    "compute_profile_metrics",
    "plot_profile_band",
    "plot_density_scatter",
    "plot_profile_comparison_with_inset",
    # Comparison plots
    "plot_inference_comparison_grid",
    "plot_mse_comparison",
    # HPO plots
    "plot_hpo_convergence",
    "plot_hpo_comparison",
    # Data analysis plots
    "plot_input_distributions",
    "plot_output_distributions",
    "plot_profile_mean_std",
    "plot_signal_to_noise",
    "plot_correlation_heatmap",
    "plot_strongest_correlation",
    "plot_data_analysis_suite",
]


