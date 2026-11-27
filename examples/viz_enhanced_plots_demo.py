#!/usr/bin/env python3
"""Demonstration of enhanced visualization plots in SURGE.

This script demonstrates the new publication-quality plotting capabilities:
- Inference comparison grids
- MSE comparison plots
- Profile comparisons with insets
- HPO convergence plots
- Comprehensive data analysis suite

Usage:
    python examples/viz_enhanced_plots_demo.py --run-dir <run folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from surge.dataset import SurrogateDataset
from surge.viz import (
    plot_correlation_heatmap,
    plot_data_analysis_suite,
    plot_hpo_convergence,
    plot_inference_comparison_grid,
    plot_input_distributions,
    plot_mse_comparison,
    plot_output_distributions,
    plot_profile_comparison_with_inset,
    plot_profile_mean_std,
    plot_signal_to_noise,
    plot_strongest_correlation,
)


def load_workflow_results(run_dir: Path):
    """Load results from a SURGE workflow run."""
    results = {}
    
    # Load workflow summary
    summary_file = run_dir / 'workflow_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load metrics
    metrics_file = run_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load predictions
    predictions_dir = run_dir / 'predictions'
    if predictions_dir.exists():
        results['predictions'] = {}
        for pred_file in predictions_dir.glob('*.csv'):
            model_name = pred_file.stem.replace('_train', '').replace('_val', '').replace('_test', '')
            dataset_type = 'train' if '_train' in pred_file.name else ('val' if '_val' in pred_file.name else 'test')
            
            df = pd.read_csv(pred_file)
            if 'y_true_00' in df.columns and 'y_pred_00' in df.columns:
                if model_name not in results['predictions']:
                    results['predictions'][model_name] = {}
                results['predictions'][model_name][dataset_type] = (
                    df['y_true_00'].values,
                    df['y_pred_00'].values
                )
    
    return results


def demo_inference_comparison_grid(run_dir: Path, output_dir: Path):
    """Demonstrate inference comparison grid plot."""
    print("Creating inference comparison grid plot...")
    
    results = load_workflow_results(run_dir)
    
    if 'predictions' not in results:
        print("⚠️  No predictions found in workflow results")
        return
    
    # Build results dictionary
    # For demo, assume single output and multiple models
    results_dict = {}
    
    # Get available models and datasets
    models = list(results['predictions'].keys())
    if not models:
        print("⚠️  No model predictions found")
        return
    
    # Assume single output for now
    output_name = "Output"  # Could extract from workflow summary
    
    results_dict[output_name] = {}
    for model_name in models:
        if model_name not in results['predictions']:
            continue
        results_dict[output_name][model_name] = {}
        for dataset_type in ['train', 'test']:
            if dataset_type in results['predictions'][model_name]:
                results_dict[output_name][model_name][dataset_type] = \
                    results['predictions'][model_name][dataset_type]
    
    if not results_dict[output_name]:
        print("⚠️  Insufficient data for comparison grid")
        return
    
    # Create plot
    fig, axes, r2_results = plot_inference_comparison_grid(
        results_dict,
        units={'Output': '[a.u.]'},
        save_path=output_dir / 'inference_comparison_grid.png',
        dpi=150
    )
    
    print(f"✅ Created inference comparison grid with R² results:")
    print(json.dumps(r2_results, indent=2))


def demo_hpo_convergence(run_dir: Path, output_dir: Path):
    """Demonstrate HPO convergence plot."""
    print("Creating HPO convergence plot...")
    
    hpo_dir = run_dir / 'hpo'
    if not hpo_dir.exists():
        print("⚠️  No HPO directory found")
        return
    
    hpo_files = list(hpo_dir.glob('*_hpo.json'))
    if not hpo_files:
        print("⚠️  No HPO files found")
        return
    
    # Plot convergence for first model
    hpo_file = hpo_files[0]
    
    fig, ax, best_values = plot_hpo_convergence(
        hpo_files=[hpo_file],
        metric='rmse',  # HPO files use val_rmse
        method_names=[hpo_file.stem.replace('_hpo', '').replace('_', ' ').title()],
        save_path=output_dir / 'hpo_convergence.png',
        dpi=150
    )
    
    print(f"✅ Created HPO convergence plot")
    print(f"   Best values: {best_values}")


def demo_data_analysis(dataset: SurrogateDataset, output_dir: Path):
    """Demonstrate data analysis suite."""
    print("Creating data analysis plots...")
    
    results = plot_data_analysis_suite(
        dataset,
        save_dir=output_dir / 'analysis',
        dpi=150,
        max_outputs=10
    )
    
    print(f"✅ Created data analysis suite with {len(results)} plots")


def main():
    parser = argparse.ArgumentParser(description='Enhanced visualization plots demo')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Folder written by `surge run` (workflow run directory)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: run_dir/plots)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset file for analysis plots')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SURGE Enhanced Visualization Plots Demo")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Demo 1: Inference comparison grid
    try:
        demo_inference_comparison_grid(run_dir, output_dir)
    except Exception as e:
        print(f"⚠️  Error creating inference comparison: {e}")
    
    print()
    
    # Demo 2: HPO convergence
    try:
        demo_hpo_convergence(run_dir, output_dir)
    except Exception as e:
        print(f"⚠️  Error creating HPO convergence: {e}")
    
    print()
    
    # Demo 3: Data analysis (if dataset provided)
    if args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            try:
                dataset = SurrogateDataset.from_path(dataset_path)
                demo_data_analysis(dataset, output_dir)
            except Exception as e:
                print(f"⚠️  Error creating data analysis plots: {e}")
        else:
            print(f"⚠️  Dataset file not found: {dataset_path}")
    
    print()
    print("=" * 80)
    print("✅ Demo completed!")
    print(f"📊 Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()





