"""Generic run visualization: inference comparison from any SURGE run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .comparison import plot_inference_comparison_grid

# Default model display names (short labels for plots)
DEFAULT_MODEL_DISPLAY = {
    "random_forest_profiles": "Random Forest",
    "torch_mlp_mc_dropout": "MLP",
    "gpflow_gpr_profiles": "GPR",
    "xgc_mlp_target": "MLP",
    "xgc_rf_target": "Random Forest",
}


def load_predictions(
    run_dir: Path,
    datasets: Tuple[str, ...] = ("train", "test"),
) -> Dict[str, Dict[str, Dict[int, Tuple[Any, Any]]]]:
    """Load predictions from a SURGE workflow run. Supports multi-output."""
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return {}

    results: Dict[str, Dict[str, Dict[int, Tuple[Any, Any]]]] = {}
    for pred_file in sorted(predictions_dir.glob("*.csv")):
        if "_uq" in pred_file.stem:
            continue
        stem = pred_file.stem
        dataset_type = (
            "train"
            if "_train" in pred_file.name
            else ("val" if "_val" in pred_file.name else "test")
        )
        if dataset_type not in datasets:
            continue
        model_name = stem.replace("_train", "").replace("_val", "").replace("_test", "")

        df = pd.read_csv(pred_file)
        y_true_cols = sorted([c for c in df.columns if c.startswith("y_true_")])
        y_pred_cols = sorted([c for c in df.columns if c.startswith("y_pred_")])
        if not y_true_cols or not y_pred_cols:
            continue

        if model_name not in results:
            results[model_name] = {}
        if dataset_type not in results[model_name]:
            results[model_name][dataset_type] = {}

        for i, (tc, pc) in enumerate(zip(y_true_cols, y_pred_cols)):
            results[model_name][dataset_type][i] = (df[tc].values, df[pc].values)

    return results


def viz_run(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    output_indices: Optional[List[int]] = None,
    model_display: Optional[Dict[str, str]] = None,
    output_display: Optional[Dict[str, str]] = None,
    xlabel_prefix: str = "Ground Truth",
    ylabel_prefix: str = "Prediction",
    axis_lim: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Generate inference comparison plots for a SURGE run directory.

    Works for any run (MHD-code, kinetic-code, etc.) by auto-detecting outputs from
    prediction CSV columns (y_true_00, y_pred_00, ...).

    Returns
    -------
    dict
        R² results and paths to saved plots.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(run_dir, datasets=("train", "test"))
    if not predictions:
        raise FileNotFoundError(
            f"No predictions found in {run_dir / 'predictions'}. "
            "Run a SURGE workflow first."
        )

    models = list(predictions.keys())
    n_outputs = 0
    for m in models:
        for ds in ("train", "test"):
            if ds in predictions[m]:
                n_outputs = max(n_outputs, len(predictions[m][ds]))
                break

    if output_indices is None:
        output_indices = list(range(n_outputs))

    model_display = model_display or {}
    for k, v in DEFAULT_MODEL_DISPLAY.items():
        if k not in model_display:
            model_display[k] = v

    output_names = [f"output_{i}" for i in output_indices]
    output_display = output_display or {f"output_{i}": f"output_{i}" for i in output_indices}

    all_r2: Dict[str, Any] = {}
    saved_paths: List[Path] = []

    for out_idx in output_indices:
        out_name = f"output_{out_idx}"
        results_dict = {out_name: {}}
        for model_name in models:
            results_dict[out_name][model_name] = {}
            for ds in ("train", "test"):
                if ds in predictions[model_name] and out_idx in predictions[model_name][ds]:
                    results_dict[out_name][model_name][ds] = predictions[model_name][ds][
                        out_idx
                    ]

        if not results_dict[out_name]:
            continue

        save_path = output_dir / f"inference_comparison_{out_name}.png"
        fig, axes, r2_results = plot_inference_comparison_grid(
            results_dict,
            units={out_name: "a.u."},
            save_path=save_path,
            dpi=dpi,
            xlabel_prefix=xlabel_prefix,
            ylabel_prefix=ylabel_prefix,
            ylabel_include_model=False,
            output_display_names=output_display,
            model_display_names=model_display,
            title_include_model_dataset=True,
            layout="models_rows",
            axis_lim=axis_lim,
        )
        all_r2[out_name] = r2_results
        saved_paths.append(save_path)

    # Combined grid if multiple outputs
    if len(output_indices) > 1:
        combined_dict = {}
        for out_idx in output_indices:
            out_name = f"output_{out_idx}"
            combined_dict[out_name] = {}
            for model_name in models:
                if model_name in predictions:
                    combined_dict[out_name][model_name] = {}
                    for ds in ("train", "test"):
                        if (
                            ds in predictions[model_name]
                            and out_idx in predictions[model_name][ds]
                        ):
                            combined_dict[out_name][model_name][ds] = predictions[
                                model_name
                            ][ds][out_idx]

        if len(combined_dict) > 1:
            save_path = output_dir / "inference_comparison_grid.png"
            fig, axes, r2_results = plot_inference_comparison_grid(
                combined_dict,
                units={k: "a.u." for k in combined_dict},
                save_path=save_path,
                dpi=dpi,
                xlabel_prefix=xlabel_prefix,
                ylabel_prefix=ylabel_prefix,
                ylabel_include_model=False,
                output_display_names=output_display,
                model_display_names=model_display,
                title_include_model_dataset=True,
                layout="outputs_rows",
                axis_lim=axis_lim,
            )
            all_r2["combined"] = r2_results
            saved_paths.append(save_path)
    elif len(saved_paths) == 1:
        # Single output: also save as inference_comparison_grid.png
        import shutil

        grid_path = output_dir / "inference_comparison_grid.png"
        shutil.copy(saved_paths[0], grid_path)
        saved_paths.append(grid_path)

    return {"r2": all_r2, "saved_paths": [str(p) for p in saved_paths]}
