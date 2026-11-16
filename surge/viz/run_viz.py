"""Generic run visualization: inference comparison, HPO, and SHAP from any SURGE run directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ..io.load_compat import load_model_compat
import pandas as pd

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .comparison import plot_inference_comparison_grid
from .hpo import plot_hpo_convergence

LOG = logging.getLogger(__name__)

# Default model display names (short labels for plots)
DEFAULT_MODEL_DISPLAY = {
    "random_forest_profiles": "Random Forest",
    "torch_mlp_mc_dropout": "MLP",
    "gpflow_gpr_profiles": "GPR",
    "mlp_aparallel": "MLP",
    "rf_aparallel": "Random Forest",
}


def load_predictions(
    run_dir: Path,
    datasets: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Dict[str, Dict[int, Tuple[Any, Any]]]]:
    """Load predictions from a SURGE workflow run. Supports multi-output.

    Accepts both ``*.parquet`` (current workflow default) and ``*.csv``
    (legacy runs). When both exist for the same model/split, parquet
    wins because it round-trips dtypes losslessly.
    """
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return {}

    # Parquet first so it overwrites csv entries for the same model/split.
    pred_files = sorted(
        list(predictions_dir.glob("*.parquet")) + list(predictions_dir.glob("*.csv"))
    )

    results: Dict[str, Dict[str, Dict[int, Tuple[Any, Any]]]] = {}
    for pred_file in pred_files:
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

        if pred_file.suffix == ".parquet":
            df = pd.read_parquet(pred_file)
        else:
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


def _model_short_name(name: str) -> str:
    """Convert model filename to short display name."""
    return DEFAULT_MODEL_DISPLAY.get(name, name.replace("_", " ").title())


def viz_hpo(
    run_dir: Path,
    output_dir: Path,
    dpi: int = 150,
    plot_metric: Optional[str] = None,
) -> List[str]:
    """
    Generate HPO convergence plots from run_dir/hpo/*.json.
    Returns list of saved plot paths.
    """
    hpo_dir = run_dir / "hpo"
    if not hpo_dir.exists():
        return []

    hpo_files = sorted(hpo_dir.glob("*_hpo.json"))
    if not hpo_files:
        return []

    # Build method names from filenames (mlp_aparallel_hpo -> MLP)
    hpo_dict = {}
    for f in hpo_files:
        model_key = f.stem.replace("_hpo", "")
        hpo_dict[_model_short_name(model_key)] = str(f)

    # Auto-detect: prefer R² when val_r2 is in trials (or use plot_metric override)
    save_path = output_dir / "hpo_convergence.png"
    try:
        plot_hpo_convergence(
            hpo_files=list(hpo_dict.values()),
            metric=plot_metric,  # None = auto-detect: R² if available, else RMSE
            method_names=list(hpo_dict.keys()),
            save_path=save_path,
            dpi=dpi,
        )
        return [str(save_path)]
    except Exception:
        return []


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
    include_hpo: bool = True,
    hpo_plot_metric: Optional[str] = None,
    include_shap: bool = False,
    shap_split: str = "val",
    shap_max_samples: int = 500,
    shap_output_index: Optional[int] = None,
    include_datastreamset_eval: bool = False,
    datastreamset_size: int = 50000,
    datastreamset_max: int = 10,
    datastreamset_eval_set: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate inference comparison plots for a SURGE run directory.

    Works for any run (any dataset format) by auto-detecting outputs from
    prediction CSV columns (y_true_00, y_pred_00, ...).

    Parameters
    ----------
    include_shap : bool, default=False
        If True, run SHAP analysis for tree models and save to plots/shap_{model}/.
    shap_split : str, default="val"
        Split to use for SHAP (train, val, or test).
    shap_max_samples : int, default=500
        Max samples for SHAP (smaller = faster).
    include_datastreamset_eval : bool, default=False
        If True, evaluate models on held-out dataset datastreamsets (stacked_npy format only).
    datastreamset_size : int, default=50000
        Rows per datastreamset for datastreamset evaluation.
    datastreamset_max : int, default=10
        Max datastreamsets to evaluate.
    datastreamset_eval_set : str, optional
        Override set_name for datastreamset eval (e.g. set2_beta0p5 for cross-set eval).

    Returns
    -------
    dict
        R² results and paths to saved plots.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(run_dir, datasets=("train", "val", "test"))
    if not predictions:
        raise FileNotFoundError(
            f"No predictions found in {run_dir / 'predictions'}. "
            "Run a SURGE workflow first."
        )

    models = list(predictions.keys())
    n_outputs = 0
    for m in models:
        for ds in ("train", "val", "test"):
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
            for ds in ("train", "val", "test"):
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
                    for ds in ("train", "val", "test"):
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

    # HPO convergence plots
    if include_hpo:
        hpo_paths = viz_hpo(run_dir, output_dir, dpi=dpi, plot_metric=hpo_plot_metric)
        saved_paths.extend(hpo_paths)

    # SHAP analysis
    if include_shap:
        shap_paths = viz_shap(
            run_dir,
            output_dir,
            split=shap_split,
            max_samples=shap_max_samples,
            output_index=shap_output_index,
            dpi=dpi,
        )
        saved_paths.extend(shap_paths)

    # Segment-based evaluation (stacked_npy format only)
    datastreamset_result: Optional[Dict[str, Any]] = None
    if include_datastreamset_eval:
        try:
            datastreamset_result = viz_datastreamset_evaluation(
                run_dir,
                output_dir,
                datastreamset_size=datastreamset_size,
                max_datastreamsets=datastreamset_max,
                eval_set_name=datastreamset_eval_set,
            )
        except Exception as e:
            LOG.warning("Segment evaluation failed: %s", e)

    out: Dict[str, Any] = {"r2": all_r2, "saved_paths": [str(p) for p in saved_paths]}
    if datastreamset_result:
        out["datastreamset_eval"] = datastreamset_result
    return out


def viz_shap(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    models: Optional[List[str]] = None,
    split: str = "val",
    max_samples: int = 500,
    max_background: int = 100,
    plot_type: str = "both",
    kernel_nsamples: int = 50,
    save_dependence: bool = True,
    save_waterfall: bool = True,
    save_force: bool = False,
    dpi: int = 150,
    tree_only: bool = False,
    output_index: Optional[int] = None,
) -> List[str]:
    """
    Run SHAP analysis for models in a SURGE run directory.

    Loads dataset, scalers, and models from the run; uses prediction indices
    to get the correct X for the chosen split. Saves SHAP plots to
    output_dir/shap_{model_name}/.

    Parameters
    ----------
    run_dir : Path
        SURGE workflow run directory (contains spec.yaml, models/, scalers/, etc.).
    output_dir : Path, optional
        Where to save plots. Default: run_dir/plots.
    models : list of str, optional
        Model names to run SHAP for. If None, uses all models in the run.
    split : str, default="val"
        Which split to explain: "train", "val", or "test".
    max_samples : int, default=500
        Max samples to explain (for speed).
    max_background : int, default=100
        Background dataset size for explainers.
    plot_type : str, default="both"
        Summary: "bar", "dot", or "both".
    kernel_nsamples : int, default=50
        nsamples for KernelExplainer (when tree_only=False).
    save_dependence : bool, default=True
        Save dependence plots for top features.
    save_waterfall : bool, default=True
        Save waterfall plot.
    save_force : bool, default=False
        Save force plot.
    dpi : int, default=150
        Plot resolution.
    tree_only : bool, default=False
        If True, only run SHAP for tree models (skip KernelExplainer for non-tree).
    output_index : int, optional
        For multi-output models, which output to explain (0-based). For the A_parallel target,
        output_1 = A_parallel. If None and dataset_format is stacked_npy with 2 outputs, uses 1.

    Returns
    -------
    list of str
        Paths to saved SHAP plots.
    """
    from ..dataset import SurrogateDataset
    from ..workflow.spec import SurrogateWorkflowSpec
    from .importance import SHAP_AVAILABLE, _is_tree_model, run_shap_analysis

    if not SHAP_AVAILABLE:
        LOG.warning("SHAP not installed. Skip viz_shap. Install with: pip install shap")
        return []

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_file = run_dir / "spec.yaml"
    summary_file = run_dir / "workflow_summary.json"
    if not spec_file.exists():
        LOG.warning("No spec.yaml in %s. Skip viz_shap.", run_dir)
        return []
    if not summary_file.exists():
        LOG.warning("No workflow_summary.json in %s. Skip viz_shap.", run_dir)
        return []

    if not YAML_AVAILABLE:
        LOG.warning("PyYAML required for viz_shap.")
        return []

    with spec_file.open("r", encoding="utf-8") as f:
        spec_dict = yaml.safe_load(f)
    spec = SurrogateWorkflowSpec.from_dict(spec_dict)

    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # Infer output_index for the A_parallel target: output_1 is A_parallel
    shap_output_index = output_index
    if shap_output_index is None:
        fmt = (spec.dataset_format or "").lower()
        n_out = summary.get("dataset", {}).get("n_outputs", 0)
        if fmt == "stacked_npy" and n_out >= 2:
            shap_output_index = 1

    dataset_path = Path(summary.get("dataset", {}).get("file_path", spec.dataset_path))
    if not dataset_path.exists():
        LOG.warning("Dataset path %s not found. Skip viz_shap.", dataset_path)
        return []

    # Load dataset
    dataset = SurrogateDataset.from_path(
        dataset_path,
        format=spec.dataset_format,
        metadata_path=spec.metadata_path,
        sample=spec.sample_rows,
        analyzer_kwargs={"hints": spec.metadata_overrides, **spec.analyzer},
    )
    if not dataset.input_columns or dataset.df is None:
        LOG.warning("Dataset has no input columns. Skip viz_shap.")
        return []

    # Load input scaler
    scalers_dir = run_dir / "scalers"
    input_scaler = None
    if spec.standardize_inputs and (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")

    model_entries = summary.get("models", [])
    if not model_entries:
        LOG.warning("No models in workflow summary. Skip viz_shap.")
        return []

    model_names = models or [m["name"] for m in model_entries]
    saved_paths: List[str] = []

    for model_entry in model_entries:
        name = model_entry.get("name")
        if name not in model_names:
            continue
        model_path = Path(model_entry.get("artifacts", {}).get("model", ""))
        if not model_path or not model_path.exists():
            if not model_path.is_absolute():
                model_path = run_dir / "models" / Path(str(model_path)).name
        if not model_path.exists():
            LOG.warning("Model %s not found at %s. Skip.", name, model_path)
            continue

        pred_files = model_entry.get("artifacts", {}).get("predictions", {})
        pred_path = pred_files.get(split)
        if not pred_path:
            LOG.warning("No %s predictions for %s. Skip SHAP.", split, name)
            continue
        pred_path = Path(pred_path)
        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        if not pred_path.exists():
            for ext in (".csv", ".parquet"):
                alt = run_dir / "predictions" / f"{name}_{split}{ext}"
                if alt.exists():
                    pred_path = alt
                    break
        if not pred_path.exists():
            LOG.warning("Predictions for %s %s not found. Skip SHAP.", name, split)
            continue

        # Load indices from predictions
        if pred_path.suffix == ".parquet":
            pred_df = pd.read_parquet(pred_path)
        else:
            pred_df = pd.read_csv(pred_path)
        if "index" not in pred_df.columns:
            LOG.warning("Predictions %s has no index column. Skip SHAP.", pred_path)
            continue
        indices = pred_df["index"].values.astype(int)
        indices = indices[:max_samples]

        # Get X from dataset (indices are row positions from the original df)
        X_raw = dataset.df[dataset.input_columns].iloc[indices].values.astype(np.float64)
        if input_scaler is not None:
            X = input_scaler.transform(X_raw)
        else:
            X = X_raw

        X_background = X[: min(max_background, len(X))]
        X = X[:max_samples]

        # Load model (with compat for sklearn 1.3.x and PyTorch pickle protocol)
        try:
            adapter = load_model_compat(model_path, model_entry)
        except Exception as e:
            LOG.warning("Failed to load model %s: %s. Skip SHAP.", name, e)
            continue

        shap_dir = output_dir / f"shap_{name}"
        shap_dir.mkdir(parents=True, exist_ok=True)

        use_tree = _is_tree_model(adapter)
        use_kernel = not tree_only or use_tree

        try:
            results = run_shap_analysis(
                adapter,
                X,
                X_background=X_background,
                output_index=shap_output_index,
                feature_names=dataset.input_columns,
                data_source=dataset,
                use_tree=use_tree,
                use_kernel=use_kernel,
                kernel_nsamples=kernel_nsamples,
                save_dir=shap_dir,
                dpi=dpi,
                plot_type=plot_type,
                save_dependence=save_dependence,
                save_waterfall=save_waterfall,
                save_force=save_force,
            )
            saved_paths.extend(results.get("saved_paths", []))
            if "tree_error" in results:
                LOG.warning("SHAP tree for %s: %s", name, results["tree_error"])
            if "kernel_error" in results:
                LOG.warning("SHAP kernel for %s: %s", name, results["kernel_error"])
        except Exception as e:
            LOG.warning("SHAP failed for %s: %s", name, e)

    return saved_paths


def _load_or_build_train_ranges(
    run_dir: Path,
    spec: Any,
    dataset_path: Path,
    summary: Dict[str, Any],
    input_scaler: Optional[Any],
    output_scaler: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Load train_data_ranges.json or build from training predictions if missing."""
    ranges_file = run_dir / "train_data_ranges.json"
    if ranges_file.exists():
        with ranges_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: build from training predictions (for runs before this feature)
    pred_files = list((run_dir / "predictions").glob("*_train.csv"))
    if not pred_files:
        LOG.warning("No train_data_ranges.json and no train predictions. Skip in-range check.")
        return None

    pred_df = pd.read_csv(pred_files[0])
    if "index" not in pred_df.columns:
        return None
    indices = pred_df["index"].values.astype(int)

    try:
        # Use same sample as workflow so indices match
        sample = getattr(spec, "sample_rows", None)
        hints = dict(spec.metadata_overrides or {})
        hints["random_state"] = getattr(spec, "seed", 42)
        dataset = SurrogateDataset.from_path(
            dataset_path,
            format=spec.dataset_format,
            metadata_path=spec.metadata_path,
            sample=sample,
            sample_random_state=getattr(spec, "seed", 42),
            analyzer_kwargs={"hints": hints, **(spec.analyzer or {})},
        )
    except Exception as e:
        LOG.warning("Could not load dataset for train ranges: %s", e)
        return None

    if dataset.df is None or len(dataset.df) == 0:
        return None

    # Use spec's input subset (n_inputs or input_columns) to match model/scaler
    spec_input_cols = getattr(spec, "input_columns", None)
    spec_n_inputs = getattr(spec, "n_inputs", None)
    if spec_input_cols and len(spec_input_cols) > 0:
        eff_input_cols = [c for c in spec_input_cols if c in dataset.df.columns]
    elif spec_n_inputs and dataset.input_columns:
        eff_input_cols = dataset.input_columns[: spec_n_inputs]
    else:
        eff_input_cols = dataset.input_columns

    # Get rows at train indices (indices are row positions in the sampled df)
    n_rows = len(dataset.df)
    valid = (indices >= 0) & (indices < n_rows)
    if not np.any(valid):
        return None
    idx = indices[valid]
    X = dataset.df[eff_input_cols].iloc[idx].values.astype(np.float64)
    y = dataset.df[dataset.output_columns].iloc[idx].values.astype(np.float64)
    if input_scaler is not None:
        X = input_scaler.transform(X)
    if output_scaler is not None:
        y = output_scaler.transform(y)

    ranges = {
        "inputs": {
            "columns": eff_input_cols,
            "min": X.min(axis=0).tolist(),
            "max": X.max(axis=0).tolist(),
        },
        "outputs": {
            "columns": dataset.output_columns,
            "min": y.min(axis=0).tolist(),
            "max": y.max(axis=0).tolist(),
        },
    }
    # Save for future use
    with (run_dir / "train_data_ranges.json").open("w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=2)
    LOG.info("Built and saved train_data_ranges.json from training predictions")
    return ranges


# Drift detection policy (thresholds below; document in workflow outputs)
DRIFT_R2_DROP_THRESHOLD = 0.1
DRIFT_RMSE_RATIO_THRESHOLD = 1.5


def _apply_drift_detection(
    results: Dict[str, Any],
    run_dir: Path,
    output_key: str = "output_1",
    r2_drop_threshold: float = DRIFT_R2_DROP_THRESHOLD,
    rmse_ratio_threshold: float = DRIFT_RMSE_RATIO_THRESHOLD,
) -> None:
    """
    Apply drift detection policy: set drift_detected and ood_triggers per datastreamset,
    and overall drift_warning. Drives continual learning recommendations.
    """
    ref_r2 = None
    ref_rmse = None
    if (run_dir / "workflow_summary.json").exists():
        try:
            with (run_dir / "workflow_summary.json").open() as f:
                wf = json.load(f)
            for m in wf.get("models", []):
                train = m.get("metrics", {}).get("train", {})
                ref_r2 = train.get("r2")
                ref_rmse = train.get("rmse")
                if ref_r2 is not None or ref_rmse is not None:
                    break
        except Exception:
            pass

    datastreamsets_with_drift = []
    for datastreamset_key, datastreamset in results.get("datastreamsets", {}).items():
        triggers = []
        in_range = datastreamset.get("in_range", {})

        # OOD input trigger
        if not in_range.get("inputs_in_range", True):
            triggers.append("input_ood")

        # OOD output trigger
        if not in_range.get("outputs_in_range", True):
            triggers.append("output_ood")

        # Accuracy drop trigger (use first model with metrics)
        for model_name, mdata in datastreamset.get("models", {}).items():
            out_data = mdata.get(output_key, mdata.get("output_0", {}))
            seg_r2 = out_data.get("r2")
            seg_rmse = out_data.get("rmse")
            if seg_r2 is not None and ref_r2 is not None:
                if seg_r2 < ref_r2 - r2_drop_threshold:
                    triggers.append("accuracy_drop")
                    break
            if seg_rmse is not None and ref_rmse is not None and ref_rmse > 0:
                if seg_rmse > ref_rmse * rmse_ratio_threshold:
                    triggers.append("accuracy_drop")
                    break

        drift_detected = len(triggers) > 0
        datastreamset["drift_detected"] = drift_detected
        datastreamset["ood_triggers"] = triggers
        if drift_detected:
            datastreamsets_with_drift.append(datastreamset_key)

    results["drift_detection"] = {
        "policy": "surge.viz.run_viz drift thresholds (DRIFT_R2_DROP_THRESHOLD, DRIFT_RMSE_RATIO_THRESHOLD)",
        "drift_warning": len(datastreamsets_with_drift) > 0,
        "datastreamsets_with_drift": datastreamsets_with_drift,
        "n_datastreamsets_evaluated": len(results.get("datastreamsets", {})),
        "continual_learning_recommendation": (
            "DRIFT DETECTED: Evaluation data is outside training regime or prediction "
            "accuracy has dropped. Recommend ingesting new data from these regions and "
            "retraining the surrogate."
            if datastreamsets_with_drift
            else None
        ),
    }


def _check_datastreamset_in_range(
    X: np.ndarray,
    y: np.ndarray,
    train_ranges: Dict[str, Any],
    input_columns: List[str],
    output_columns: List[str],
) -> Dict[str, Any]:
    """Check if datastreamset min/max fall within training data ranges."""
    in_min = np.array(train_ranges["inputs"]["min"], dtype=np.float64)
    in_max = np.array(train_ranges["inputs"]["max"], dtype=np.float64)
    out_min = np.array(train_ranges["outputs"]["min"], dtype=np.float64)
    out_max = np.array(train_ranges["outputs"]["max"], dtype=np.float64)

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(axis=0), y.max(axis=0)

    # Out of range: datastreamset extends beyond training bounds
    inputs_low = X_min < in_min
    inputs_high = X_max > in_max
    outputs_low = y_min < out_min
    outputs_high = y_max > out_max

    n_in = int(np.sum(inputs_low | inputs_high))
    n_out = int(np.sum(outputs_low | outputs_high))
    inputs_in_range = n_in == 0
    outputs_in_range = n_out == 0

    out_cols = []
    for i in range(len(output_columns)):
        if outputs_low[i] or outputs_high[i]:
            out_cols.append(output_columns[i])
    in_cols = []
    for i in range(len(input_columns)):
        if inputs_low[i] or inputs_high[i]:
            in_cols.append(input_columns[i])

    # Per-OOD-input min/max for plotting range evolution across datastreamsets
    input_min_max_ood = {}
    for col in in_cols:
        i = input_columns.index(col)
        input_min_max_ood[col] = {
            "min": float(X_min[i]),
            "max": float(X_max[i]),
            "train_min": float(in_min[i]),
            "train_max": float(in_max[i]),
        }

    return {
        "inputs_in_range": inputs_in_range,
        "outputs_in_range": outputs_in_range,
        "n_inputs_out_of_range": n_in,
        "n_outputs_out_of_range": n_out,
        "inputs_out_of_range": in_cols[:20],  # cap for readability
        "outputs_out_of_range": out_cols,
        "input_min_max_ood": input_min_max_ood,
    }


def viz_datastreamset_evaluation(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    datastreamset_size: int = 50000,
    max_datastreamsets: int = 10,
    datastreamset_offset: int = 0,
    eval_set_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained models on held-out datastreamsets of the dataset.

    Loads contiguous row ranges (datastreamsets) from the full dataset and computes
    R², RMSE, MAE per datastreamset. Use to test generalization to unseen data.

    Only supported for the stacked_npy format (uses row_range). The workflow's train/val/test
    come from a random sample; these datastreamsets are different contiguous regions.

    Parameters
    ----------
    run_dir : Path
        SURGE workflow run directory.
    output_dir : Path, optional
        Where to save datastreamset_stats.json. Default: run_dir/plots.
    datastreamset_size : int, default=50000
        Rows per datastreamset.
    max_datastreamsets : int, default=10
        Max number of datastreamsets to evaluate (datastreamset 0 = rows 0:datastreamset_size, etc.).
    datastreamset_offset : int, default=0
        Start from this datastreamset index (datastreamset_offset * datastreamset_size).
    eval_set_name : str, optional
        Override set_name when loading datastreamsets (e.g. set2_beta0p5 for cross-set eval).
        If None, uses spec.metadata_overrides.set_name (train set).

    Returns
    -------
    dict
        Per-model, per-datastreamset R², RMSE, MAE; and path to saved JSON.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from ..dataset import SurrogateDataset

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_file = run_dir / "spec.yaml"
    summary_file = run_dir / "workflow_summary.json"
    if not spec_file.exists() or not summary_file.exists():
        raise FileNotFoundError(
            f"Run dir must have spec.yaml and workflow_summary.json: {run_dir}"
        )

    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for viz_datastreamset_evaluation.")

    with spec_file.open("r", encoding="utf-8") as f:
        spec_dict = yaml.safe_load(f)
    from ..workflow.spec import SurrogateWorkflowSpec

    spec = SurrogateWorkflowSpec.from_dict(spec_dict)
    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    dataset_path = Path(summary.get("dataset", {}).get("file_path", spec.dataset_path))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    fmt = (spec.dataset_format or "").lower()
    if fmt != "stacked_npy":
        LOG.warning("viz_datastreamset_evaluation only supports stacked_npy format. Skip.")
        return {}

    scalers_dir = run_dir / "scalers"
    input_scaler = None
    output_scaler = None
    if spec.standardize_inputs and (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")
    if spec.standardize_outputs and (scalers_dir / "outputs.joblib").exists():
        output_scaler = joblib.load(scalers_dir / "outputs.joblib")

    model_entries = summary.get("models", [])
    if not model_entries:
        raise ValueError("No models in workflow summary.")

    # Load training data ranges for in-distribution checks
    train_ranges = _load_or_build_train_ranges(
        run_dir, spec, dataset_path, summary, input_scaler, output_scaler
    )
    train_set_name = (spec.metadata_overrides or {}).get("set_name", "set1")
    effective_eval_set = eval_set_name if eval_set_name is not None else train_set_name
    if train_ranges:
        results: Dict[str, Any] = {
            "datastreamset_size": datastreamset_size,
            "train_ranges_path": str(run_dir / "train_data_ranges.json"),
            "train_set_name": train_set_name,
            "eval_set_name": effective_eval_set,
            "datastreamsets": {},
        }
    else:
        results = {
            "datastreamset_size": datastreamset_size,
            "train_set_name": train_set_name,
            "eval_set_name": effective_eval_set,
            "datastreamsets": {},
        }

    for datastreamset_idx in range(datastreamset_offset, datastreamset_offset + max_datastreamsets):
        start = datastreamset_idx * datastreamset_size
        end = start + datastreamset_size
        datastreamset_key = f"datastreamset_{datastreamset_idx}_rows_{start}_{end}"

        try:
            hints = dict(spec.metadata_overrides or {})
            hints["set_name"] = effective_eval_set
            hints["row_range"] = (start, end)
            dataset = SurrogateDataset.from_path(
                dataset_path,
                format=spec.dataset_format,
                metadata_path=spec.metadata_path,
                sample=None,
                analyzer_kwargs={"hints": hints, **(spec.analyzer or {})},
            )
        except Exception as e:
            LOG.warning("Failed to load %s: %s", datastreamset_key, e)
            continue

        if dataset.df is None or len(dataset.df) == 0:
            LOG.warning("%s empty. Skip.", datastreamset_key)
            continue

        # Use spec's input subset (n_inputs or input_columns) to match model/scaler
        spec_input_cols = getattr(spec, "input_columns", None)
        spec_n_inputs = getattr(spec, "n_inputs", None)
        if spec_input_cols and len(spec_input_cols) > 0:
            eff_input_cols = [c for c in spec_input_cols if c in dataset.df.columns]
        elif spec_n_inputs and dataset.input_columns:
            eff_input_cols = dataset.input_columns[: spec_n_inputs]
        else:
            eff_input_cols = dataset.input_columns
        X = dataset.df[eff_input_cols].values.astype(np.float64)
        y_true = dataset.df[dataset.output_columns].values.astype(np.float64)
        if input_scaler is not None:
            X = input_scaler.transform(X)
        if output_scaler is not None:
            y_true = output_scaler.transform(y_true)

        datastreamset_entry: Dict[str, Any] = {"n_samples": len(dataset.df), "models": {}}

        # In-distribution check: are datastreamset min/max within training ranges?
        if train_ranges:
            in_range = _check_datastreamset_in_range(
                X, y_true, train_ranges, eff_input_cols, dataset.output_columns
            )
            datastreamset_entry["in_range"] = in_range

        results["datastreamsets"][datastreamset_key] = datastreamset_entry

        for model_entry in model_entries:
            name = model_entry.get("name")
            model_path = Path(model_entry.get("artifacts", {}).get("model", ""))
            if not model_path or not model_path.exists():
                model_path = run_dir / "models" / Path(str(model_path)).name
            if not model_path.exists():
                continue

            try:
                adapter = load_model_compat(model_path, model_entry)
                y_pred = adapter.predict(X)
                if hasattr(y_pred, "numpy"):
                    y_pred = y_pred.numpy()
                y_pred = np.asarray(y_pred)
            except Exception as e:
                LOG.warning("Model %s failed on %s: %s", name, datastreamset_key, e)
                continue

            n_out = y_true.shape[1]
            metrics = {}
            for i in range(n_out):
                r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
                rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
                mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
                resid = y_pred[:, i] - y_true[:, i]
                mean_err = float(np.mean(resid))
                std_err = float(np.std(resid))
                metrics[f"output_{i}"] = {
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "mean_error": mean_err,
                    "std_error": std_err,
                }
            results["datastreamsets"][datastreamset_key]["models"][name] = metrics

    # Apply drift detection policy (OOD + accuracy triggers)
    _apply_drift_detection(results, run_dir, output_key="output_1")

    out_path = output_dir / "datastreamset_evaluation_stats.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    results["saved_path"] = str(out_path)

    # Plot R², RMSE, and prediction error vs datastreamset index
    _plot_datastreamset_evaluation(results, output_dir, datastreamset_size, run_dir=run_dir)
    return results


def _plot_datastreamset_evaluation(
    results: Dict[str, Any],
    output_dir: Path,
    datastreamset_size: int,
    dpi: int = 150,
    run_dir: Optional[Path] = None,
) -> None:
    """Plot R², RMSE, prediction error vs datastreamset index; shade out-of-range datastreamsets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    datastreamsets_data = results.get("datastreamsets", {})
    if not datastreamsets_data:
        return

    # Parse datastreamset indices and row ranges (datastreamset_0_rows_0_50000, datastreamset_1_rows_50000_100000, ...)
    def _datastreamset_sort_key(k: str) -> int:
        parts = k.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    sorted_keys = sorted(datastreamsets_data.keys(), key=_datastreamset_sort_key)
    datastreamset_indices = []
    for key in sorted_keys:
        parts = key.split("_")
        try:
            datastreamset_indices.append(int(parts[1]))
        except (IndexError, ValueError):
            datastreamset_indices.append(len(datastreamset_indices))

    models = set()
    for s in datastreamsets_data.values():
        models.update(s.get("models", {}).keys())
    models = sorted(models)
    model_display = {m: DEFAULT_MODEL_DISPLAY.get(m, m.replace("_", " ").title()) for m in models}

    # Collect R² and RMSE per model, per datastreamset, for output_1 (A_parallel)
    output_key = "output_1"  # A_parallel
    for out_k in ("output_1", "output_0"):
        if any(
            out_k in datastreamsets_data[s].get("models", {}).get(m, {})
            for s in datastreamsets_data
            for m in models
        ):
            output_key = out_k
            break

    # Check if we have in_range data and input_min_max_ood for range evolution
    has_in_range = any("in_range" in s for s in datastreamsets_data.values())
    has_input_range_evolution = has_in_range and any(
        datastreamsets_data[s].get("in_range", {}).get("input_min_max_ood")
        for s in datastreamsets_data
    )
    n_rows = 4 if has_input_range_evolution else (3 if has_in_range else 2)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    if n_rows == 2:
        ax_r2, ax_rmse = axes[0], axes[1]
        ax_inrange = None
        ax_input_range = None
    elif n_rows == 3:
        ax_r2, ax_rmse, ax_inrange = axes[0], axes[1], axes[2]
        ax_input_range = None
    else:
        ax_r2, ax_rmse, ax_inrange, ax_input_range = axes[0], axes[1], axes[2], axes[3]

    # Reference R² from train/val (workflow summary)
    ref_r2_train = None
    ref_r2_val = None
    if run_dir and (run_dir / "workflow_summary.json").exists():
        try:
            with (run_dir / "workflow_summary.json").open() as f:
                wf = json.load(f)
            for m in wf.get("models", []):
                train_r2 = m.get("metrics", {}).get("train", {}).get("r2")
                val_r2 = m.get("metrics", {}).get("val", {}).get("r2")
                if train_r2 is not None:
                    ref_r2_train = float(train_r2) if ref_r2_train is None else ref_r2_train
                if val_r2 is not None:
                    ref_r2_val = float(val_r2) if ref_r2_val is None else ref_r2_val
                break
        except Exception:
            pass

    # Out-of-range mask for shading
    oor_mask = []
    if has_in_range:
        for key in sorted_keys:
            ir = datastreamsets_data[key].get("in_range", {})
            oor_mask.append(not ir.get("inputs_in_range", True) or not ir.get("outputs_in_range", True))

    has_model_metrics = bool(models)
    for model in models:
        r2_vals = []
        rmse_vals = []
        mae_vals = []
        for key in sorted_keys:
            mdata = datastreamsets_data[key].get("models", {}).get(model, {})
            out_data = mdata.get(output_key, mdata.get("output_0", {}))
            r2_vals.append(out_data.get("r2", np.nan))
            rmse_vals.append(out_data.get("rmse", np.nan))
            mae_vals.append(out_data.get("mae", np.nan))
        if not r2_vals:
            continue
        label = model_display.get(model, model)
        ax_r2.plot(datastreamset_indices, r2_vals, "o-", label=label, linewidth=2, markersize=8)
        ax_rmse.plot(datastreamset_indices, rmse_vals, "o-", label=label, linewidth=2, markersize=8)

    # Shade out-of-range datastreamsets on R² and RMSE
    if has_in_range and oor_mask:
        for i, is_oor in enumerate(oor_mask):
            if is_oor:
                ax_r2.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")
                ax_rmse.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")

    ax_r2.set_ylabel("R² score")
    ax_r2.set_title("Generalization: R² vs held-out datastreamset (A_parallel, red = out-of-range)")
    if ref_r2_train is not None:
        ax_r2.axhline(ref_r2_train, color="gray", linestyle="--", alpha=0.7, label=f"Train R²={ref_r2_train:.3f}")
    if ref_r2_val is not None:
        ax_r2.axhline(ref_r2_val, color="gray", linestyle=":", alpha=0.7, label=f"Val R²={ref_r2_val:.3f}")
    if has_model_metrics:
        ax_r2.legend(loc="best", fontsize=9)
    else:
        ax_r2.text(0.5, 0.5, "No model metrics (retrain in current env)", ha="center", va="center", transform=ax_r2.transAxes)
    ax_r2.grid(True, alpha=0.3)
    # Zoom in on R² around train/val range for better visibility
    if has_model_metrics:
        r2_all = []
        for key in sorted_keys:
            for m in models:
                mdata = datastreamsets_data[key].get("models", {}).get(m, {})
                out_data = mdata.get(output_key, mdata.get("output_0", {}))
                v = out_data.get("r2")
                if v is not None:
                    r2_all.append(v)
        refs = [r for r in (ref_r2_train, ref_r2_val) if r is not None]
        if r2_all or refs:
            y_min = min(r2_all + refs) - 0.02 if (r2_all or refs) else 0
            y_max = max(r2_all + refs) + 0.01 if (r2_all or refs) else 1
            ax_r2.set_ylim(max(0, y_min), min(1.02, y_max))
    else:
        ax_r2.set_ylim(bottom=0)

    ax_rmse.set_xlabel("Segment index" if not has_in_range and ax_input_range is None else "")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Prediction error (RMSE) vs held-out datastreamset (A_parallel)")
    if has_model_metrics:
        ax_rmse.legend(loc="best", fontsize=9)
    else:
        ax_rmse.text(0.5, 0.5, "No model metrics (retrain in current env)", ha="center", va="center", transform=ax_rmse.transAxes)
    ax_rmse.grid(True, alpha=0.3)

    if has_in_range and ax_inrange is not None:
        in_range_inputs = []
        in_range_outputs = []
        for key in sorted_keys:
            ir = datastreamsets_data[key].get("in_range", {})
            in_range_inputs.append(ir.get("inputs_in_range", True))
            in_range_outputs.append(ir.get("outputs_in_range", True))
        x_pos = np.arange(len(datastreamset_indices))
        width = 0.35
        ax_inrange.bar(x_pos - width / 2, [1 if x else 0 for x in in_range_inputs], width, label="Inputs in range", color="steelblue", alpha=0.8)
        ax_inrange.bar(x_pos + width / 2, [1 if x else 0 for x in in_range_outputs], width, label="Outputs in range", color="coral", alpha=0.8)
        ax_inrange.set_ylabel("In range (1=yes)")
        ax_inrange.set_title("In-distribution: within training min/max?")
        ax_inrange.set_yticks([0, 1])
        ax_inrange.set_yticklabels(["No", "Yes"])
        ax_inrange.set_xticks(x_pos)
        ax_inrange.set_xticklabels([str(i) for i in datastreamset_indices])
        ax_inrange.legend(loc="best", fontsize=9)
        ax_inrange.grid(True, alpha=0.3, axis="y")
        if ax_input_range is None:
            ax_inrange.set_xlabel("Segment index")

    # Input range evolution: how min/max of OOD inputs change across datastreamsets
    if ax_input_range is not None:
        from collections import Counter
        all_ood = []
        for key in sorted_keys:
            ir = datastreamsets_data[key].get("in_range", {})
            mm = ir.get("input_min_max_ood", {})
            all_ood.extend(mm.keys())
        if all_ood:
            top_ood = [c[0] for c in Counter(all_ood).most_common(6)]
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_ood)))
            ax_input_range.axhspan(0, 1, alpha=0.15, color="green", label="Training range (norm)")
            for idx, col in enumerate(top_ood):
                mins_norm, maxs_norm = [], []
                for key in sorted_keys:
                    ir = datastreamsets_data[key].get("in_range", {})
                    mm = ir.get("input_min_max_ood", {}).get(col)
                    if mm:
                        tmin, tmax = mm["train_min"], mm["train_max"]
                        span = tmax - tmin if tmax != tmin else 1.0
                        mins_norm.append((mm["min"] - tmin) / span)
                        maxs_norm.append((mm["max"] - tmin) / span)
                    else:
                        mins_norm.append(np.nan)
                        maxs_norm.append(np.nan)
                c = colors[idx]
                ax_input_range.plot(datastreamset_indices, mins_norm, "o--", color=c, markersize=5, alpha=0.9)
                ax_input_range.plot(datastreamset_indices, maxs_norm, "s-", color=c, markersize=5, alpha=0.9, label=col)
            ax_input_range.set_ylabel("Normalized value (0–1 = train range)")
            ax_input_range.set_xlabel("Datastreamset index")
            ax_input_range.set_title("Input range evolution: OOD inputs min/max vs datastreamset (top 6)")
            ax_input_range.set_ylim(-0.1, 1.5)
            ax_input_range.legend(loc="upper right", fontsize=7, ncol=2)
            ax_input_range.grid(True, alpha=0.3)
        else:
            ax_input_range.set_visible(False)

    plt.tight_layout()
    save_path = output_dir / "datastreamset_evaluation_r2_rmse.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plot_paths = [str(save_path)]

    # Second plot: R² degradation (train R² - datastreamset R²) and MAE vs datastreamset
    if has_model_metrics and ref_r2_train is not None:
        fig2, (ax_degrad, ax_mae) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for model in models:
            r2_vals = []
            mae_vals = []
            for key in sorted_keys:
                mdata = datastreamsets_data[key].get("models", {}).get(model, {})
                out_data = mdata.get(output_key, mdata.get("output_0", {}))
                r2_vals.append(out_data.get("r2", np.nan))
                mae_vals.append(out_data.get("mae", np.nan))
            if not r2_vals:
                continue
            degrad = np.array([ref_r2_train - r for r in r2_vals])
            label = model_display.get(model, model)
            ax_degrad.plot(datastreamset_indices, degrad, "o-", label=label, linewidth=2, markersize=8)
            ax_mae.plot(datastreamset_indices, mae_vals, "o-", label=label, linewidth=2, markersize=8)
        if has_in_range and oor_mask:
            for i, is_oor in enumerate(oor_mask):
                if is_oor:
                    ax_degrad.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")
                    ax_mae.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")
        ax_degrad.set_ylabel("R² degradation (train − datastreamset)")
        ax_degrad.set_title("R² drop vs held-out datastreamset (higher = worse generalization)")
        ax_degrad.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax_degrad.legend(loc="best", fontsize=9)
        ax_degrad.grid(True, alpha=0.3)
        ax_mae.set_ylabel("MAE (|pred − true|)")
        ax_mae.set_xlabel("Segment index")
        ax_mae.set_title("Mean absolute error vs held-out datastreamset")
        ax_mae.legend(loc="best", fontsize=9)
        ax_mae.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path2 = output_dir / "datastreamset_evaluation_degradation.png"
        fig2.savefig(save_path2, dpi=dpi, bbox_inches="tight")
        plot_paths.append(str(save_path2))
    results["saved_plot_path"] = plot_paths[0]
    results["saved_plot_paths"] = plot_paths
