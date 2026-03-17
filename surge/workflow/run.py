"""Workflow runner that stitches together dataset loading, training, and artifacts."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..engine import EngineRunConfig, ModelRunResult, ModelSpec, SurrogateEngine
from ..dataset import SurrogateDataset
from ..hpc import detect_compute_resources
from ..io import (
    init_artifact_paths,
    save_environment_snapshot,
    save_git_revision,
    save_hpo_results,
    save_metrics,
    save_model,
    save_predictions,
    save_scaler,
    save_spec,
    save_workflow_summary,
)
from ..io.artifacts import ArtifactPaths
from ..registry import registry_summary
from .spec import HPOConfig, ModelConfig, SurrogateWorkflowSpec

LOGGER = logging.getLogger(__name__)


def _progress_step(step: int, total: int, message: str, *, file: Any = None) -> None:
    """Print a workflow step line so the user sees progress (e.g. [2/8] Preparing splits...)."""
    file = file or sys.stdout
    print(f"[{step}/{total}] {message}", flush=True, file=file)

try:  # pragma: no cover
    import optuna
    from optuna.samplers import TPESampler

    try:
        from optuna.integration import BoTorchSampler

        BOTORCH_AVAILABLE = True
    except Exception:  # pragma: no cover
        BoTorchSampler = None
        BOTORCH_AVAILABLE = False

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    optuna = None
    TPESampler = None
    BoTorchSampler = None
    BOTORCH_AVAILABLE = False
    OPTUNA_AVAILABLE = False


def run_surrogate_workflow(spec: SurrogateWorkflowSpec) -> Dict[str, Any]:
    """Execute the full surrogate workflow described by the spec."""
    print("SURGE workflow started.", flush=True)

    # Total steps: 1 load dataset, 1 prepare, then per model: 1 (HPO if enabled) + 1 (train)
    n_model_steps = sum(
        2 if (c.hpo and c.hpo.enabled and c.hpo.search_space) else 1 for c in spec.models
    )
    total_steps = 2 + n_model_steps
    step = 0

    def next_step(msg: str) -> None:
        nonlocal step
        step += 1
        _progress_step(step, total_steps, msg)

    LOGGER.info("Starting SURGE workflow for dataset %s", spec.dataset_path)
    next_step("Loading dataset...")
    dataset = SurrogateDataset.from_path(
        spec.dataset_path,
        format=spec.dataset_format,
        metadata_path=spec.metadata_path,
        sample=spec.sample_rows,
        analyzer_kwargs={"hints": spec.metadata_overrides, **spec.analyzer},
    )

    next_step("Preparing splits and scalers...")
    engine = SurrogateEngine(
        run_config=EngineRunConfig(
            test_fraction=spec.test_fraction,
            val_fraction=spec.val_fraction,
            standardize_inputs=spec.standardize_inputs,
            standardize_outputs=spec.standardize_outputs,
            random_state=spec.seed,
        )
    )
    engine.configure_dataframe(
        dataset.df,
        dataset.input_columns,
        dataset.output_columns,
    )
    engine.prepare()
    raw_splits = engine.get_raw_splits()
    split_info = {
        "train": len(raw_splits.X_train),
        "val": len(raw_splits.X_val),
        "test": len(raw_splits.X_test) if raw_splits.X_test is not None else 0,
    }

    run_tag = spec.run_tag or _default_run_tag(dataset.file_path)
    paths = init_artifact_paths(spec.output_dir, run_tag, exist_ok=spec.overwrite_existing_run)
    save_spec(spec.to_dict(), paths)

    resources = detect_compute_resources()
    save_environment_snapshot(paths, extras=resources.to_dict())
    save_git_revision(paths, repo_dir=".")

    # Persist scalers if available
    scalers = engine.scalers
    scaler_artifacts: Dict[str, Optional[str]] = {}
    if scalers.input_scaler is not None:
        scaler_artifacts["input_scaler"] = str(save_scaler(scalers.input_scaler, "inputs", paths))
    if scalers.output_scaler is not None:
        scaler_artifacts["output_scaler"] = str(save_scaler(scalers.output_scaler, "outputs", paths))

    workflow_metrics: Dict[str, Any] = {}
    workflow_models: List[Dict[str, Any]] = []

    for model_cfg in spec.models:
        if model_cfg.key not in engine.registry:
            LOGGER.warning("Skipping model '%s' because it is not registered.", model_cfg.key)
            continue
        model_spec = _model_spec_from_config(model_cfg)
        model_name = model_spec.name or model_spec.key
        hpo_artifact = None
        if model_cfg.hpo and model_cfg.hpo.enabled and model_cfg.hpo.search_space:
            next_step(f"HPO for {model_name} ({model_cfg.hpo.n_trials} trials)...")
            model_spec, hpo_summary = _run_hpo(engine, model_spec, model_cfg.hpo, run_tag, paths)
            hpo_artifact = str(save_hpo_results(hpo_summary, paths, filename=f"{model_spec.name}_hpo.json"))

        next_step(f"Training {model_name}...")
        result = engine.train_model(model_spec, record=True)
        model_entry = _persist_model_artifacts(
            result,
            spec.predictions_scope,
            spec.predictions_format,
            paths,
        )

        workflow_metrics[model_entry["name"]] = {
            "train": result.train_metrics,
            "val": result.val_metrics,
            "test": result.test_metrics,
            "timings": result.timings,
        }
        if hpo_artifact:
            model_entry["artifacts"]["hpo"] = hpo_artifact
        workflow_models.append(model_entry)

    metrics_path = save_metrics(workflow_metrics, paths)
    summary = {
        "run_tag": run_tag,
        "dataset": dataset.summary(),
        "resources": resources.to_dict(),
        "splits": split_info,
        "scalers": scaler_artifacts,
        "models": workflow_models,
        "registry": registry_summary(),
        "artifacts": {
            "root": str(paths.root),
            "metrics": str(metrics_path),
            "summary": str(paths.summary_file),
            "spec": str(paths.spec_file),
        },
    }
    save_workflow_summary(summary, paths)
    _progress_step(total_steps, total_steps, f"Done. Artifacts in {paths.root}")
    LOGGER.info("Workflow %s complete. Artifacts saved to %s", run_tag, paths.root)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_run_tag(file_path: Optional[Path]) -> str:
    prefix = file_path.stem if file_path else "dataset"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}"


def _model_spec_from_config(config: ModelConfig) -> ModelSpec:
    return ModelSpec(
        key=config.key,
        name=config.name,
        params=dict(config.params),
        request_uncertainty=config.request_uncertainty,
        store_predictions=config.store_predictions,
        tags=tuple(config.tags),
    )


def _persist_model_artifacts(
    result: ModelRunResult,
    prediction_splits: Sequence[str],
    predictions_format: str,
    paths: ArtifactPaths,
) -> Dict[str, Any]:
    model_name = result.spec.name or result.spec.key
    model_path = save_model(result.adapter, model_name, paths)
    prediction_files: Dict[str, str] = {}
    for split in prediction_splits:
        payload = result.predictions.get(split)
        if payload is None:
            continue
        prediction_files[split] = str(
            save_predictions(payload, model_name, split, paths, format=predictions_format)
        )

    if "val_uq" in result.predictions:
        uq_payload = result.predictions["val_uq"]
        uq_path = paths.predictions_dir / f"{model_name}_val_uq.json"
        with uq_path.open("w", encoding="utf-8") as handle:
            json.dump(_ensure_serializable(uq_payload), handle, indent=2)
        prediction_files["val_uq"] = str(uq_path)

    return {
        "name": model_name,
        "backend": result.adapter.backend,
        "params": result.spec.params,
        "metrics": {
            "train": result.train_metrics,
            "val": result.val_metrics,
            "test": result.test_metrics,
        },
        "timings": result.timings,
        "artifacts": {
            "model": str(model_path),
            "predictions": prediction_files,
        },
    }


def _run_hpo(
    engine: SurrogateEngine,
    base_spec: ModelSpec,
    config: HPOConfig,
    run_tag: str,
    paths: ArtifactPaths,
) -> Tuple[ModelSpec, Dict[str, Any]]:
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna must be installed to enable HPO.")
    sampler = None
    if config.sampler == "botorch" and BOTORCH_AVAILABLE:
        sampler = BoTorchSampler()
    elif config.sampler == "tpe" or sampler is None:
        sampler = TPESampler(seed=engine.config.random_state)

    import time as _time_module
    import traceback as _traceback_module

    def objective(trial: "optuna.Trial") -> float:
        print(f"  [HPO] Trial {trial.number} started...", flush=True)
        t0 = _time_module.perf_counter()
        try:
            sampled_params = _sample_hpo_params(trial, config.search_space)
            trial_spec = replace(
                base_spec,
                params={**base_spec.params, **sampled_params},
                store_predictions=False,
            )
            result = engine.train_model(trial_spec, record=False)
            metric_value = _extract_metric(result, config.metric)
            elapsed = _time_module.perf_counter() - t0
            print(f"  [HPO] Trial {trial.number} finished in {elapsed:.1f}s ({config.metric}={metric_value:.4f})", flush=True)
            return metric_value
        except Exception as e:
            elapsed = _time_module.perf_counter() - t0
            print(f"  [HPO] Trial {trial.number} FAILED after {elapsed:.1f}s: {e!r}", flush=True)
            _traceback_module.print_exc()
            raise

    study = optuna.create_study(
        direction=config.direction,
        sampler=sampler,
        study_name=f"{run_tag}-{base_spec.key}",
    )

    # Progress: tqdm bar only when stdout is a TTY (e.g. real terminal). Otherwise print each trial
    # so progress is visible when run via conda run / nohup / CI where there is no TTY.
    callbacks: List[Callable[["optuna.Study", "optuna.FrozenTrial"], None]] = []
    tqdm_pbar = None
    use_tqdm = sys.stdout.isatty()
    if use_tqdm:
        try:
            from tqdm import tqdm
            tqdm_pbar = tqdm(
                total=config.n_trials,
                desc=f"  HPO {base_spec.name or base_spec.key}",
                unit="trial",
                file=sys.stdout,
            )
            def _tqdm_cb(study: "optuna.Study", trial: "optuna.FrozenTrial") -> None:
                tqdm_pbar.update(1)
                if study.best_trial is not None and study.best_trial.value is not None:
                    tqdm_pbar.set_postfix(best=f"{study.best_trial.value:.4f}")
            callbacks.append(_tqdm_cb)
        except ImportError:
            use_tqdm = False
    if not use_tqdm:
        def _print_cb(study: "optuna.Study", trial: "optuna.FrozenTrial") -> None:
            n = len(study.trials)
            best = study.best_trial.value if study.best_trial is not None and study.best_trial.value is not None else None
            best_str = f"{best:.4f}" if best is not None else "?"
            print(f"    Trial {n}/{config.n_trials} (best {config.metric}={best_str})", flush=True)
        callbacks.append(_print_cb)

    # Suppress Optuna categorical serialization warning during HPO so tqdm bar is not flooded
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*categorical distribution should be a tuple of None, bool, int, float and str.*",
            module="optuna.distributions",
        )
        try:
            study.optimize(
                objective,
                n_trials=config.n_trials,
                timeout=config.timeout,
                callbacks=callbacks,
                show_progress_bar=False,  # we use our own tqdm or print callback
            )
        finally:
            if tqdm_pbar is not None:
                tqdm_pbar.close()

    # Resolve _index params (categoricals with list/tuple choices) back to actual values
    best_params = dict(base_spec.params)
    for k, v in study.best_trial.params.items():
        if k.endswith("_index") and k[:-6] in config.search_space:
            cat_spec = config.search_space[k[:-6]]
            if cat_spec.get("type") == "categorical":
                choices = cat_spec["choices"]
                normalized = [
                    tuple(c) if isinstance(c, list) else c for c in choices
                ]
                val = normalized[v]
                best_params[k[:-6]] = list(val) if isinstance(val, tuple) else val
                continue
        best_params[k] = v
    updated_spec = replace(base_spec, params=best_params)

    hpo_summary = {
        "metric": config.metric,
        "direction": config.direction,
        "best_trial": {
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "number": study.best_trial.number,
        },
        "n_trials": len(study.trials),
        "trials": [
            {"number": trial.number, "value": trial.value, "params": trial.params}
            for trial in study.trials
        ],
    }
    topk = sorted(
        (
            trial
            for trial in study.trials
            if trial.value is not None
        ),
        key=lambda t: t.value,
    )[: min(5, len(study.trials))]
    hpo_summary["top_trials"] = [
        {"number": trial.number, "value": trial.value, "params": trial.params}
        for trial in topk
    ]
    return updated_spec, hpo_summary


def _sample_hpo_params(trial: "optuna.Trial", space: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        spec_type = spec.get("type")
        if spec_type == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=spec.get("step", 1)
            )
        elif spec_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=spec.get("log", False),
            )
        elif spec_type == "loguniform":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=True,
            )
        elif spec_type == "categorical":
            choices = spec["choices"]
            normalized = [
                tuple(c) if isinstance(c, list) else c for c in choices
            ]
            # Use an int index for non-serializable choices (e.g. lists/tuples) so Optuna
            # doesn't warn and the study remains serializable; we resolve the index when
            # building best_params in _run_hpo.
            simple = all(
                isinstance(c, (type(None), bool, int, float, str)) for c in normalized
            )
            if simple:
                sampled = trial.suggest_categorical(name, normalized)
            else:
                idx = trial.suggest_int(f"{name}_index", 0, len(normalized) - 1)
                sampled = normalized[idx]
            if isinstance(sampled, tuple):
                sampled = list(sampled)
            params[name] = sampled
        elif spec_type == "variable_list":
            # For variable-length lists (e.g., hidden_layers per layer)
            # First sample the number of layers
            num_layers_name = spec.get("num_layers_name", f"{name}_num_layers")
            num_layers = trial.suggest_int(
                num_layers_name,
                int(spec["num_layers"]["low"]),
                int(spec["num_layers"]["high"])
            )
            # Then sample each element independently
            element_spec = spec["element"]
            element_type = element_spec.get("type")
            layer_list = []
            for i in range(num_layers):
                element_name = f"{name}_{i}"
                if element_type == "int":
                    layer_list.append(trial.suggest_int(
                        element_name,
                        int(element_spec["low"]),
                        int(element_spec["high"]),
                        step=element_spec.get("step", 1)
                    ))
                elif element_type == "logint":
                    # Log-uniform integer sampling
                    log_low = np.log(float(element_spec["low"]))
                    log_high = np.log(float(element_spec["high"]))
                    sampled = np.exp(trial.suggest_float(
                        f"{element_name}_log",
                        log_low,
                        log_high,
                        log=False
                    ))
                    step = element_spec.get("step", 1)
                    if step > 1:
                        sampled = int(np.round(sampled / step) * step)
                    else:
                        sampled = int(np.round(sampled))
                    layer_list.append(max(int(element_spec["low"]), 
                                        min(int(element_spec["high"]), sampled)))
                else:
                    raise ValueError(f"Unsupported element type for variable_list: {element_type}")
            params[name] = layer_list
        else:
            raise ValueError(f"Unsupported HPO parameter type: {spec_type}")
    return params


def _extract_metric(result: ModelRunResult, metric_key: str) -> float:
    split, metric = metric_key.split("_", 1)
    split_metrics = {
        "train": result.train_metrics,
        "val": result.val_metrics,
        "test": result.test_metrics or {},
    }.get(split)
    if split_metrics is None or metric not in split_metrics:
        raise KeyError(f"Metric '{metric_key}' not available.")
    return float(split_metrics[metric])


def _ensure_serializable(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result

