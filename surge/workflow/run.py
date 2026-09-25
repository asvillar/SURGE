"""Workflow runner that stitches together dataset loading, training, and artifacts."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..engine import EngineRunConfig, ModelRunResult, ModelSpec, ScalerBundle, SurrogateEngine
from ..dataset import SurrogateDataset
from ..hpc import detect_compute_resources
from ..io import (
    init_artifact_paths,
    save_environment_snapshot,
    save_git_revision,
    save_hpo_results,
    save_metrics,
    save_model,
    save_model_card,
    save_predictions,
    save_scaler,
    save_spec,
    save_train_data_ranges,
    save_workflow_summary,
)
from ..io.artifacts import (
    ArtifactPaths,
    copy_invoked_config_source,
    save_run_invocation,
)
from ..io.run_logging import TeeText
from ..io.load_compat import load_model_compat
from ..model.registry import registry_summary
from ..utils import posix_str
from .spec import HPOConfig, ModelConfig, SurrogateWorkflowSpec

LOGGER = logging.getLogger(__name__)


def _safe_model_artifact_tag(name: str) -> str:
    """Filesystem-safe stem for training_progress / training_history artifacts."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())[:120]
    return slug or "model"


def _registry_backend(registry: Any, model_key: str) -> str:
    """Return the backend name declared by the registered adapter class."""
    if hasattr(registry, "get"):
        entry = registry.get(model_key)
        return str(getattr(getattr(entry, "adapter_cls", None), "backend", "") or "")
    return ""


def _hpo_architecture_from_trial_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize MLP depth/width from Optuna trial.params or merged sampled params."""
    hl = params.get("hidden_layers")
    if isinstance(hl, (list, tuple)) and hl:
        layers = [int(x) for x in hl]
        return {"num_hidden_layers": len(layers), "hidden_layers": layers}
    n = params.get("mlp_num_hidden_layers")
    if n is not None:
        ni = int(n)
        layers = [int(params[f"hidden_layers_{i}"]) for i in range(ni)]
        return {"num_hidden_layers": ni, "hidden_layers": layers}
    return {"num_hidden_layers": 0, "hidden_layers": []}


def _hpo_write_trial_training_history(
    paths: ArtifactPaths, trial_number: int, history: Any
) -> Optional[Path]:
    if not history:
        return None
    th_path = paths.root / f"hpo_trial_{trial_number:04d}_training_history.json"
    th_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    return th_path


def _hpo_refresh_best_progress_jsonl(paths: ArtifactPaths, trial_number: int) -> None:
    """Rewrite training_progress_hpo_best.jsonl from a trial's saved training_history JSON."""
    th_path = paths.root / f"hpo_trial_{trial_number:04d}_training_history.json"
    best_path = paths.root / "training_progress_hpo_best.jsonl"
    if not th_path.is_file():
        return
    try:
        data = json.loads(th_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, list):
        return
    with best_path.open("w", encoding="utf-8") as handle:
        for row in data:
            handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _progress_step(step: int, total: int, message: str, *, file: Any = None) -> None:
    """Print a workflow step line so the user sees progress (e.g. [2/8] Preparing splits...)."""
    file = file or sys.stdout
    print(f"[{step}/{total}] {message}", flush=True, file=file)


def _load_pretrained_scalers(run_dir: Path) -> ScalerBundle:
    """Load input/output scalers from a pretrained run directory."""
    import joblib

    scalers_dir = run_dir / "scalers"
    input_scaler = None
    output_scaler = None
    if (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")
    if (scalers_dir / "outputs.joblib").exists():
        output_scaler = joblib.load(scalers_dir / "outputs.joblib")
    return ScalerBundle(input_scaler=input_scaler, output_scaler=output_scaler)

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


def run_surrogate_workflow(
    spec: SurrogateWorkflowSpec,
    *,
    invocation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the full surrogate workflow described by the spec."""
    _real_out = sys.__stdout__
    _real_err = sys.__stderr__
    capture_buf = io.StringIO()
    sys.stdout = TeeText(_real_out, capture_buf)
    sys.stderr = TeeText(_real_err, capture_buf)
    log_fp = None
    try:
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
        if getattr(spec, "cv_folds", 0) and int(spec.cv_folds) > 0:
            warnings.warn(
                f"spec.cv_folds={spec.cv_folds} is not yet implemented in the unified "
                "workflow (k-fold training and aggregated fold metrics; see docs/ROADMAP.md). "
                "This run uses the usual single random train/val/test split; cv_folds is ignored.",
                UserWarning,
                stacklevel=2,
            )
            LOGGER.warning(
                "Ignoring spec.cv_folds=%s (cross-validation in workflow not implemented yet).",
                spec.cv_folds,
            )
        next_step("Loading dataset...")
        if spec.dataset_source == "m3dc1_batch":
            from ..datasets import M3DC1Dataset
            batch_dir = Path(spec.dataset_path)
            if not batch_dir.is_dir():
                raise FileNotFoundError(f"dataset_source=m3dc1_batch requires a directory: {batch_dir}")
            _batch_kw = {}
            if spec.batch_dir_filename:
                _batch_kw["filename"] = spec.batch_dir_filename
            dataset = M3DC1Dataset.from_batch_dir(
                batch_dir,
                mode_step=spec.batch_dir_mode_step,
                psi_step=spec.batch_dir_psi_step,
                target_shape=spec.batch_dir_target_shape,
                include_eigenmodes=spec.batch_dir_include_eigenmodes,
                verbose=True,
                **_batch_kw,
            )
        elif spec.dataset_source == "m3dc1_batch_per_mode":
            from ..datasets import M3DC1Dataset
            batch_dir = Path(spec.dataset_path)
            if not batch_dir.is_dir():
                raise FileNotFoundError(
                    f"dataset_source=m3dc1_batch_per_mode requires a directory: {batch_dir}"
                )
            _batch_kw = {}
            if spec.batch_dir_filename:
                _batch_kw["filename"] = spec.batch_dir_filename
            dataset = M3DC1Dataset.from_batch_dir_per_mode(
                batch_dir,
                verbose=True,
                **_batch_kw,
            )
            if spec.sample_rows:
                dataset.df = dataset.df.sample(n=min(spec.sample_rows, len(dataset.df)), random_state=spec.seed)
        else:
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
                resources=spec.resources,
            )
        )
        engine.configure_dataframe(
            dataset.df,
            dataset.input_columns,
            dataset.output_columns,
        )
        pretrained_dir: Optional[Path] = None
        pretrained_models: Dict[str, Any] = {}
        if spec.pretrained_run_dir:
            pretrained_dir = Path(spec.pretrained_run_dir).resolve()
            if not pretrained_dir.is_dir():
                raise FileNotFoundError(
                    f"pretrained_run_dir not found: {pretrained_dir}"
                )
            pretrained_scalers = _load_pretrained_scalers(pretrained_dir)
            engine.prepare(pretrained_scalers=pretrained_scalers)
            # Load workflow summary to map model names to pretrained paths
            summary_path = pretrained_dir / "workflow_summary.json"
            if summary_path.exists():
                with summary_path.open() as f:
                    pretrained_summary = json.load(f)
                for m in pretrained_summary.get("models", []):
                    name = m.get("name") or m.get("key", "")
                    if name:
                        model_path = pretrained_dir / "models" / f"{name}.joblib"
                        if model_path.exists():
                            try:
                                pretrained_models[name] = load_model_compat(
                                    model_path, m, for_finetune=True
                                )
                            except Exception as e:
                                LOGGER.warning(
                                    "Could not load pretrained model %s: %s",
                                    name,
                                    e,
                                )
        else:
            engine.prepare()
        raw_splits = engine.get_raw_splits()
        split_info = {
            "train": len(raw_splits.X_train),
            "val": len(raw_splits.X_val),
            "test": len(raw_splits.X_test) if raw_splits.X_test is not None else 0,
        }

        run_tag = spec.run_tag or _default_run_tag(dataset.file_path)
        paths = init_artifact_paths(spec.output_dir, run_tag, exist_ok=spec.overwrite_existing_run)
        log_path = paths.root / "run.log"
        log_fp = log_path.open("w", encoding="utf-8", buffering=1)
        log_fp.write(capture_buf.getvalue())
        capture_buf.close()
        sys.stdout = TeeText(_real_out, log_fp)
        sys.stderr = TeeText(_real_err, log_fp)
        if invocation:
            save_run_invocation(paths, invocation)
            spec_src = invocation.get("spec_path")
            if spec_src:
                copy_invoked_config_source(paths, Path(spec_src))
        save_spec(spec.to_dict(), paths)

        # Save training data min/max for in-distribution checks on new datastreamsets
        proc = engine.get_processed_splits()
        save_train_data_ranges(
            proc.X_train,
            proc.y_train,
            dataset.input_columns,
            dataset.output_columns,
            paths,
        )

        resources = detect_compute_resources()
        save_environment_snapshot(paths, extras=resources.to_dict())
        save_git_revision(paths, repo_dir=".")

        # Persist scalers if available
        scalers = engine.scalers
        scaler_artifacts: Dict[str, Optional[str]] = {}
        if scalers.input_scaler is not None:
            scaler_artifacts["input_scaler"] = posix_str(save_scaler(scalers.input_scaler, "inputs", paths))
        if scalers.output_scaler is not None:
            scaler_artifacts["output_scaler"] = posix_str(save_scaler(scalers.output_scaler, "outputs", paths))

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
                hpo_artifact = posix_str(save_hpo_results(hpo_summary, paths, filename=f"{model_spec.name}_hpo.json"))

            next_step(f"Training {model_name}...")
            pretrained_adapter = pretrained_models.get(model_name) if pretrained_dir else None
            tag = _safe_model_artifact_tag(model_name)
            try:
                _entry_backend = _registry_backend(engine.registry, model_spec.key)
            except KeyError:
                _entry_backend = ""
            if _entry_backend == "pytorch":
                prog_path = paths.root / f"training_progress_{tag}.jsonl"
                prog_path.write_text("", encoding="utf-8")
                os.environ["SURGE_TRAINING_PROGRESS_JSONL"] = str(prog_path.resolve())
            ckpt_every = int(model_cfg.params.get("checkpoint_every_n_epochs", 0) or 0)
            if _entry_backend == "pytorch" and ckpt_every > 0:
                ck_root = paths.root / "checkpoints" / tag
                ck_root.mkdir(parents=True, exist_ok=True)
                os.environ["SURGE_CHECKPOINT_DIR"] = str(ck_root.resolve())
            try:
                result = engine.train_model(
                    model_spec,
                    record=True,
                    pretrained_adapter=pretrained_adapter,
                    finetune_lr_scale=spec.finetune_lr_scale,
                )
            finally:
                os.environ.pop("SURGE_TRAINING_PROGRESS_JSONL", None)
                os.environ.pop("SURGE_CHECKPOINT_DIR", None)
            training_config = {
                "run_tag": run_tag,
                "dataset_path": posix_str(spec.dataset_path),
                "sample_rows": spec.sample_rows,
                "n_train": split_info["train"],
                "n_val": split_info["val"],
                "n_test": split_info["test"],
                "metadata_overrides": dict(spec.metadata_overrides or {}),
            }
            model_entry = _persist_model_artifacts(
                result,
                spec.predictions_scope,
                spec.predictions_format,
                paths,
                training_config=training_config,
            )

            workflow_metrics[model_entry["name"]] = {
                "train": result.train_metrics,
                "val": result.val_metrics,
                "test": result.test_metrics,
                "timings": result.timings,
                # profile lands in metrics.json too so leaderboards/CI can
                # compare model size and latency without reading the full
                # workflow_summary.json.
                "profile": (result.extra or {}).get("profile"),
            }
            if hpo_artifact:
                model_entry["artifacts"]["hpo"] = hpo_artifact
            workflow_models.append(model_entry)

        metrics_path = save_metrics(workflow_metrics, paths)
        ranges_path = paths.root / "train_data_ranges.json"
        summary = {
            "run_tag": run_tag,
            "dataset": dataset.summary(),
            "resources": resources.to_dict(),
            "splits": split_info,
            "scalers": scaler_artifacts,
            "models": workflow_models,
            "registry": registry_summary(),
        "artifacts": {
            "root": posix_str(paths.root),
            "metrics": posix_str(metrics_path),
            "summary": posix_str(paths.summary_file),
            "spec": posix_str(paths.spec_file),
            "train_data_ranges": posix_str(ranges_path) if ranges_path.exists() else None,
            "invocation": posix_str(paths.root / "invocation.json")
            if (paths.root / "invocation.json").exists()
            else None,
            "run_log": posix_str(paths.root / "run.log")
            if (paths.root / "run.log").exists()
            else None,
        },
    }
        save_workflow_summary(summary, paths)

        # Optional MLflow logging (AmSC-style tracking)
        if spec.mlflow_tracking:
            try:
                from ..integrations.mlflow_logger import log_surge_run
                log_surge_run(
                    paths.root,
                    experiment_name=spec.mlflow_experiment or "surge",
                    run_name=run_tag,
                )
            except ImportError:
                LOGGER.warning(
                    "MLflow not installed. pip install surge[mlflow]"
                )

        _progress_step(total_steps, total_steps, f"Done. Artifacts in {paths.root}")
        LOGGER.info("Workflow %s complete. Artifacts saved to %s", run_tag, paths.root)
        return summary
    finally:
        sys.stdout, sys.stderr = _real_out, _real_err
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass


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
    *,
    training_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model_name = result.spec.name or result.spec.key
    model_path = save_model(result.adapter, model_name, paths)
    save_model_card(
        result.adapter,
        model_name,
        paths,
        training_config=training_config,
    )
    prediction_files: Dict[str, str] = {}
    for split in prediction_splits:
        payload = result.predictions.get(split)
        if payload is None:
            continue
        prediction_files[split] = posix_str(
            save_predictions(payload, model_name, split, paths, format=predictions_format)
        )

    if "val_uq" in result.predictions:
        uq_payload = result.predictions["val_uq"]
        uq_path = paths.predictions_dir / f"{model_name}_val_uq.json"
        with uq_path.open("w", encoding="utf-8") as handle:
            json.dump(_ensure_serializable(uq_payload), handle, indent=2)
        prediction_files["val_uq"] = posix_str(uq_path)

    artifact_extra: Dict[str, str] = {}
    tag = _safe_model_artifact_tag(model_name)
    th = (result.extra or {}).get("training_history")
    if th:
        th_path = paths.root / f"training_history_{tag}.json"
        with th_path.open("w", encoding="utf-8") as handle:
            json.dump(th, handle, indent=2)
        artifact_extra["training_history"] = posix_str(th_path)
        log_path = paths.root / f"training_log_{tag}.jsonl"
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                for row in th:
                    handle.write(json.dumps(row, default=str) + "\n")
            artifact_extra["training_log_jsonl"] = posix_str(log_path.resolve())
        except OSError:
            LOGGER.warning("Could not write training_log_jsonl for %s", model_name, exc_info=True)
        try:
            from ..viz.training import plot_training_dashboard

            plots_dir = paths.root / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_training_dashboard(
                th,
                model_name=model_name,
                save_path=plots_dir / f"training_dashboard_{tag}.png",
            )
        except Exception:
            LOGGER.debug(
                "Training dashboard plot skipped for %s",
                model_name,
                exc_info=True,
            )
    prog_jsonl = paths.root / f"training_progress_{tag}.jsonl"
    if prog_jsonl.is_file() and prog_jsonl.stat().st_size > 0:
        artifact_extra["training_progress_jsonl"] = posix_str(prog_jsonl.resolve())
    ck_dir = paths.root / "checkpoints" / tag
    if ck_dir.is_dir() and any(ck_dir.glob("*.pt")):
        artifact_extra["checkpoints_dir"] = posix_str(ck_dir.resolve())

    entry: Dict[str, Any] = {
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
            "model": posix_str(model_path),
            "predictions": prediction_files,
            **artifact_extra,
        },
    }
    # Persist the resolved ResourceSpec + concrete backend knobs that the
    # engine emitted in the [surge.fit] banner, so every run records the
    # exact device/n_jobs/num_workers combination that trained the model.
    resources_used = getattr(result.adapter, "_last_fit_resources", None)
    if resources_used:
        entry["resources_used"] = resources_used
    # Model-size + inference-latency profile (see surge.model.profiling).
    profile = (result.extra or {}).get("profile")
    if profile:
        entry["profile"] = profile
    return entry


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

    hpo_track = False
    hpo_current: Optional[Path] = None
    hpo_all: Optional[Path] = None
    try:
        if _registry_backend(engine.registry, base_spec.key) == "pytorch":
            hpo_track = True
            hpo_current = paths.root / "training_progress_hpo_current.jsonl"
            hpo_all = paths.root / "training_progress_hpo_all.jsonl"
            for p in (
                hpo_current,
                hpo_all,
                paths.root / "training_progress_hpo_best.jsonl",
                paths.root / "hpo_trials_manifest.jsonl",
            ):
                p.write_text("", encoding="utf-8")
            (paths.root / "hpo_checkpoints").mkdir(parents=True, exist_ok=True)
    except KeyError:
        pass

    from optuna.trial import FixedTrial, TrialState

    def _hpo_manifest_best_cb(
        study: "optuna.Study", finished: "optuna.FrozenTrial"
    ) -> None:
        if finished.state != TrialState.COMPLETE or finished.value is None:
            return
        if not hpo_track:
            return
        resolved_params = _sample_hpo_params(
            FixedTrial(finished.params), config.search_space
        )
        arch = _hpo_architecture_from_trial_params(resolved_params)
        line = {
            "trial": finished.number,
            "value": float(finished.value),
            "metric": config.metric,
            "is_best_yet": study.best_trial.number == finished.number,
            "architecture": arch,
            "params": dict(finished.params),
        }
        ua = getattr(finished, "user_attrs", {}) or {}
        if ua:
            line["user_attrs"] = dict(ua)
        manifest_path = paths.root / "hpo_trials_manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as mh:
            mh.write(json.dumps(line, default=str) + "\n")
            mh.flush()
            os.fsync(mh.fileno())
        if study.best_trial.number == finished.number:
            _hpo_refresh_best_progress_jsonl(paths, finished.number)

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
            arch_view = _hpo_architecture_from_trial_params(
                {**base_spec.params, **sampled_params}
            )
            if arch_view.get("hidden_layers"):
                print(
                    f"  [HPO] Trial {trial.number} architecture: "
                    f"{arch_view['num_hidden_layers']} layers, widths={arch_view['hidden_layers']}",
                    flush=True,
                )

            trial_ckpt_dir: Optional[Path] = None
            if hpo_track:
                assert hpo_current is not None and hpo_all is not None
                trial_ckpt_dir = paths.root / "hpo_checkpoints" / f"trial_{trial.number:04d}"
                trial_ckpt_dir.mkdir(parents=True, exist_ok=True)
                os.environ["SURGE_CHECKPOINT_DIR"] = str(trial_ckpt_dir.resolve())
                hpo_current.write_text("", encoding="utf-8")
                os.environ["SURGE_TRAINING_PROGRESS_JSONL"] = str(hpo_current.resolve())
                os.environ["SURGE_TRAINING_PROGRESS_JSONL_MIRROR"] = str(
                    hpo_all.resolve()
                )
                os.environ["SURGE_HPO_TRIAL"] = str(trial.number)
            try:
                result = engine.train_model(trial_spec, record=False)
            finally:
                if hpo_track:
                    os.environ.pop("SURGE_TRAINING_PROGRESS_JSONL", None)
                    os.environ.pop("SURGE_TRAINING_PROGRESS_JSONL_MIRROR", None)
                    os.environ.pop("SURGE_HPO_TRIAL", None)
                    os.environ.pop("SURGE_CHECKPOINT_DIR", None)

            th = (result.extra or {}).get("training_history")
            _hpo_write_trial_training_history(paths, trial.number, th)
            if hpo_track and getattr(result.adapter, "backend", None) == "pytorch":
                try:
                    assert trial_ckpt_dir is not None
                    result.adapter.save(str(trial_ckpt_dir / "model_final.pt"))
                except Exception as ex:
                    LOGGER.warning(
                        "HPO trial %s: could not save checkpoint: %s",
                        trial.number,
                        ex,
                    )

            metric_value = _extract_metric(result, config.metric)
            # Store additional metrics for visualization (R², RMSE)
            trial.set_user_attr("val_r2", result.val_metrics.get("r2"))
            trial.set_user_attr("val_rmse", result.val_metrics.get("rmse"))
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
    callbacks.append(_hpo_manifest_best_cb)
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

    # Replay the search space against the winning trial's raw Optuna params so every
    # spec type (categorical `_index`, `variable_list` num-layers/log-int elements, ...)
    # is reconstructed the same way it was sampled during the study, instead of copying
    # raw Optuna keys (e.g. "hidden_layers_0_log") straight into the refit params.
    best_sampled = _sample_hpo_params(
        FixedTrial(study.best_trial.params), config.search_space
    )
    best_params = {**base_spec.params, **best_sampled}
    updated_spec = replace(base_spec, params=best_params)

    hpo_summary: Dict[str, Any] = {
        "metric": config.metric,
        "direction": config.direction,
        "best_trial": {
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "number": study.best_trial.number,
        },
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                **{k: v for k, v in getattr(trial, "user_attrs", {}).items()},
            }
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
    if hpo_track:
        hpo_summary["artifact_files"] = {
            "trials_manifest": posix_str((paths.root / "hpo_trials_manifest.jsonl").resolve()),
            "training_progress_current": posix_str(
                (paths.root / "training_progress_hpo_current.jsonl").resolve()
            ),
            "training_progress_all": posix_str(
                (paths.root / "training_progress_hpo_all.jsonl").resolve()
            ),
            "training_progress_best": posix_str(
                (paths.root / "training_progress_hpo_best.jsonl").resolve()
            ),
            "trial_training_history_glob": "hpo_trial_*_training_history.json",
            "hpo_checkpoints_dir": posix_str((paths.root / "hpo_checkpoints").resolve()),
        }
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

