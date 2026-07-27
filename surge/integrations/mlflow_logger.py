"""Optional MLflow integration for SURGE run tracking (auditing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

MLFLOW_AVAILABLE: bool
try:
    import mlflow as _mlflow  # noqa: F401
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_surge_run(
    run_dir: Path,
    *,
    experiment_name: str = "surge",
    run_name: Optional[str] = None,
) -> bool:
    """
    Log a SURGE run to MLflow: params, metrics, and artifacts.

    Args:
        run_dir: Path to the run directory (e.g. runs/xgc_target_set1_v2)
        experiment_name: MLflow experiment name
        run_name: MLflow run name (default: run_dir.stem)

    Returns:
        True if logging succeeded, False if MLflow is not installed or logging failed.
    """
    try:
        import mlflow
    except ImportError:
        LOG.warning(
            "MLflow not installed. Install with: pip install surge[mlflow]"
        )
        return False

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        LOG.warning("Run directory does not exist: %s", run_dir)
        return False

    run_name = run_name or run_dir.stem

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # Log params from spec
            spec_file = run_dir / "spec.yaml"
            if spec_file.exists():
                try:
                    import yaml
                    with spec_file.open() as f:
                        spec = yaml.safe_load(f)
                    if spec:
                        for k, v in spec.items():
                            if k == "models":
                                continue
                            if isinstance(v, (str, int, float, bool)):
                                mlflow.log_param(k, str(v))
                            elif isinstance(v, dict):
                                mlflow.log_params(
                                    {f"{k}.{kk}": str(vv) for kk, vv in v.items()}
                                )
                except Exception as e:
                    LOG.debug("Could not log spec params: %s", e)

            # Log params from workflow summary
            summary_file = run_dir / "workflow_summary.json"
            if summary_file.exists():
                try:
                    with summary_file.open() as f:
                        summary = json.load(f)
                    mlflow.log_param("run_tag", summary.get("run_tag", ""))
                    if "splits" in summary:
                        for k, v in summary["splits"].items():
                            mlflow.log_param(f"split_{k}", v)
                except Exception as e:
                    LOG.debug("Could not log summary params: %s", e)

            # Log metrics
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with metrics_file.open() as f:
                        metrics = json.load(f)
                    flat: Dict[str, float] = {}
                    for model_name, model_metrics in metrics.items():
                        if isinstance(model_metrics, dict):
                            for split in ("train", "val", "test"):
                                if (
                                    split in model_metrics
                                    and isinstance(model_metrics[split], dict)
                                ):
                                    for mk, mv in model_metrics[split].items():
                                        if isinstance(mv, (int, float)):
                                            flat[
                                                f"{model_name}.{split}.{mk}"
                                            ] = float(mv)
                    if flat:
                        mlflow.log_metrics(flat)
                except Exception as e:
                    LOG.debug("Could not log metrics: %s", e)

            # Per-epoch training curves (chart view in the MLflow UI):
            # training_log_<model>.jsonl -> <model>.train_loss / .val_loss
            for log_f in sorted(run_dir.glob("training_log_*.jsonl")):
                model_name = log_f.stem.replace("training_log_", "")
                try:
                    for line in log_f.read_text().splitlines():
                        rec = json.loads(line)
                        step = int(rec.get("epoch", 0))
                        series = {
                            f"{model_name}.{k}": float(v)
                            for k, v in rec.items()
                            if k != "epoch" and isinstance(v, (int, float))
                        }
                        if series:
                            mlflow.log_metrics(series, step=step)
                except Exception as e:
                    LOG.debug("Could not log training curve %s: %s", log_f, e)

            # HPO campaign: one nested child run per trial, with its params,
            # objective value, and per-epoch curves
            manifest = run_dir / "hpo_trials_manifest.jsonl"
            if manifest.exists():
                try:
                    for line in manifest.read_text().splitlines():
                        t = json.loads(line)
                        idx = int(t.get("trial", 0))
                        with mlflow.start_run(
                            run_name=f"hpo-trial-{idx:04d}", nested=True
                        ):
                            params = {
                                k: str(v)
                                for k, v in (t.get("params") or {}).items()
                            }
                            if params:
                                mlflow.log_params(params)
                            if isinstance(t.get("value"), (int, float)):
                                mlflow.log_metric(
                                    t.get("metric", "value"),
                                    float(t["value"]))
                            hist_f = (run_dir /
                                      f"hpo_trial_{idx:04d}_training_history.json")
                            if hist_f.exists():
                                for rec in json.loads(hist_f.read_text()):
                                    step = int(rec.get("epoch", 0))
                                    series = {
                                        k: float(v) for k, v in rec.items()
                                        if k != "epoch"
                                        and isinstance(v, (int, float))
                                    }
                                    if series:
                                        mlflow.log_metrics(series, step=step)
                except Exception as e:
                    LOG.debug("Could not log HPO trials: %s", e)

            # Log artifacts
            for name, subpath in [
                ("workflow_summary", "workflow_summary.json"),
                ("train_data_ranges", "train_data_ranges.json"),
            ]:
                p = run_dir / subpath
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path=name)

            models_dir = run_dir / "models"
            if models_dir.is_dir():
                mlflow.log_artifacts(str(models_dir), artifact_path="models")

            for card in run_dir.glob("model_card_*.json"):
                mlflow.log_artifact(str(card), artifact_path="model_cards")

        return True
    except Exception as e:
        LOG.warning("MLflow logging failed: %s", e)
        return False


def log_benchmark_result(
    result: Any,
    *,
    experiment_name: str = "surge_benchmarks",
    tracking_uri: Optional[str] = None,
    result_path: Optional[Path] = None,
) -> bool:
    """
    Log a :class:`~surge.benchmarks.base.BenchmarkResult` to MLflow.

    Creates (or reuses) an experiment named *experiment_name*. Each call
    starts a new child run tagged with the benchmark key, model, tier, and
    SURGE version so results are comparable across versions.

    Parameters
    ----------
    result:
        A ``BenchmarkResult`` instance.
    experiment_name:
        MLflow experiment name. Default ``"surge_benchmarks"``.
    tracking_uri:
        MLflow tracking URI (e.g. ``"http://localhost:5000"``). If ``None``
        the local ``./mlruns`` directory (MLflow default) is used.
    result_path:
        If the result has already been saved to disk, log that JSON file as
        an artifact so it is retrievable from the UI.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if MLflow is unavailable or logging
        fails (so callers can degrade gracefully).
    """
    if not MLFLOW_AVAILABLE:
        LOG.warning(
            "MLflow not installed. Install with: pip install 'surge-ml[mlflow]'"
        )
        return False

    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        run_name = f"{result.benchmark_key}__{result.model_key}"
        with mlflow.start_run(run_name=run_name):
            # ── Tags ────────────────────────────────────────────────────────
            mlflow.set_tags(
                {
                    "benchmark_key": result.benchmark_key,
                    "model_key": result.model_key,
                    "tier": result.tier,
                    "task_type": result.task_type,
                    "passed": str(result.passed),
                    "surge_version": result.surge_version or "",
                    "timestamp": result.timestamp or "",
                }
            )

            # ── Params ──────────────────────────────────────────────────────
            mlflow.log_params(
                {
                    "benchmark_key": result.benchmark_key,
                    "model_key": result.model_key,
                    "tier": result.tier,
                    "task_type": result.task_type,
                    "n_train": result.extra.get("n_train", ""),
                    "n_test": result.extra.get("n_test", ""),
                }
            )

            # ── Metrics ─────────────────────────────────────────────────────
            numeric_metrics: Dict[str, float] = {}
            for k, v in result.metrics.items():
                if isinstance(v, (int, float)) and k != "runtime_s":
                    numeric_metrics[k] = float(v)
            if "runtime_s" in result.metrics:
                numeric_metrics["runtime_s"] = float(result.metrics["runtime_s"])
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)

            # ── Artifact ────────────────────────────────────────────────────
            if result_path is not None:
                rp = Path(result_path)
                if rp.exists():
                    mlflow.log_artifact(str(rp), artifact_path="benchmark_result")

        return True
    except Exception as exc:
        LOG.warning("MLflow benchmark logging failed: %s", exc)
        return False
