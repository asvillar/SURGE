"""Optional MLflow integration for SURGE run tracking (auditing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)


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
