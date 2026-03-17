"""Artifact helpers for SURGE workflow runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False

from ..registry import BaseModelAdapter


@dataclass
class ArtifactPaths:
    root: Path
    models_dir: Path
    scalers_dir: Path
    predictions_dir: Path
    metrics_file: Path
    summary_file: Path
    spec_file: Path
    env_file: Path
    git_rev_file: Path
    hpo_dir: Path


def init_artifact_paths(
    output_dir: Union[str, Path],
    run_tag: str,
    *,
    exist_ok: bool = False,
) -> ArtifactPaths:
    root = Path(output_dir) / "runs" / run_tag
    if root.exists() and not exist_ok:
        raise FileExistsError(f"Run directory already exists: {root}")

    models_dir = root / "models"
    scalers_dir = root / "scalers"
    predictions_dir = root / "predictions"
    hpo_dir = root / "hpo"

    for directory in (models_dir, scalers_dir, predictions_dir, hpo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ArtifactPaths(
        root=root,
        models_dir=models_dir,
        scalers_dir=scalers_dir,
        predictions_dir=predictions_dir,
        metrics_file=root / "metrics.json",
        summary_file=root / "workflow_summary.json",
        spec_file=root / "spec.yaml",
        env_file=root / "env.txt",
        git_rev_file=root / "git_rev.txt",
        hpo_dir=hpo_dir,
    )


def save_metrics(metrics: Mapping[str, Any], paths: ArtifactPaths) -> Path:
    _write_json(paths.metrics_file, metrics)
    return paths.metrics_file


def save_workflow_summary(summary: Mapping[str, Any], paths: ArtifactPaths) -> Path:
    _write_json(paths.summary_file, summary)
    return paths.summary_file


def save_spec(spec: Mapping[str, Any], paths: ArtifactPaths) -> Path:
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required to serialize workflow specs.")
    with paths.spec_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(spec), handle, sort_keys=False)
    return paths.spec_file


def save_environment_snapshot(paths: ArtifactPaths, extras: Optional[Mapping[str, Any]] = None) -> Path:
    lines = [
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"cwd: {os.getcwd()}",
    ]
    if extras:
        for key, value in extras.items():
            lines.append(f"{key}: {value}")
    with paths.env_file.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return paths.env_file


def save_git_revision(paths: ArtifactPaths, repo_dir: Optional[Union[str, Path]] = None) -> Path:
    repo_dir = Path(repo_dir or ".")
    try:
        rev = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        rev = "unknown"
    with paths.git_rev_file.open("w", encoding="utf-8") as handle:
        handle.write(rev + "\n")
    return paths.git_rev_file


def save_model(adapter: BaseModelAdapter, name: str, paths: ArtifactPaths) -> Path:
    target = paths.models_dir / f"{name}.joblib"
    try:
        adapter.save(target)
        return target
    except Exception:
        pass
    joblib.dump(adapter, target, protocol=4)
    return target


def save_scaler(scaler: Any, name: str, paths: ArtifactPaths) -> Path:
    target = paths.scalers_dir / f"{name}.joblib"
    joblib.dump(scaler, target, protocol=4)
    return target


def save_train_data_ranges(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_columns: list,
    output_columns: list,
    paths: ArtifactPaths,
) -> Path:
    """
    Save min/max of training data (in model-input space) for in-distribution checks.

    Use when evaluating new datastreamsets: compare datastreamset min/max to these ranges.
    """
    target = paths.root / "train_data_ranges.json"
    payload = {
        "inputs": {
            "columns": input_columns,
            "min": X_train.min(axis=0).tolist(),
            "max": X_train.max(axis=0).tolist(),
        },
        "outputs": {
            "columns": output_columns,
            "min": y_train.min(axis=0).tolist(),
            "max": y_train.max(axis=0).tolist(),
        },
    }
    _write_json(target, payload)
    return target


def save_predictions(
    predictions: Union[pd.DataFrame, np.ndarray, Mapping[str, Any]],
    name: str,
    split: str,
    paths: ArtifactPaths,
    *,
    format: str = "parquet",
) -> Path:
    filename = f"{name}_{split}.{ 'parquet' if format == 'parquet' else 'csv'}"
    target = paths.predictions_dir / filename
    df = _ensure_dataframe(predictions)
    if format == "parquet":
        df.to_parquet(target, index=False)
    else:
        df.to_csv(target, index=False)
    return target


def save_hpo_results(results: Mapping[str, Any], paths: ArtifactPaths, filename: str = "hpo_results.json") -> Path:
    target = paths.hpo_dir / filename
    _write_json(target, results)
    return target


def _ensure_dataframe(data: Union[pd.DataFrame, np.ndarray, Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray):
        rows, cols = data.shape if data.ndim == 2 else (len(data), 1)
        columns = [f"y_{idx}" for idx in range(cols)]
        return pd.DataFrame(data.reshape(rows, cols), columns=columns)
    if isinstance(data, Mapping):
        frame_dict: Dict[str, Any] = {}
        for key, value in data.items():
            arr = np.asarray(value)
            if arr.ndim == 1:
                frame_dict[key] = arr
            else:
                for idx in range(arr.shape[1]):
                    frame_dict[f"{key}_{idx:02d}"] = arr[:, idx]
        return pd.DataFrame(frame_dict)
    raise TypeError(f"Unsupported prediction payload type: {type(data)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)

