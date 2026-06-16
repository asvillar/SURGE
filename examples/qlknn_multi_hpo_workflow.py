#!/usr/bin/env python3
"""
QLKNN plasma transport — multi-model workflow with HPO.

Trains **sklearn.random_forest** and **pytorch.residual_mlp** in one
``run_surrogate_workflow`` call, each with its own Optuna search. Predicts
electron ITG heat flux (``efeITG``) from 10 gyrokinetic plasma parameters.

On first run the QLKNN NPZ cache is generated via ``fusion_surrogates``
(Python ≥ 3.10 required). Subsequent runs reuse
``data/datasets/benchmarks/plasma/qlknn_transport.npz``.

Examples
--------
    pip install fusion_surrogates
    python examples/qlknn_multi_hpo_workflow.py --hpo-trials 10 --overwrite

    python examples/qlknn_multi_hpo_workflow.py --hpo-trials 3 --epochs 25 --overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from surge.benchmarks.dataset_io import QLKNN_FEATURE_NAMES, QLKNN_TARGET_NAME

QLKNN_FEATURES = QLKNN_FEATURE_NAMES
TARGET = QLKNN_TARGET_NAME
DEFAULT_NPZ = _REPO / "data/datasets/benchmarks/plasma/qlknn_transport.npz"
PARQUET_CACHE = _REPO / "runs" / ".cache" / "qlknn_transport.parquet"


def _ensure_npz(npz_path: Path) -> None:
    if npz_path.is_file():
        return
    print(f"[qlknn] cache missing — generating {npz_path} via fusion_surrogates …")
    from surge.benchmarks.leaderboard import _load_qlknn_transport

    _load_qlknn_transport()


def _prepare_parquet(npz_path: Path, out_path: Path) -> Path:
    data = np.load(npz_path)
    X, y = data["X"], np.asarray(data["y"]).ravel()
    if X.shape[1] != len(QLKNN_FEATURES):
        raise ValueError(
            f"Expected {len(QLKNN_FEATURES)} QLKNN inputs, got {X.shape[1]} in {npz_path}"
        )
    frame = pd.DataFrame(X, columns=QLKNN_FEATURES)
    frame[TARGET] = y
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path


def _build_models(n_trials: int, epochs: int) -> list:
    from surge.workflow.spec import HPOConfig, ModelConfig

    train_epochs = min(epochs, 50) if n_trials > 0 else epochs

    rf = ModelConfig(
        key="sklearn.random_forest",
        name="qlknn_rf",
        params={"n_jobs": -1},
    )
    if n_trials > 0:
        rf.hpo = HPOConfig(
            enabled=True,
            n_trials=n_trials,
            direction="maximize",
            metric="val_r2",
            search_space={
                "n_estimators": {"type": "int", "low": 50, "high": 400},
                "max_depth": {
                    "type": "categorical",
                    "choices": [None, 8, 12, 16, 24],
                },
                "min_samples_leaf": {"type": "int", "low": 1, "high": 8},
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2", 0.5, 0.8],
                },
            },
        )

    mlp = ModelConfig(
        key="pytorch.residual_mlp",
        name="qlknn_residual_mlp",
        params={
            "n_epochs": train_epochs,
            "verbose": True,
            "checkpoint_every_n_epochs": 10,
        },
    )
    if n_trials > 0:
        mlp.hpo = HPOConfig(
            enabled=True,
            n_trials=n_trials,
            direction="maximize",
            metric="val_r2",
            search_space={
                "hidden_layers": {
                    "type": "categorical",
                    "choices": [[128, 128], [256, 128], [256, 256, 128]],
                },
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.4},
            },
        )

    return [rf, mlp]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLKNN multi-model workflow (RF + Residual MLP) with optional HPO.",
    )
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--run-tag", default=None)
    p.add_argument("--output-dir", type=Path, default=_REPO)
    p.add_argument("--epochs", type=int, default=200, help="Residual MLP epochs when HPO is off")
    p.add_argument("--hpo-trials", type=int, default=10)
    p.add_argument("--no-hpo", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_trials = 0 if args.no_hpo else args.hpo_trials
    run_tag = args.run_tag or (
        "qlknn_multi_hpo" if n_trials > 0 else "qlknn_multi"
    )
    run_root = Path(args.output_dir) / "runs" / run_tag
    if run_root.exists() and not args.overwrite:
        raise SystemExit(
            f"Run directory already exists: {run_root}\n"
            "Use --overwrite or pass --run-tag <new_name>."
        )

    try:
        _ensure_npz(args.npz)
    except ImportError as exc:
        raise SystemExit(
            "QLKNN cache not found and fusion_surrogates is not installed.\n"
            "  pip install fusion_surrogates   # requires Python ≥ 3.10\n"
            "Or run once: surge run -b qlknn -m sklearn.random_forest"
        ) from exc

    _prepare_parquet(args.npz, PARQUET_CACHE)

    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    spec = SurrogateWorkflowSpec(
        dataset_path=PARQUET_CACHE,
        dataset_format="parquet",
        run_tag=run_tag,
        output_dir=args.output_dir,
        seed=args.seed,
        metadata_overrides={"inputs": QLKNN_FEATURES, "outputs": [TARGET]},
        overwrite_existing_run=args.overwrite,
        models=_build_models(n_trials, args.epochs),
    )

    summary = run_surrogate_workflow(spec)
    root = Path(summary["artifacts"]["root"])
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    print(f"\nRun artifacts: {root}")
    for name in ("qlknn_rf", "qlknn_residual_mlp"):
        if name in metrics:
            test = metrics[name].get("test", {})
            print(f"  {name:22s}  test_r2={test.get('r2', float('nan')):.4f}")
    if n_trials > 0:
        print(f"  HPO manifest: {root}/hpo_trials_manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
