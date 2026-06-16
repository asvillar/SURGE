#!/usr/bin/env python3
"""
QLKNN transport surrogate workflow — Residual MLP with loss logs and checkpoints.

The benchmark NPZ is not a workflow-native format; this script materialises a
Parquet table, then runs :func:`surge.workflow.run.run_surrogate_workflow`.

Examples
--------
    # Single train (200 epochs, checkpoints every 10, live loss bar)
    python examples/qlknn_workflow.py

    # Short smoke run
    python examples/qlknn_workflow.py --epochs 20 --no-hpo

    # HPO over architecture and regularisation (optimises val R²)
    python examples/qlknn_workflow.py --hpo-trials 40

    # Tail MSE/loss while training (second terminal)
    tail -f runs/qlknn_residual_mlp_hpo/training_progress_qlknn_residual_mlp.jsonl

    # If the run directory already exists, use --overwrite or a new --run-tag
    python examples/qlknn_workflow.py --hpo-trials 40 --run-tag qlknn_hpo_40
"""
from __future__ import annotations

import argparse
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


def _prepare_parquet(npz_path: Path, out_path: Path) -> Path:
    data = np.load(npz_path)
    X, y = data["X"], np.asarray(data["y"]).ravel()
    if X.shape[1] != len(QLKNN_FEATURES):
        raise ValueError(
            f"Expected {len(QLKNN_FEATURES)} QLKNN inputs, got {X.shape[1]} columns in {npz_path}"
        )
    frame = pd.DataFrame(X, columns=QLKNN_FEATURES)
    frame[TARGET] = y
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLKNN Residual MLP workflow with optional HPO.")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Source QLKNN NPZ cache")
    p.add_argument("--run-tag", default=None, help="Artifacts under runs/<run-tag>/ (auto if omitted)")
    p.add_argument("--output-dir", type=Path, default=_REPO, help="Parent dir; artifacts go to <output-dir>/runs/<run-tag>/")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hpo-trials", type=int, default=0, help="0 = train only; >0 enables Optuna HPO")
    p.add_argument("--no-hpo", action="store_true", help="Alias for --hpo-trials 0")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        raise SystemExit(
            f"QLKNN cache not found: {args.npz}\n"
            "Run: surge run -b qlknn -m sklearn.random_forest  (downloads NPZ on first use)"
        )

    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import HPOConfig, ModelConfig, SurrogateWorkflowSpec

    parquet_path = _REPO / "runs" / ".cache" / "qlknn_transport.parquet"
    _prepare_parquet(args.npz, parquet_path)

    n_trials = 0 if args.no_hpo else args.hpo_trials
    run_tag = args.run_tag or ("qlknn_residual_mlp_hpo" if n_trials > 0 else "qlknn_residual_mlp")
    run_root = Path(args.output_dir) / "runs" / run_tag
    if run_root.exists() and not args.overwrite:
        raise SystemExit(
            f"Run directory already exists: {run_root}\n"
            "Use --overwrite to replace it, or pick a new name:\n"
            f"  python examples/qlknn_workflow.py --hpo-trials {n_trials or 40} "
            f"--run-tag {run_tag}_v2"
        )

    model_cfg = ModelConfig(
        key="pytorch.residual_mlp",
        name="qlknn_residual_mlp",
        params={
            "n_epochs": 50 if n_trials > 0 else args.epochs,
            "verbose": True,
            "checkpoint_every_n_epochs": args.checkpoint_every,
        },
    )
    if n_trials > 0:
        model_cfg.hpo = HPOConfig(
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

    spec = SurrogateWorkflowSpec(
        dataset_path=parquet_path,
        dataset_format="parquet",
        run_tag=run_tag,
        output_dir=args.output_dir,
        seed=args.seed,
        metadata_overrides={"inputs": QLKNN_FEATURES, "outputs": [TARGET]},
        overwrite_existing_run=args.overwrite,
        models=[model_cfg],
    )

    summary = run_surrogate_workflow(spec)
    root = summary["artifacts"]["root"]
    print(f"\nRun artifacts: {root}")
    print(f"  training log: {root}/training_progress_qlknn_residual_mlp.jsonl")
    print(f"  checkpoints:  {root}/checkpoints/qlknn_residual_mlp/")
    if n_trials > 0:
        print(f"  HPO manifest: {root}/hpo_trials_manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
