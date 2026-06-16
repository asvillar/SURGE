#!/usr/bin/env python3
"""
ConStellaration 90→12 multi-output workflow — single Residual MLP.

Uses the paper-filtered cache (26,897 rows, 12 metrics). One model predicts
all outputs jointly (unlike ``constellaration_paper``, which trains 12 separate
90→1 models).

Examples
--------
    python examples/constellaration_multioutput_workflow.py --epochs 50

    python examples/constellaration_multioutput_workflow.py --hpo-trials 20

    surge run -b constellaration_multi -m pytorch.residual_mlp
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

from surge.benchmarks.dataset_io import (
    CONSTELLARATION_INPUT_NAMES,
    CONSTELLARATION_OUTPUT_NAMES,
)
from surge.benchmarks.leaderboard import _load_constellaration_paper

INPUT_NAMES = CONSTELLARATION_INPUT_NAMES
OUTPUT_NAMES = CONSTELLARATION_OUTPUT_NAMES


def _prepare_parquet(out_path: Path) -> Path:
    X, Y, names = _load_constellaration_paper()
    if X.shape[1] != len(INPUT_NAMES):
        raise ValueError(f"Expected {len(INPUT_NAMES)} inputs, got {X.shape[1]}")
    if Y.shape[1] != len(OUTPUT_NAMES):
        raise ValueError(f"Expected {len(OUTPUT_NAMES)} outputs, got {Y.shape[1]}")
    frame = pd.DataFrame(X, columns=INPUT_NAMES)
    for j, name in enumerate(names):
        frame[name] = Y[:, j]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConStellaration 90→12 Residual MLP workflow.")
    p.add_argument("--run-tag", default=None, help="Artifacts under runs/<run-tag>/")
    p.add_argument("--output-dir", type=Path, default=_REPO)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hpo-trials", type=int, default=0)
    p.add_argument("--no-hpo", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import HPOConfig, ModelConfig, SurrogateWorkflowSpec

    parquet_path = _REPO / "runs" / ".cache" / "constellaration_multioutput.parquet"
    _prepare_parquet(parquet_path)

    n_trials = 0 if args.no_hpo else args.hpo_trials
    run_tag = args.run_tag or (
        "constellaration_mimo_hpo" if n_trials > 0 else "constellaration_mimo_residual_mlp"
    )
    run_root = Path(args.output_dir) / "runs" / run_tag
    if run_root.exists() and not args.overwrite:
        raise SystemExit(
            f"Run directory already exists: {run_root}\n"
            "Use --overwrite or pass --run-tag <new_name>."
        )

    model_cfg = ModelConfig(
        key="pytorch.residual_mlp",
        name="constellaration_mimo_residual_mlp",
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
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.4},
            },
        )

    spec = SurrogateWorkflowSpec(
        dataset_path=parquet_path,
        dataset_format="parquet",
        run_tag=run_tag,
        output_dir=args.output_dir,
        seed=args.seed,
        metadata_overrides={"inputs": INPUT_NAMES, "outputs": OUTPUT_NAMES},
        overwrite_existing_run=args.overwrite,
        models=[model_cfg],
    )

    summary = run_surrogate_workflow(spec)
    root = summary["artifacts"]["root"]
    print(f"\nRun artifacts: {root}")
    print(f"  training log: {root}/training_progress_constellaration_mimo_residual_mlp.jsonl")
    print(f"  checkpoints:  {root}/checkpoints/constellaration_mimo_residual_mlp/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
