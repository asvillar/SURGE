#!/usr/bin/env python3
"""
SURGE quickstart CLI.

End-to-end: fetch a public scikit-learn regression dataset, train a surrogate
with SURGE, print metrics + profile, and optionally round-trip the trained
artifact for inference. No files outside the current directory are created
beyond ``<dataset>.csv`` and ``runs/<tag>/``.

Examples
--------
    # 442-row diabetes benchmark, random forest, ~5 s on a laptop
    python -m examples.quickstart --dataset diabetes

    # 20,640-row California housing benchmark, ~30 s on a laptop
    python -m examples.quickstart --dataset california

    # PyTorch MLP, 80 epochs (no HPO), same dataset
    python -m examples.quickstart --dataset california --model mlp

    # Add a short Optuna HPO sweep over the MLP
    python -m examples.quickstart --dataset diabetes --model mlp --n-trials 20

    # Only run the inference round-trip against an existing run
    python -m examples.quickstart --dataset diabetes --skip-train --infer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Stay importable even when run from a checkout that isn't pip-installed.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------- datasets ---

# Each entry returns (csv_path, target_column, n_rows, n_inputs).
# We write the CSV into the current working directory so the user can inspect
# it and re-run inference without relying on sklearn's cache.
def _write_diabetes(out_path: Path) -> Tuple[Path, str, list[str], int]:
    from sklearn.datasets import load_diabetes  # type: ignore

    frame = load_diabetes(as_frame=True).frame
    frame.to_csv(out_path, index=False)
    inputs = [c for c in frame.columns if c != "target"]
    return out_path, "target", inputs, len(frame)


def _write_california(out_path: Path) -> Tuple[Path, str, list[str], int]:
    from sklearn.datasets import fetch_california_housing  # type: ignore

    frame = fetch_california_housing(as_frame=True).frame
    frame.to_csv(out_path, index=False)
    inputs = [c for c in frame.columns if c != "MedHouseVal"]
    return out_path, "MedHouseVal", inputs, len(frame)


# Each value is (csv_filename, writer, human_blurb).
# The writer returns (csv_path, target_col, input_cols, n_rows) so the caller
# can plumb explicit input/output hints into the workflow spec — this bypasses
# SURGE's prefix-based auto-detection, which doesn't recognize sklearn's CSV
# conventions (no "y_" / "target_" prefix on the target column).
DATASETS = {
    "diabetes": ("diabetes.csv", _write_diabetes,
                 "scikit-learn diabetes (442 rows, 10 inputs -> 1 output)"),
    "california": ("california.csv", _write_california,
                   "California housing (20,640 rows, 8 inputs -> 1 output)"),
}


# ------------------------------------------------------------------ models ---

def _build_model_entry(model: str, n_trials: int) -> Dict[str, Any]:
    """Return a SurrogateWorkflowSpec-compatible `models[i]` dict."""
    if model == "rf":
        entry: Dict[str, Any] = {
            "key": "sklearn.random_forest",
            "params": {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": -1},
        }
    elif model == "mlp":
        entry = {
            "key": "pytorch.mlp",
            "params": {
                "hidden_layers": [128, 64],
                "dropout_rate": 0.1,
                "learning_rate": 1e-3,
                "n_epochs": 200,
                "batch_size": 256,
            },
        }
    else:  # pragma: no cover — argparse restricts choices
        raise ValueError(f"unknown model: {model}")

    if n_trials > 0:
        entry["hpo"] = {
            "enabled": True,
            "n_trials": n_trials,
            "metric": "val_rmse",
            "direction": "minimize",
            "sampler": "tpe",
        }
        if model == "mlp":
            entry["hpo"]["search_space"] = {
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
                "dropout_rate":  {"type": "float",      "low": 0.05, "high": 0.5},
                "batch_size":    {"type": "categorical", "choices": [32, 64, 128, 256]},
                "hidden_layers": {"type": "categorical", "choices": [
                    [64], [128], [128, 64], [256, 128], [256, 128, 64],
                ]},
            }
        else:  # rf
            entry["hpo"]["search_space"] = {
                "n_estimators":     {"type": "int",  "low": 50, "high": 400},
                "max_depth":        {"type": "int",  "low": 3,  "high": 20},
                "min_samples_leaf": {"type": "int",  "low": 1,  "high": 10},
            }

    return entry


# --------------------------------------------------------------- artifacts --

def _fmt_bytes(n: int) -> str:
    """Compact, byte-accurate human rendering (B / KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# Annotations for the most common artifacts; unknown files just get sized.
_ARTIFACT_NOTES: Dict[str, str] = {
    "spec.yaml":               "workflow spec (re-runnable)",
    "env.txt":                 "pip freeze at run time",
    "git_rev.txt":             "repo HEAD or 'unknown'",
    "run.log":                 "stdout capture",
    "workflow_summary.json":   "metrics + profile + resources_used",
    "metrics.json":            "per-model train/val/test + timings",
    "train_data_ranges.json":  "canonical input column order + min/max",
}

# Suffix- or prefix-based fallbacks for files whose stems include the model key.
_ARTIFACT_NOTE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("model_card_",        "data + model provenance card"),
    ("training_history_",  "per-epoch loss / metric curves"),
    ("training_progress_", "streaming JSONL progress log"),
)
_ARTIFACT_NOTE_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    (".onnx",              "ONNX export for cross-runtime inference"),
)


def _print_artifact_tree(run_dir: Path) -> None:
    """Walk runs/<tag>/ and render a tree with sizes + annotations.

    Mirrors the layout shown in the README quickstart so the CLI output is
    self-describing: a user can tell at a glance which file carries which
    piece of the run.
    """
    if not run_dir.is_dir():
        print(f"[artifacts] (no directory at {run_dir})")
        return

    print(f"\n[artifacts] {run_dir}")

    entries = sorted(run_dir.iterdir(), key=lambda p: (p.is_file(), p.name))
    for idx, entry in enumerate(entries):
        is_last = idx == len(entries) - 1
        branch = "└── " if is_last else "├── "
        if entry.is_file():
            note = _ARTIFACT_NOTES.get(entry.name, "")
            if not note:
                for prefix, pref_note in _ARTIFACT_NOTE_PATTERNS:
                    if entry.name.startswith(prefix):
                        note = pref_note
                        break
            if not note:
                for suffix, suf_note in _ARTIFACT_NOTE_SUFFIXES:
                    if entry.name.endswith(suffix):
                        note = suf_note
                        break
            size = _fmt_bytes(entry.stat().st_size)
            extra = f"  — {note}" if note else ""
            print(f"  {branch}{entry.name:36s}  {size:>9s}{extra}")
            continue

        print(f"  {branch}{entry.name}/")
        prefix = "      " if is_last else "  │   "
        sub_entries = sorted(entry.iterdir())
        for j, sub in enumerate(sub_entries):
            sub_last = j == len(sub_entries) - 1
            sub_branch = "└── " if sub_last else "├── "
            if sub.is_file():
                size = _fmt_bytes(sub.stat().st_size)
                print(f"{prefix}{sub_branch}{sub.name:32s}  {size:>9s}")
            else:
                print(f"{prefix}{sub_branch}{sub.name}/")


# ---------------------------------------------------------- visualization ---

def _demo_viz(run_dir: Path) -> None:
    """Generate parity plots (predicted vs ground truth) under runs/<tag>/plots/.

    Wraps ``surge.viz.viz_run``, which reads every
    ``predictions/<model>_<split>.parquet`` and builds a 2D-density
    "regression map" per output with R² annotations and a diagonal
    reference.
    """
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        from surge.viz.run_viz import viz_run  # type: ignore
    except ImportError as exc:
        print(f"\n[viz] matplotlib / surge.viz unavailable: {exc}")
        return

    result = viz_run(run_dir=run_dir, dpi=120, include_hpo=True)
    paths = result.get("saved_paths") or []
    if not paths:
        print("\n[viz] no plots saved (no predictions found?)")
        return

    print("\n[viz] parity plots (predictions vs ground truth):")
    for p in paths:
        print(f"        {Path(p).relative_to(run_dir)}")


# --------------------------------------------------------------- inference ---

def _demo_inference(run_dir: Path, csv_path: Path, target_col: str) -> None:
    """Load the saved model + scaler and predict on the first 5 rows.

    Handles both sklearn and pytorch backends. SURGE writes both as
    ``models/<key>.joblib`` but the file format differs:

    * **sklearn**: a joblib-pickled estimator; ``joblib.load`` returns an
      object whose ``.predict`` takes *scaled* inputs and returns raw targets.
    * **pytorch**: a torch-zip archive containing the state_dict plus the
      input/output scalers baked in. The ``PyTorchMLPModel`` class rebuilds
      itself from this and handles both scalers internally on ``.predict``.
    """
    import json

    import joblib  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    candidates = sorted(run_dir.glob("models/*.joblib"))
    if not candidates:
        print(f"\n[infer] no model artifact under {run_dir}/models/")
        return

    model_path = candidates[0]
    backend = "pytorch" if model_path.name.startswith("pytorch") else "sklearn"

    # SURGE sorts input columns alphabetically during preprocessing, so we
    # must read the canonical order from the run's train_data_ranges.json
    # (written by the workflow) rather than trusting the CSV column order.
    ranges_path = run_dir / "train_data_ranges.json"
    if not ranges_path.exists():
        print(f"\n[infer] missing {ranges_path}; cannot infer canonical input order")
        return
    input_cols = json.loads(ranges_path.read_text())["inputs"]["columns"]

    df = pd.read_csv(csv_path)
    x_head = df[input_cols].head(5)
    y_true = df[target_col].head(5).values

    # SURGE pre-scales inputs via runs/<tag>/scalers/inputs.joblib before
    # handing X to the adapter during fit. To reproduce that at inference
    # time we apply the same scaler (in the same column order it was fit on)
    # before calling predict.
    scaler_x = joblib.load(run_dir / "scalers" / "inputs.joblib")
    X_scaled = scaler_x.transform(x_head.values)

    if backend == "sklearn":
        model = joblib.load(model_path)
        y_hat = np.asarray(model.predict(X_scaled)).ravel()
    else:
        # PyTorch path: rebuild PyTorchMLPModel from the torch-zip checkpoint.
        # Its internal scaler_X was fit on already-scaled inputs (near-identity)
        # and its scaler_y handles the inverse transform back to the original
        # target units.
        try:
            from surge.model.pytorch_impl import PyTorchMLPModel  # type: ignore
        except ImportError as exc:
            print(f"\n[infer] torch backend requested but PyTorch unavailable: {exc}")
            return
        wrapped = PyTorchMLPModel()
        wrapped.load(model_path)
        y_hat = np.asarray(wrapped.predict(X_scaled)).ravel()

    print("\n[infer] round-trip inference on the first 5 rows:")
    print(f"        model  : {model_path.name}  ({backend})")
    print(f"        inputs : {input_cols}")
    print(f"        y_true : {np.round(y_true, 2)}")
    print(f"        y_hat  : {np.round(y_hat, 2)}")


# -------------------------------------------------------------------- main ---

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="examples.quickstart",
        description="Train a SURGE surrogate on a public scikit-learn dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="diabetes",
        help="Which public dataset to use (default: diabetes).",
    )
    parser.add_argument(
        "--model", choices=["rf", "mlp"], default="rf",
        help="rf = sklearn.random_forest, mlp = pytorch.mlp (default: rf).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=0,
        help="Optuna HPO trials. 0 (default) disables HPO; >0 enables it.",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Name of the run directory under output-dir/runs/. "
             "Defaults to '<dataset>_<model>[_hpo<N>]'.",
    )
    parser.add_argument(
        "--output-dir", default=str(_REPO),
        help=(
            "Parent directory for runs/ and the generated CSV. "
            "Defaults to the SURGE repo root so `runs/<tag>/` lands next to "
            "the source tree regardless of where you run python from."
        ),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="cpu",
        help="Resource hint passed to the workflow (default: cpu).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Worker count hint for sklearn / torch DataLoader (default: 4).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Don't train — useful with --infer against an existing run.",
    )
    parser.add_argument(
        "--infer", action="store_true",
        help="After training, round-trip the saved model on the first 5 rows.",
    )
    parser.add_argument(
        "--viz", action="store_true",
        help=(
            "Generate parity / regression-map plots under runs/<tag>/plots/ "
            "via surge.viz.viz_run (requires matplotlib)."
        ),
    )
    args = parser.parse_args(argv)

    # Lazy imports: keep argparse --help fast and avoid surfacing backend
    # warnings (GPflow, etc.) unless we actually run something.
    from surge import SurrogateWorkflowSpec, run_surrogate_workflow  # type: ignore
    from surge.hpc import ResourceSpec  # type: ignore

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_name, writer, blurb = DATASETS[args.dataset]
    csv_path = out_dir / csv_name

    if not args.skip_train:
        print(f"[quickstart] dataset : {args.dataset} — {blurb}")
        print(f"[quickstart] writing : {csv_path}")
        csv_path, target_col, input_cols, n_rows = writer(csv_path)
        print(f"[quickstart]           {n_rows:,} rows, "
              f"{len(input_cols)} inputs -> 1 output ({target_col})")

        run_tag = args.run_tag or (
            f"{args.dataset}_{args.model}"
            + (f"_hpo{args.n_trials}" if args.n_trials > 0 else "")
        )
        print(f"[quickstart] model   : {args.model}  "
              f"(HPO trials: {args.n_trials})")
        print(f"[quickstart] run tag : {run_tag}")

        spec = SurrogateWorkflowSpec(
            dataset_path=str(csv_path),
            # Explicit I/O — SURGE's prefix-based auto-detection is tuned for
            # scientific-simulation column names and doesn't match sklearn's
            # CSV convention (no "y_" / "target_" prefix).
            metadata_overrides={"inputs": input_cols, "outputs": [target_col]},
            models=[_build_model_entry(args.model, args.n_trials)],
            resources=ResourceSpec(device=args.device, num_workers=args.num_workers),
            output_dir=str(out_dir),
            run_tag=run_tag,
            overwrite_existing_run=True,
        )
        summary = run_surrogate_workflow(spec)

        m = summary["models"][0]
        prof = m.get("profile", {}) or {}
        print()
        print(f"  train R2    = {m['metrics']['train']['r2']:.3f}")
        print(f"  val   R2    = {m['metrics']['val']['r2']:.3f}")
        print(f"  test  R2    = {m['metrics']['test']['r2']:.3f}")
        print(f"  test  RMSE  = {m['metrics']['test']['rmse']:.3f}")
        size_kb = (prof.get("model_size_bytes") or 0) / 1024
        n_params = prof.get("parameter_count")
        params_str = f"{n_params:,} parameters" if n_params else "parameter count n/a"
        print(f"  model       = {size_kb:,.1f} KB, {params_str}")
        inf_ms = prof.get("inference_ms_per_sample")
        if inf_ms is not None:
            print(f"  inference   = {inf_ms:.2f} ms/sample")

        run_dir = out_dir / "runs" / run_tag
    else:
        # --skip-train: discover the most recent run under runs/ for --infer
        run_tag = args.run_tag or f"{args.dataset}_{args.model}"
        target_col = "target" if args.dataset == "diabetes" else "MedHouseVal"
        run_dir = out_dir / "runs" / run_tag
        if not run_dir.is_dir():
            print(f"[quickstart] no run at {run_dir} — omit --skip-train or set --run-tag")
            return 1

    if args.viz:
        _demo_viz(run_dir)

    _print_artifact_tree(run_dir)

    if args.infer:
        _demo_inference(run_dir, csv_path, target_col)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
