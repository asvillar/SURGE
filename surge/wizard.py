"""``surge init`` — interactive spec wizard.

Inspects a data file with the existing loader/schema-inference, asks
three questions (target, goal, budget), suggests a model slate by task
shape, and writes a fully commented ``spec.yaml``. Non-interactive flags
(``--data --target --goal --budget --yes``) cover scripting and tests.

YAML stays the source of truth — the wizard only generates it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

GOALS = ("accuracy", "speed", "uncertainty")
BUDGETS = ("smoke", "standard", "thorough")

_BUDGET_NOTES = {
    "smoke": "~1 min: subsampled data, few epochs, no HPO",
    "standard": "~10 min: full data, HPO 10 trials on the headline model",
    "thorough": "hours: HPO 40 trials; rerun with --seeds 3 for error bars",
}


def _ask(prompt: str, default: str, choices: tuple[str, ...] | None = None) -> str:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip() or default
        if choices is None or raw in choices:
            return raw
        print(f"  please choose one of: {', '.join(choices)}")


def _inspect(data_path: Path, sample_rows: int = 5000) -> dict[str, Any]:
    """Load (a sample of) the dataset and characterize its columns."""
    from surge.dataset import SurrogateDataset

    ds = SurrogateDataset.from_path(data_path)
    df = ds.dataframe if hasattr(ds, "dataframe") else ds.df
    n_rows, n_cols = df.shape
    numeric = df.select_dtypes("number").columns.tolist()
    return {"df": df, "n_rows": n_rows, "n_cols": n_cols, "numeric": numeric}


def _target_candidates(info: dict[str, Any]) -> list[str]:
    """Plausible target columns: known names first, then the last numeric."""
    hints = ("target", "y", "label", "output", "efe", "gamma", "growth")
    numeric = info["numeric"]
    named = [c for c in numeric if any(h in c.lower() for h in hints)]
    tail = [c for c in reversed(numeric) if c not in named]
    return (named + tail)[:5]


def _model_slate(goal: str, budget: str, n_rows: int,
                 available: set[str]) -> list[dict[str, Any]]:
    """Model blocks by goal/size — mirrors surge-build-surrogate guidance."""
    def have(key: str, fallback: str | None = None) -> str | None:
        if key in available:
            return key
        return fallback if fallback is None or fallback in available else None

    slate: list[dict[str, Any]] = []
    if goal == "speed":
        slate.append({"key": "sklearn.ridge", "name": "baseline_ridge",
                      "comment": "linear baseline — anchors the leaderboard"})
        gbm = have("lgbm.regressor", "sklearn.gradient_boosting_regressor")
        if gbm:
            slate.append({"key": gbm, "name": "fast_gbm",
                          "comment": "fast, strong tabular default"})
    elif goal == "uncertainty":
        slate.append({"key": "sklearn.random_forest", "name": "baseline_rf",
                      "comment": "robust baseline"})
        if n_rows <= 5000 and "sklearn.gpr" in available:
            slate.append({"key": "sklearn.gpr", "name": "gp",
                          "comment": "exact GP — calibrated bands (n <= 5k)",
                          "request_uncertainty": True})
        elif have("pytorch.mlp_ensemble"):
            slate.append({"key": "pytorch.mlp_ensemble", "name": "ensemble",
                          "comment": "deep ensemble — mean +- spread "
                                     "(calibrate on held-out data)",
                          "request_uncertainty": True})
    else:  # accuracy
        slate.append({"key": "sklearn.random_forest", "name": "baseline_rf",
                      "comment": "robust baseline — anchors the leaderboard"})
        nn = have("pytorch.residual_mlp")
        if nn:
            block: dict[str, Any] = {
                "key": nn, "name": "residual_mlp",
                "comment": "headline model — residual MLP with smoothed "
                           "early stopping"}
            if budget == "smoke":
                block["params"] = {"n_epochs": 10}
            else:
                trials = 10 if budget == "standard" else 40
                block["hpo"] = {
                    "n_trials": trials, "direction": "maximize",
                    "metric": "val_r2",
                    "search_space": {
                        "hidden_layers": {"type": "categorical",
                                          "choices": [[128, 128], [256, 128],
                                                      [256, 256, 128]]},
                        "learning_rate": {"type": "loguniform",
                                          "low": 1e-4, "high": 1e-1},
                        "dropout_rate": {"type": "float",
                                         "low": 0.0, "high": 0.4},
                    }}
            slate.append(block)
        else:
            slate.append({"key": "sklearn.mlp", "name": "mlp",
                          "comment": "neural baseline (torch not installed)"})
    return slate


def _render_yaml(data_path: Path, target: str, goal: str, budget: str,
                 info: dict[str, Any], slate: list[dict[str, Any]],
                 run_tag: str) -> str:
    import yaml

    lines: list[str] = [
        "# SURGE workflow spec — generated by `surge init`",
        f"# data: {data_path.name} ({info['n_rows']:,} rows x "
        f"{info['n_cols']} cols) · goal: {goal} · budget: {budget} "
        f"({_BUDGET_NOTES[budget]})",
        "# validate: surge validate <this file> · run: surge run <this file>",
        "",
        f"dataset_path: {data_path.resolve()}",
        f"run_tag: {run_tag}",
        "",
        "# hold-out fractions (train gets the remainder)",
        "test_fraction: 0.2",
        "val_fraction: 0.1",
        "standardize_inputs: true   # z-score inputs (recommended for NNs/GPs)",
        "seed: 42                   # reproducible split + training",
    ]
    if budget == "smoke":
        lines.append("sample_rows: 2000          # smoke budget: subsample "
                     "for a fast first pass")
    lines += [
        "",
        "# every model trains on the same splits -> directly comparable",
        f"# target column inferred as '{target}' — the analyzer treats the",
        "# last numeric column(s) as outputs; override via metadata_overrides",
        "models:",
    ]
    for block in slate:
        comment = block.pop("comment", "")
        text = yaml.safe_dump([block], sort_keys=False,
                              default_flow_style=False)
        text = "\n".join("  " + ln if ln.strip() else ln
                         for ln in text.rstrip().splitlines())
        if comment:
            lines.append(f"  # {comment}")
        lines.append(text)
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="surge init",
        description="Interactive wizard: inspect a dataset and write a "
                    "commented workflow spec.")
    ap.add_argument("--data", help="path to the data file (CSV/Parquet/PKL/"
                                   "HDF5/NetCDF)")
    ap.add_argument("--target", help="target column to predict")
    ap.add_argument("--goal", choices=GOALS, help="what matters most")
    ap.add_argument("--budget", choices=BUDGETS, help="time budget")
    ap.add_argument("--out", default="spec.yaml", help="output spec path")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="non-interactive: accept defaults for anything "
                         "not given via flags")
    args = ap.parse_args(argv)

    interactive = not args.yes and sys.stdin.isatty()

    data = args.data
    if not data:
        if not interactive:
            print("error: --data is required with --yes", file=sys.stderr)
            return 2
        data = _ask("Data file", "data.csv")
    data_path = Path(data)
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        return 2

    print(f"Inspecting {data_path.name} ...")
    try:
        info = _inspect(data_path)
    except Exception as exc:  # noqa: BLE001 - wizard reports, never tracebacks
        print(f"error: could not load data: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    print(f"  {info['n_rows']:,} rows x {info['n_cols']} columns "
          f"({len(info['numeric'])} numeric)")

    candidates = _target_candidates(info)
    target = args.target
    if not target:
        target = (candidates[0] if not interactive else
                  _ask(f"Target column (candidates: {', '.join(candidates)})",
                       candidates[0]))
    if target not in info["df"].columns:
        print(f"error: column '{target}' not in the data", file=sys.stderr)
        return 2

    goal = args.goal or ("accuracy" if not interactive else
                         _ask("Goal — accuracy | speed | uncertainty",
                              "accuracy", GOALS))
    budget = args.budget or ("standard" if not interactive else
                             _ask("Budget — smoke | standard | thorough",
                                  "standard", BUDGETS))

    try:
        from surge.model.registry import MODEL_REGISTRY
        available = set(MODEL_REGISTRY.list_keys()) if hasattr(
            MODEL_REGISTRY, "list_keys") else set(
            getattr(MODEL_REGISTRY, "_registry", {}))
    except Exception:  # noqa: BLE001
        available = {"sklearn.ridge", "sklearn.random_forest", "sklearn.mlp"}

    slate = _model_slate(goal, budget, info["n_rows"], available)
    run_tag = data_path.stem.lower().replace(" ", "_")
    text = _render_yaml(data_path, target, goal, budget, info, slate, run_tag)

    out = Path(args.out)
    if out.exists() and interactive:
        if _ask(f"{out} exists — overwrite? (y/n)", "n") != "y":
            print("aborted")
            return 1
    out.write_text(text)

    from surge.workflow.schema import validate_file
    errors = validate_file(out)
    if errors:  # should not happen — the wizard validates its own output
        print("warning: generated spec has validation issues:",
              file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)

    print(f"\nwrote {out}  ({len(slate)} models, goal={goal}, "
          f"budget={budget})")
    print(f"next:  surge validate {out}\n       surge run {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
