"""SURGE command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None  # type: ignore[assignment]


def _viz(args: argparse.Namespace) -> int:
    """Generate inference comparison plots from a run directory."""
    import matplotlib
    matplotlib.use("Agg")

    from surge.viz.run_viz import viz_run

    run_dir = args.run_dir
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    axis_lim = None
    if args.axis_lim:
        parts = args.axis_lim.split(",")
        if len(parts) == 2:
            axis_lim = (float(parts[0].strip()), float(parts[1].strip()))

    result = viz_run(
        run_dir=Path(run_dir),
        output_dir=args.output_dir,
        axis_lim=axis_lim,
        include_hpo=not getattr(args, "no_hpo", False),
        hpo_plot_metric=getattr(args, "hpo_metric", None),
        include_shap=getattr(args, "shap", False),
        shap_split=getattr(args, "shap_split", "val"),
        shap_max_samples=getattr(args, "shap_max_samples", 500),
        shap_output_index=getattr(args, "shap_output_index", None),
        include_datastreamset_eval=getattr(args, "datastreamset_eval", False),
        datastreamset_size=getattr(args, "datastreamset_size", 50000),
        datastreamset_max=getattr(args, "datastreamset_max", 5),
    )
    print("SURGE inference comparison plots")
    print("=" * 50)
    print(f"Run directory: {run_dir}")
    print()
    print("R² scores:")
    print(json.dumps(result["r2"], indent=2))
    print()
    print("Plots saved:")
    for p in result["saved_paths"]:
        print(f"  {p}")
    if "datastreamset_eval" in result and result["datastreamset_eval"]:
        se = result["datastreamset_eval"]
        print()
        print("Datastreamset evaluation saved:")
        print(f"  {se.get('saved_path', 'N/A')}")
        dd = se.get("drift_detection", {})
        if dd:
            print()
            print("Drift detection:")
            print(f"  Policy: {dd.get('policy', 'N/A')}")
            print(f"  Drift warning: {dd.get('drift_warning', False)}")
            if dd.get("datastreamsets_with_drift"):
                print(f"  Datastreamsets with drift: {dd['datastreamsets_with_drift']}")
            if dd.get("continual_learning_recommendation"):
                print(f"  Recommendation: {dd['continual_learning_recommendation']}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    """Load dataset from spec (or path) and display inputs/outputs/stats."""
    import surge  # noqa: F401
    from surge.dataset import SurrogateDataset

    path: Path
    meta_path: Optional[Path] = None

    if args.dataset:
        path = Path(args.dataset)
        cand = path.parent / (path.stem + "_metadata.yaml")
        if not cand.exists():
            cand = path.parent / "per_mode_metadata.yaml"
        if cand.exists():
            meta_path = cand
    else:
        spec_path = Path(args.spec)
        if not spec_path.exists():
            print(f"Spec not found: {spec_path}", file=sys.stderr)
            return 1
        if spec_path.suffix.lower() in (".pkl", ".csv", ".parquet"):
            path = spec_path
            meta_path = spec_path.parent / (spec_path.stem + "_metadata.yaml")
            if not meta_path.exists():
                meta_path = None
        else:
            if yaml is None:
                print("PyYAML required for spec. pip install pyyaml", file=sys.stderr)
                return 1
            with spec_path.open("r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
            path = Path(payload.get("dataset_path", ""))
            if not path.is_absolute():
                # Resolve relative to cwd (project root), then config dir
                path = Path.cwd() / path if (Path.cwd() / path).exists() else spec_path.parent / path
            if not path.exists():
                print(f"Dataset not found: {path}", file=sys.stderr)
                return 1
            mp = payload.get("metadata_path")
            if mp:
                mp_path = Path(mp)
                meta_path = (Path.cwd() / mp_path) if (Path.cwd() / mp_path).exists() else (spec_path.parent / mp_path)
                if not meta_path.exists():
                    meta_path = None

    dataset = SurrogateDataset.from_path(
        path,
        metadata_path=meta_path if meta_path and meta_path.exists() else None,
    )
    if dataset.df is None:
        print("Failed to load dataset", file=sys.stderr)
        return 1

    inc = dataset.input_columns
    outc = dataset.output_columns
    print("=" * 60)
    print("SURGE Dataset Analysis")
    print("=" * 60)
    print(f"Path: {path}")
    print(f"Rows: {len(dataset.df)}")
    print(f"Inputs ({len(inc)}): {inc}")
    if len(outc) > 3:
        print(f"Outputs ({len(outc)}): {outc[0]} .. {outc[-1]}")
    else:
        print(f"Outputs: {outc}")
    print()
    print("Input stats (sample):")
    print(dataset.stats(inc[:6] if len(inc) > 6 else inc).to_string())
    print()
    print("Output stats (first 3):")
    print(dataset.stats(outc[:3]).to_string())
    return 0


def _run(args: argparse.Namespace) -> int:
    """Run surrogate workflow from spec file."""
    if yaml is None:
        print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        return 1

    import surge  # noqa: F401 - ensures package initialization registers adapters
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    with spec_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    spec = SurrogateWorkflowSpec.from_dict(payload)

    if args.run_tag:
        spec.run_tag = args.run_tag
    if args.output_dir:
        spec.output_dir = str(args.output_dir)

    summary = run_surrogate_workflow(spec)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="surge",
        description="SURGE - Surrogate Unified Robust Generation Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # surge run
    run_parser = subparsers.add_parser("run", help="Run surrogate workflow from spec YAML")
    run_parser.add_argument(
        "spec",
        type=Path,
        help="Path to workflow spec YAML (e.g. configs/xgc_target_set1.yaml)",
    )
    run_parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Override run tag in spec",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory in spec",
    )
    run_parser.set_defaults(func=_run)

    # surge analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Load and display dataset (inputs, outputs, stats)",
    )
    analyze_parser.add_argument(
        "spec",
        type=Path,
        help="Path to workflow spec YAML (or dataset .pkl for per-mode)",
    )
    analyze_parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset path (pkl/csv)",
    )
    analyze_parser.set_defaults(func=_analyze)

    # surge viz
    viz_parser = subparsers.add_parser(
        "viz",
        help="Generate inference comparison plots from a run directory",
    )
    viz_parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path("runs/xgc_target_set1_v2"),
        help="Path to SURGE workflow run directory (default: runs/xgc_target_set1_v2)",
    )
    viz_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: run_dir/plots)",
    )
    viz_parser.add_argument(
        "--axis-lim",
        type=str,
        default=None,
        help="Axis limits as 'min,max' e.g. '-0.5,1.5'",
    )
    viz_parser.add_argument(
        "--no-hpo",
        action="store_true",
        help="Skip HPO convergence plots",
    )
    viz_parser.add_argument(
        "--hpo-metric",
        type=str,
        choices=["r2", "rmse"],
        default=None,
        help="HPO y-axis metric: r2 or rmse (default: auto-detect, prefers r2 when available)",
    )
    viz_parser.add_argument(
        "--shap",
        action="store_true",
        help="Run SHAP analysis for tree models (saves to plots/shap_{model}/)",
    )
    viz_parser.add_argument(
        "--shap-split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Split for SHAP (default: val)",
    )
    viz_parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=500,
        help="Max samples for SHAP (default: 500)",
    )
    viz_parser.add_argument(
        "--shap-output-index",
        type=int,
        default=None,
        help="Output index for SHAP (0-based). For kinetic-code target use 1 (default: auto for stacked)",
    )
    viz_parser.add_argument(
        "--datastreamset-eval",
        action="store_true",
        help="Evaluate models on held-out dataset datastreamsets (kinetic-code only)",
    )
    viz_parser.add_argument(
        "--datastreamset-size",
        type=int,
        default=50000,
        help="Rows per datastreamset for --datastreamset-eval (default: 50000)",
    )
    viz_parser.add_argument(
        "--datastreamset-max",
        type=int,
        default=10,
        help="Max datastreamsets for --datastreamset-eval (default: 10)",
    )
    viz_parser.set_defaults(func=_viz)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Setup logging for run command
    if args.command == "run":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    return args.func(args)


def cli() -> None:
    """Entry point for console_scripts."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
