"""SURGE command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
