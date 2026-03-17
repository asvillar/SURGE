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
