#!/usr/bin/env python3
"""Run a SURGE workflow from a YAML spec file.

This is the generic CLI wrapper referenced by docs/BUILD_YOUR_OWN_SURROGATE.md.
It loads a :class:`~surge.workflow.spec.SurrogateWorkflowSpec` and calls
:func:`~surge.workflow.run.run_surrogate_workflow`.

Examples
--------
    # Bundled tutorial config (create sample data first — see custom_dataset_tutorial.py)
    python examples/run_workflow.py \\
        --spec examples/configs/custom_dataset_tutorial.yaml

    # Override run tag or output directory
    python examples/run_workflow.py \\
        --spec examples/configs/custom_dataset_tutorial.yaml \\
        --run-tag my_experiment_v2 \\
        --output-dir /tmp/surge_runs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import surge  # noqa: F401,E402 — register model adapters
from surge.workflow.run import run_surrogate_workflow  # noqa: E402
from surge.workflow.spec import SurrogateWorkflowSpec  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a SURGE surrogate workflow from a YAML spec.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to the workflow YAML (SurrogateWorkflowSpec).",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional override for spec.run_tag.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for spec.output_dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    spec_path = args.spec.resolve()
    if not spec_path.is_file():
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec = SurrogateWorkflowSpec.from_dict(payload)
    if args.run_tag:
        spec.run_tag = args.run_tag
    if args.output_dir:
        spec.output_dir = str(args.output_dir.resolve())

    invocation = {"spec_path": str(spec_path)}
    summary = run_surrogate_workflow(spec, invocation=invocation)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
