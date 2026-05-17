#!/usr/bin/env python3
"""CLI: ``python -m surge.benchmarks.run --benchmark synthetic.regression_1d``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import list_benchmarks, run_benchmark


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Run SURGE standard CPU benchmarks.")
    ap.add_argument("--benchmark", "-b", help="Registry key (see --list).")
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print registered benchmark keys and exit.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits / models.")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write BenchmarkResult JSON to this path (dirs created).",
    )
    args = ap.parse_args(argv)

    if args.list:
        for k in list_benchmarks():
            print(k)
        return 0

    if not args.benchmark:
        ap.error("provide --benchmark KEY or use --list")

    result = run_benchmark(args.benchmark, seed=args.seed)
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, default=str)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
