#!/usr/bin/env python3
"""
SURGE benchmark CLI.

Usage examples
--------------
List all benchmarks::

    python -m surge.benchmarks.run --list
    python -m surge.benchmarks.run --list --verbose

Run a benchmark with its default model::

    python -m surge.benchmarks.run --benchmark tabular.diabetes

Run with a specific model from MODEL_REGISTRY::

    python -m surge.benchmarks.run --benchmark tabular.iris --model sklearn.logistic_regression
    python -m surge.benchmarks.run --benchmark tabular.breast_cancer --model sklearn.gradient_boosting_classifier

Run all benchmarks (default models)::

    python -m surge.benchmarks.run --all

Save result JSON::

    python -m surge.benchmarks.run --benchmark tabular.diabetes --output results/diabetes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import benchmark_info, list_benchmarks, run_benchmark


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _fmt_metrics(metrics: dict) -> str:
    """Compact single-line metric string, skipping runtime_s."""
    parts = []
    for k, v in metrics.items():
        if k == "runtime_s":
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


def _print_result_banner(result) -> None:
    """Print a human-readable one-line summary of a BenchmarkResult."""
    status = _STATUS_OK if result.passed else _STATUS_FAIL
    runtime = result.metrics.get("runtime_s", 0.0)
    metrics_str = _fmt_metrics(result.metrics)
    print(
        f"[{status}] {result.benchmark_key:<40}  model={result.model_key}"
    )
    print(f"       {metrics_str}  ({runtime:.2f}s)")


# ANSI-free status labels (works on all terminals).
_STATUS_OK = "PASS"
_STATUS_FAIL = "FAIL"


def _print_list(verbose: bool) -> None:
    keys = list_benchmarks()
    if not verbose:
        for k in keys:
            print(k)
        return

    # Verbose table
    col_widths = (42, 6, 16, 8)
    header = f"{'Key':<{col_widths[0]}}  {'Tier':<{col_widths[1]}}  {'Task':<{col_widths[2]}}  {'Shape':<{col_widths[3]}}  Description"
    print(header)
    print("-" * (len(header) + 20))
    for k in keys:
        info = benchmark_info(k)
        print(
            f"{info['key']:<{col_widths[0]}}  "
            f"{info['tier']:<{col_widths[1]}}  "
            f"{info['task_type']:<{col_widths[2]}}  "
            f"{info['shape']:<{col_widths[3]}}  "
            f"{info['description']}"
        )


def _print_list_models() -> None:
    """Print all models currently in MODEL_REGISTRY."""
    from surge.model.registry import MODEL_REGISTRY

    print("Registered models in MODEL_REGISTRY:")
    for key, cls_name in sorted(MODEL_REGISTRY.list_models().items()):
        print(f"  {key:<45}  {cls_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description="SURGE benchmark runner — train models and report metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--benchmark", "-b", metavar="KEY", help="Benchmark registry key (see --list).")
    ap.add_argument("--model", "-m", metavar="KEY", default=None,
                    help="Model registry key to run (default: per-benchmark default). See --list-models.")
    ap.add_argument("--all", "-a", dest="run_all", action="store_true",
                    help="Run all registered benchmarks with their default models.")
    ap.add_argument("--list", "-l", action="store_true",
                    help="Print registered benchmark keys and exit.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="With --list: show tier, task_type, shape, and description.")
    ap.add_argument("--list-models", action="store_true",
                    help="Print all models available in MODEL_REGISTRY and exit.")
    ap.add_argument("--tier", metavar="TIER",
                    help="With --list or --all: filter by tier (0, 1, …).")
    ap.add_argument("--task-type", metavar="TYPE",
                    help="With --list or --all: filter by task type (regression|classification).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write BenchmarkResult JSON to this file (dirs created).")
    args = ap.parse_args(argv)

    # ── list-models ──────────────────────────────────────────────────────────
    if args.list_models:
        _print_list_models()
        return 0

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        _print_list(verbose=args.verbose)
        return 0

    # ── --all ────────────────────────────────────────────────────────────────
    if args.run_all:
        keys = list_benchmarks(
            tier=args.tier,
            task_type=args.task_type,
        )
        if not keys:
            print("No benchmarks match the specified filters.", file=sys.stderr)
            return 1

        results = []
        any_failed = False
        print(f"Running {len(keys)} benchmark(s) — seed={args.seed}\n")
        for k in keys:
            print(f"  → {k}", end="", flush=True)
            try:
                r = run_benchmark(k, seed=args.seed, model_key=args.model)
                results.append(r)
                _print_result_banner(r)
                if not r.passed:
                    any_failed = True
            except Exception as exc:
                print(f"\n[ERROR] {k}: {exc}", file=sys.stderr)
                any_failed = True

        # Summary table
        n_pass = sum(1 for r in results if r.passed)
        print(f"\n{'-' * 60}")
        print(f"Results: {n_pass}/{len(results)} passed")

        if args.output is not None:
            payload = [r.to_dict() for r in results]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}", file=sys.stderr)

        return 1 if any_failed else 0

    # ── single benchmark ─────────────────────────────────────────────────────
    if not args.benchmark:
        ap.error("provide --benchmark KEY, --all, --list, or --list-models")

    try:
        result = run_benchmark(args.benchmark, seed=args.seed, model_key=args.model)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _print_result_banner(result)

    payload = result.to_dict()
    text = json.dumps(payload, indent=2, default=str)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
