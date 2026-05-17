#!/usr/bin/env python3
"""
SURGE benchmark CLI.

Results are **automatically saved** to ``benchmark_reports/<key>/<timestamp>/result.json``
after every run (pass ``--no-save`` to skip).  Pass ``--mlflow`` to additionally
log metrics and tags to an MLflow tracking server.

Usage examples
--------------
List all benchmarks::

    python -m surge.benchmarks.run --list
    python -m surge.benchmarks.run --list --verbose
    python -m surge.benchmarks.run --list --task-type classification

List available models::

    python -m surge.benchmarks.run --list-models

Run a benchmark with its default model::

    python -m surge.benchmarks.run --benchmark tabular.diabetes

Run with a specific model from MODEL_REGISTRY::

    python -m surge.benchmarks.run --benchmark tabular.iris --model sklearn.logistic_regression

Run all benchmarks (default models)::

    python -m surge.benchmarks.run --all
    python -m surge.benchmarks.run --all --tier 0
    python -m surge.benchmarks.run --all --task-type classification

Track with MLflow (local mlruns/ by default)::

    python -m surge.benchmarks.run --benchmark tabular.iris --mlflow
    python -m surge.benchmarks.run --all --mlflow --mlflow-experiment my_project
    python -m surge.benchmarks.run --all --mlflow --mlflow-tracking-uri http://localhost:5000

Run a leaderboard: all compatible models vs a benchmark or group of benchmarks::

    python -m surge.benchmarks.run --leaderboard --benchmark tabular.iris
    python -m surge.benchmarks.run --leaderboard --tier 1 --task-type classification
    python -m surge.benchmarks.run --leaderboard --all-benchmarks --mlflow

Compare specific models on a single benchmark::

    python -m surge.benchmarks.run --benchmark tabular.diabetes \\
        --compare-models sklearn.random_forest,sklearn.mlp,pytorch.mlp

Hyperparameter optimisation (HPO) via Optuna::

    python -m surge.benchmarks.run --benchmark tabular.california_housing \\
        --model xgboost.xgbregressor --hpo --hpo-trials 40
    python -m surge.benchmarks.run --benchmark tabular.iris \\
        --model xgboost.xgbclassifier --hpo --hpo-trials 30 --mlflow
    python -m surge.benchmarks.run --benchmark tabular.diabetes \\
        --model pytorch.residual_mlp --hpo --hpo-trials 20 --hpo-metric test_r2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .leaderboard import (
    _default_models_for,
    format_leaderboard_table,
    log_leaderboard_to_mlflow,
    print_leaderboard,
    run_leaderboard,
)
from .registry import benchmark_info, list_benchmarks, run_benchmark


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_metrics(metrics: dict) -> str:
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
    runtime = result.metrics.get("runtime_s", 0.0)
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.benchmark_key:<40}  model={result.model_key}")
    print(f"       {_fmt_metrics(result.metrics)}  ({runtime:.2f}s)")


def _print_list(verbose: bool, tier: str | None = None, task_type: str | None = None) -> None:
    keys = list_benchmarks(tier=tier, task_type=task_type)
    if not verbose:
        for k in keys:
            print(k)
        return
    col = (42, 6, 16, 8)
    header = (
        f"{'Key':<{col[0]}}  {'Tier':<{col[1]}}  {'Task':<{col[2]}}  {'Shape':<{col[3]}}  Description"
    )
    print(header)
    print("-" * (len(header) + 20))
    for k in keys:
        info = benchmark_info(k)
        print(
            f"{info['key']:<{col[0]}}  "
            f"{info['tier']:<{col[1]}}  "
            f"{info['task_type']:<{col[2]}}  "
            f"{info['shape']:<{col[3]}}  "
            f"{info['description']}"
        )


def _print_list_models() -> None:
    from surge.model.registry import MODEL_REGISTRY

    print("Registered models in MODEL_REGISTRY:")
    for key, cls_name in sorted(MODEL_REGISTRY.list_models().items()):
        print(f"  {key:<45}  {cls_name}")


# ---------------------------------------------------------------------------
# Save + MLflow helpers
# ---------------------------------------------------------------------------


def _persist(result, *, save_root: Path | None, no_save: bool) -> Path | None:
    """Save result JSON; returns path or None."""
    if no_save or save_root is None:
        return None
    try:
        path = result.save(root=save_root)
        print(f"       Saved  → {path}", file=sys.stderr)
        return path
    except Exception as exc:
        print(f"       [warn] Could not save result: {exc}", file=sys.stderr)
        return None


def _save_leaderboard_plots(results_by_benchmark: dict, *, save_dir: Path) -> None:
    """Save PNG/PDF bar charts and metric tables for a leaderboard run."""
    try:
        from surge.viz.benchmark import plot_benchmark_leaderboard, plot_metric_table, plot_multi_benchmark_dashboard
    except ImportError:
        print("       [warn] matplotlib not available — skipping plots", file=sys.stderr)
        return

    plots_dir = save_dir / ".plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for bk, rl in results_by_benchmark.items():
        if not rl:
            continue
        safe = bk.replace(".", "_")
        task = rl[0].task_type
        primary_metric = "test_accuracy" if task == "classification" else "test_r2"
        if not any(primary_metric in r.metrics for r in rl):
            primary_metric = next(
                (k for r in rl for k in r.metrics if isinstance(r.metrics[k], float)), None
            )
        if primary_metric:
            try:
                plot_benchmark_leaderboard(
                    rl, metric=primary_metric,
                    save_path=plots_dir / f"{safe}_leaderboard.png",
                )
                print(f"       Plot  → {plots_dir / f'{safe}_leaderboard.png'}", file=sys.stderr)
            except Exception as exc:
                print(f"       [warn] leaderboard plot failed for {bk}: {exc}", file=sys.stderr)
        try:
            plot_metric_table(rl, save_path=plots_dir / f"{safe}_table.png")
            print(f"       Plot  → {plots_dir / f'{safe}_table.png'}", file=sys.stderr)
        except Exception as exc:
            print(f"       [warn] metric table failed for {bk}: {exc}", file=sys.stderr)

    # Multi-benchmark dashboard (all benchmarks in one figure).
    if len(results_by_benchmark) > 1:
        try:
            plot_multi_benchmark_dashboard(
                results_by_benchmark,
                save_path=plots_dir / "dashboard.png",
            )
            print(f"       Plot  → {plots_dir / 'dashboard.png'}", file=sys.stderr)
        except Exception as exc:
            print(f"       [warn] dashboard plot failed: {exc}", file=sys.stderr)


def _mlflow_log(result, *, result_path, experiment: str, tracking_uri: str | None) -> None:
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE, log_benchmark_result

    if not MLFLOW_AVAILABLE:
        print(
            "       [warn] MLflow not installed. pip install 'surge-ml[mlflow]'",
            file=sys.stderr,
        )
        return
    ok = log_benchmark_result(
        result,
        experiment_name=experiment,
        tracking_uri=tracking_uri,
        result_path=result_path,
    )
    if ok:
        print(f"       MLflow → experiment={experiment!r}", file=sys.stderr)


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
    # ── selection ────────────────────────────────────────────────────────────
    ap.add_argument("--benchmark", "-b", metavar="KEY",
                    help="Benchmark registry key (see --list).")
    ap.add_argument("--model", "-m", metavar="KEY", default=None,
                    help="Model registry key (default: per-benchmark default). See --list-models.")
    ap.add_argument("--all", "-a", dest="run_all", action="store_true",
                    help="Run all registered benchmarks with their default models.")
    ap.add_argument("--tier", metavar="TIER",
                    help="Filter by tier (0, 1, …). Works with --list, --all, and --leaderboard.")
    ap.add_argument("--task-type", metavar="TYPE",
                    help="Filter by task type (regression|classification). Works with --list, --all, --leaderboard.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    # ── leaderboard ──────────────────────────────────────────────────────────
    ap.add_argument("--leaderboard", action="store_true",
                    help=(
                        "Run all compatible models against the selected benchmark(s) and "
                        "print a per-benchmark comparison table. Use with --benchmark, "
                        "--tier, --task-type, or --all-benchmarks."
                    ))
    ap.add_argument("--all-benchmarks", action="store_true",
                    help="With --leaderboard: run all registered benchmarks.")
    ap.add_argument("--compare-models", metavar="KEY1,KEY2,...", default=None,
                    help=(
                        "Comma-separated list of model keys to compare on --benchmark. "
                        "Overrides the default compatible-model set."
                    ))
    ap.add_argument("--plot", action="store_true",
                    help=(
                        "With --leaderboard: save PNG/PDF leaderboard bar charts and "
                        "metric tables to <save-dir>/.plots/. Requires matplotlib."
                    ))
    # ── HPO ──────────────────────────────────────────────────────────────────
    ap.add_argument("--hpo", action="store_true",
                    help=(
                        "Run Optuna hyperparameter search for --benchmark / --model. "
                        "Requires optuna (pip install optuna)."
                    ))
    ap.add_argument("--hpo-trials", type=int, default=20, metavar="N",
                    help="Number of Optuna trials. Default: 20.")
    ap.add_argument("--hpo-metric", default=None, metavar="METRIC",
                    help=(
                        "Metric to optimise (e.g. test_r2, test_accuracy, test_rmse). "
                        "Defaults to the primary metric for the benchmark task type."
                    ))
    ap.add_argument("--hpo-epochs-cap", type=int, default=50, metavar="N",
                    help="Cap n_epochs for PyTorch models during HPO trials. Default: 50.")
    ap.add_argument("--list-hpo-models", action="store_true",
                    help="Print model keys that have a registered HPO search space and exit.")
    # ── listing ──────────────────────────────────────────────────────────────
    ap.add_argument("--list", "-l", action="store_true",
                    help="Print registered benchmark keys and exit.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="With --list: show tier, task_type, shape, and description.")
    ap.add_argument("--list-models", action="store_true",
                    help="Print all models available in MODEL_REGISTRY and exit.")
    # ── persistence ──────────────────────────────────────────────────────────
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Also write result JSON to this explicit path (legacy; auto-save still runs).")
    ap.add_argument("--save-dir", type=Path, default=Path("benchmark_reports"),
                    metavar="DIR",
                    help="Root directory for auto-saved results. Default: benchmark_reports/")
    ap.add_argument("--no-save", action="store_true",
                    help="Disable automatic saving to benchmark_reports/.")
    # ── MLflow ───────────────────────────────────────────────────────────────
    ap.add_argument("--mlflow", action="store_true",
                    help="Log results to MLflow (requires mlflow package).")
    ap.add_argument("--mlflow-experiment", default="surge_benchmarks", metavar="NAME",
                    help="MLflow experiment name. Default: surge_benchmarks.")
    ap.add_argument("--mlflow-tracking-uri", default=None, metavar="URI",
                    help="MLflow tracking URI. Default: local ./mlruns.")
    args = ap.parse_args(argv)

    # ── list-hpo-models ──────────────────────────────────────────────────────
    if args.list_hpo_models:
        from .hpo import list_hpo_models
        print("\nModels with HPO search spaces:")
        for mk in list_hpo_models():
            print(f"  {mk}")
        return 0

    # ── list-models ──────────────────────────────────────────────────────────
    if args.list_models:
        _print_list_models()
        return 0

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        _print_list(verbose=args.verbose, tier=args.tier, task_type=args.task_type)
        return 0

    # ── --hpo ────────────────────────────────────────────────────────────────
    if args.hpo:
        try:
            import optuna as _optuna  # noqa: F401
        except ImportError:
            print(
                "ERROR: optuna is required for --hpo.  pip install optuna",
                file=sys.stderr,
            )
            return 1

        if not args.benchmark:
            print("ERROR: --hpo requires --benchmark KEY", file=sys.stderr)
            return 1
        if not args.model:
            print("ERROR: --hpo requires --model KEY", file=sys.stderr)
            return 1

        from .hpo import print_hpo_summary, run_benchmark_hpo

        print(
            f"\nHPO: {args.benchmark}  /  {args.model}"
            f"  ({args.hpo_trials} trials)",
            file=sys.stderr,
        )

        result, best_params = run_benchmark_hpo(
            args.benchmark,
            args.model,
            n_trials=args.hpo_trials,
            seed=args.seed,
            metric=args.hpo_metric,
            n_epochs_cap=args.hpo_epochs_cap,
            save_root=None if args.no_save else args.save_dir,
            verbose=args.verbose,
            mlflow_experiment=args.mlflow_experiment if args.mlflow else None,
            mlflow_tracking_uri=args.mlflow_tracking_uri if args.mlflow else None,
        )

        if result is None:
            print("HPO failed — no successful trials.", file=sys.stderr)
            return 1

        metric_used = result.extra.get("hpo_metric", args.hpo_metric or "?")
        print_hpo_summary(
            result,
            best_params,
            benchmark_key=args.benchmark,
            model_key=args.model,
            n_trials=args.hpo_trials,
            metric=metric_used,
        )

        if args.output:
            import json
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result.to_dict(), indent=2, default=str) + "\n",
                encoding="utf-8",
            )

        return 0 if result.passed else 1

    # ── --leaderboard ────────────────────────────────────────────────────────
    if args.leaderboard:
        # Determine which benchmarks to run.
        if args.all_benchmarks or (not args.benchmark):
            bm_keys = list_benchmarks(tier=args.tier, task_type=args.task_type)
        else:
            bm_keys = [args.benchmark]

        if not bm_keys:
            print("No benchmarks match the specified filters.", file=sys.stderr)
            return 1

        # Determine which models to use.
        custom_models = (
            [m.strip() for m in args.compare_models.split(",") if m.strip()]
            if args.compare_models
            else None
        )

        print(
            f"\nLeaderboard: {len(bm_keys)} benchmark(s) — seed={args.seed}\n"
        )

        lb_results = run_leaderboard(
            bm_keys,
            model_keys=custom_models,
            seed=args.seed,
            save_root=None if args.no_save else args.save_dir,
        )

        # Print per-benchmark tables.
        print_leaderboard(lb_results)

        # Matplotlib plots.
        if args.plot:
            _save_leaderboard_plots(lb_results, save_dir=args.save_dir)

        # MLflow.
        if args.mlflow:
            log_leaderboard_to_mlflow(
                lb_results,
                experiment_name=args.mlflow_experiment,
                tracking_uri=args.mlflow_tracking_uri,
                save_tables=True,
            )
            print(f"\nMLflow → experiment={args.mlflow_experiment!r}", file=sys.stderr)

        # Legacy --output: write all results as flat JSON list.
        if args.output is not None:
            import json
            all_res = [r.to_dict() for rl in lb_results.values() for r in rl]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(all_res, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"Wrote {args.output}", file=sys.stderr)

        any_failed = any(not r.passed for rl in lb_results.values() for r in rl)
        return 1 if any_failed else 0

    # ── --all ────────────────────────────────────────────────────────────────
    if args.run_all:
        keys = list_benchmarks(tier=args.tier, task_type=args.task_type)
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
                rp = _persist(r, save_root=args.save_dir, no_save=args.no_save)
                if args.mlflow:
                    _mlflow_log(r, result_path=rp,
                                experiment=args.mlflow_experiment,
                                tracking_uri=args.mlflow_tracking_uri)
                if not r.passed:
                    any_failed = True
            except Exception as exc:
                print(f"\n[ERROR] {k}: {exc}", file=sys.stderr)
                any_failed = True

        n_pass = sum(1 for r in results if r.passed)
        print(f"\n{'-' * 60}")
        print(f"Results: {n_pass}/{len(results)} passed")
        if not args.no_save:
            print(f"Reports: {args.save_dir}/")

        # Legacy --output: write list JSON
        if args.output is not None:
            import json
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps([r.to_dict() for r in results], indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {args.output}", file=sys.stderr)

        return 1 if any_failed else 0

    # ── single benchmark (optionally --compare-models) ───────────────────────
    if not args.benchmark:
        ap.error("provide --benchmark KEY, --all, --leaderboard, --list, or --list-models")

    # If --compare-models is given without --leaderboard, run as a mini-leaderboard.
    if args.compare_models:
        models = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        lb_results = run_leaderboard(
            [args.benchmark],
            model_keys=models,
            seed=args.seed,
            save_root=None if args.no_save else args.save_dir,
        )
        print_leaderboard(lb_results)
        if args.mlflow:
            log_leaderboard_to_mlflow(
                lb_results,
                experiment_name=args.mlflow_experiment,
                tracking_uri=args.mlflow_tracking_uri,
            )
            print(f"MLflow → experiment={args.mlflow_experiment!r}", file=sys.stderr)
        any_failed = any(not r.passed for rl in lb_results.values() for r in rl)
        return 1 if any_failed else 0

    try:
        result = run_benchmark(args.benchmark, seed=args.seed, model_key=args.model)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _print_result_banner(result)
    rp = _persist(result, save_root=args.save_dir, no_save=args.no_save)

    if args.mlflow:
        _mlflow_log(result, result_path=rp,
                    experiment=args.mlflow_experiment,
                    tracking_uri=args.mlflow_tracking_uri)

    # Legacy --output
    if args.output is not None:
        import json
        text = json.dumps(result.to_dict(), indent=2, default=str)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
