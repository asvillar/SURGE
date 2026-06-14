#!/usr/bin/env python3
"""
SURGE benchmark runner.

Results are automatically saved to ``benchmark_reports/<key>/<timestamp>/result.json``
after every run (pass ``--no-save`` to skip).  Pass ``--mlflow`` to additionally
log metrics and tags to an MLflow tracking server.

Quick-start
-----------
List all benchmarks (grouped by category)::

    surge run --list
    surge run --list --category plasma
    surge run --list --category tabular

Run a single benchmark with its default model::

    surge run -b cifar10
    surge run -b tabular.diabetes

Run a benchmark with a specific model::

    surge run -b iris -m sklearn.random_forest
    surge run -b california_housing -m pytorch.ft_transformer

Run all models against one or more benchmarks (leaderboard)::

    surge run -b cifar10 -m all
    surge run -b cifar10 mnist -m all
    surge run -b qlknn constellaration -m all --seeds 3

Run all benchmarks in a category::

    surge run --category plasma -m all
    surge run --category tabular --task-type regression -m all

Compare a specific subset of models::

    surge run -b diabetes --compare-models sklearn.ridge,pytorch.mlp,pytorch.ft_transformer

HPO::

    surge run -b california_housing -m xgboost.xgbregressor --hpo --hpo-trials 40

MLflow tracking::

    surge run -b iris -m all --mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .leaderboard import (
    log_leaderboard_to_mlflow,
    print_leaderboard,
    run_leaderboard,
)
from .registry import (
    benchmark_info,
    list_benchmarks,
    list_categories,
    resolve_benchmark_key,
    run_benchmark,
)


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


def _print_list(
    verbose: bool,
    category: str | None = None,
    task_type: str | None = None,
    include_smoke: bool = False,
    tier: str | None = None,
) -> None:
    keys = list_benchmarks(
        category=category, task_type=task_type, include_smoke=include_smoke, tier=tier
    )
    if not keys:
        print("No benchmarks match the specified filters.")
        return

    if not verbose:
        # Group by category for readability.
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            info = benchmark_info(k)
            groups[info["category"]].append(k)
        for cat in sorted(groups):
            print(f"\n  [{cat}]")
            for k in groups[cat]:
                print(f"    {k}")
        print()
        return

    # Verbose table.
    col = (42, 14, 16, 22)
    header = (
        f"  {'Key':<{col[0]}}  {'Category':<{col[1]}}  {'Task':<{col[2]}}  "
        f"{'Shape':<{col[3]}}  Description"
    )
    print(header)
    print("  " + "-" * (sum(col) + 12))
    prev_cat = None
    for k in keys:
        info = benchmark_info(k)
        if info["category"] != prev_cat:
            print()
            prev_cat = info["category"]
        print(
            f"  {info['key']:<{col[0]}}  "
            f"{info['category']:<{col[1]}}  "
            f"{info['task_type']:<{col[2]}}  "
            f"{info['shape']:<{col[3]}}  "
            f"{info['description']}"
        )
    print()


def _print_list_models() -> None:
    from surge.model.registry import MODEL_REGISTRY

    print("\nRegistered models in MODEL_REGISTRY:")
    by_backend: dict[str, list[str]] = {}
    for key in sorted(MODEL_REGISTRY.list_models()):
        backend = key.split(".")[0]
        by_backend.setdefault(backend, []).append(key)
    for backend in sorted(by_backend):
        print(f"\n  [{backend}]")
        for k in by_backend[backend]:
            print(f"    {k}")
    print()


# ---------------------------------------------------------------------------
# Save + MLflow helpers
# ---------------------------------------------------------------------------


def _persist(result, *, save_root: Path | None, no_save: bool) -> Path | None:
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
    try:
        from surge.viz.benchmark import (
            plot_benchmark_leaderboard,
            plot_metric_table,
            plot_multi_benchmark_dashboard,
        )
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
                    rl,
                    metric=primary_metric,
                    save_path=plots_dir / f"{safe}_leaderboard.png",
                )
            except Exception as exc:
                print(f"       [warn] leaderboard plot failed for {bk}: {exc}", file=sys.stderr)
        try:
            plot_metric_table(rl, save_path=plots_dir / f"{safe}_table.png")
        except Exception as exc:
            print(f"       [warn] metric table failed for {bk}: {exc}", file=sys.stderr)

    if len(results_by_benchmark) > 1:
        try:
            plot_multi_benchmark_dashboard(
                results_by_benchmark,
                save_path=plots_dir / "dashboard.png",
            )
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
# Argument resolution helpers
# ---------------------------------------------------------------------------


def _resolve_benchmarks(keys: list[str], *, include_smoke: bool = False) -> list[str]:
    """Resolve a list of keys/short-aliases to canonical registry keys."""
    resolved = []
    for k in keys:
        try:
            resolved.append(resolve_benchmark_key(k))
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
    return resolved


def _model_is_all(model_arg: str | None) -> bool:
    return model_arg is not None and model_arg.strip().lower() in ("all", "*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        prog="surge run",
        description="SURGE benchmark runner — train models and report metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── selection ─────────────────────────────────────────────────────────────
    ap.add_argument(
        "--benchmark", "-b", metavar="KEY", nargs="+",
        help=(
            "One or more benchmark keys (full or short alias). "
            "Examples: -b cifar10  -b tabular.iris diabetes  -b qlknn constellaration"
        ),
    )
    ap.add_argument(
        "--model", "-m", metavar="KEY",
        help=(
            "Model registry key, or 'all' to run all compatible models "
            "(equivalent to --leaderboard).  See --list-models."
        ),
    )
    ap.add_argument("--all", "-a", dest="run_all", action="store_true",
                    help="Run all registered benchmarks (excluding smoke tests).")
    ap.add_argument(
        "--category", "-c", metavar="CAT",
        help=(
            f"Filter benchmarks by category. "
            f"Choices: {', '.join(list_categories())}. "
            f"Use 'smoke' to include inline fixtures."
        ),
    )
    ap.add_argument("--task-type", metavar="TYPE",
                    help="Filter by task type: regression | classification.")
    ap.add_argument("--smoke", action="store_true",
                    help="Include synthetic smoke-test benchmarks in --list / --all.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    ap.add_argument(
        "--seeds", type=int, default=1, metavar="N",
        help="Number of evaluation seeds (leaderboard). Reports mean ± std. Default: 1.",
    )

    # ── leaderboard ───────────────────────────────────────────────────────────
    ap.add_argument(
        "--leaderboard", action="store_true",
        help="Run all compatible models vs the selected benchmark(s). "
             "Equivalent to passing -m all.",
    )
    ap.add_argument("--all-benchmarks", action="store_true",
                    help="With --leaderboard: run all registered benchmarks.")
    ap.add_argument(
        "--compare-models", metavar="KEY1,KEY2,...",
        help="Comma-separated model keys to compare. Overrides --model.",
    )
    ap.add_argument("--plot", action="store_true",
                    help="Save PNG leaderboard charts to <save-dir>/.plots/. Requires matplotlib.")

    # ── HPO ───────────────────────────────────────────────────────────────────
    ap.add_argument("--hpo", action="store_true",
                    help="Run Optuna HPO for --benchmark / --model. Requires optuna.")
    ap.add_argument("--hpo-trials", type=int, default=20, metavar="N",
                    help="Number of Optuna trials. Default: 20.")
    ap.add_argument("--hpo-metric", default=None, metavar="METRIC",
                    help="Metric to optimise (e.g. test_r2, test_accuracy).")
    ap.add_argument("--hpo-epochs-cap", type=int, default=50, metavar="N",
                    help="Cap n_epochs for PyTorch models during HPO. Default: 50.")
    ap.add_argument("--list-hpo-models", action="store_true",
                    help="Print model keys with a registered HPO search space and exit.")
    ap.add_argument("--use-hpo-cache", action="store_true",
                    help="Reload cached best hyperparameters from a previous --hpo run.")

    # ── listing ───────────────────────────────────────────────────────────────
    ap.add_argument("--list", "-l", action="store_true",
                    help="Print registered benchmark keys grouped by category and exit.")
    ap.add_argument("--list-models", action="store_true",
                    help="Print all models available in MODEL_REGISTRY and exit.")
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="With --list: show category, task_type, shape, description. "
             "With --benchmark: show live tqdm training progress bar.",
    )
    ap.add_argument("--train-log-file", metavar="PATH", default=None,
                    help="Stream per-epoch loss records to this JSONL file during training.")

    # ── persistence ───────────────────────────────────────────────────────────
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write result JSON to this path.")
    ap.add_argument("--save-dir", type=Path, default=Path("benchmark_reports"), metavar="DIR",
                    help="Root directory for auto-saved results. Default: benchmark_reports/")
    ap.add_argument("--no-save", action="store_true",
                    help="Disable automatic result saving.")

    # ── MLflow ────────────────────────────────────────────────────────────────
    ap.add_argument("--mlflow", action="store_true",
                    help="Log results to MLflow (requires mlflow package).")
    ap.add_argument("--mlflow-experiment", default="surge_benchmarks", metavar="NAME",
                    help="MLflow experiment name. Default: surge_benchmarks.")
    ap.add_argument("--mlflow-tracking-uri", default=None, metavar="URI",
                    help="MLflow tracking URI. Default: local ./mlruns.")

    # ── legacy tier filter (kept for back-compat, maps to --category) ─────────
    ap.add_argument("--tier", metavar="TIER", help=argparse.SUPPRESS)

    args = ap.parse_args(argv)

    # --tier accepted legacy category names historically; numeric tiers stay
    # numeric and are passed to the registry.
    if args.tier and not args.category and not str(args.tier).isdigit():
        args.category = args.tier
    _include_smoke = args.smoke or str(args.tier) == "0"

    # ── list-hpo-models ───────────────────────────────────────────────────────
    if args.list_hpo_models:
        from .hpo import list_hpo_models
        print("\nModels with HPO search spaces:")
        for mk in list_hpo_models():
            print(f"  {mk}")
        return 0

    # ── list-models ───────────────────────────────────────────────────────────
    if args.list_models:
        _print_list_models()
        return 0

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        _print_list(
            verbose=args.verbose,
            category=args.category,
            task_type=args.task_type,
            include_smoke=_include_smoke,
            tier=args.tier if args.tier and str(args.tier).isdigit() else None,
        )
        return 0

    # ── --hpo ─────────────────────────────────────────────────────────────────
    if args.hpo:
        try:
            import optuna as _optuna  # noqa: F401
        except ImportError:
            print("ERROR: optuna is required for --hpo.  pip install optuna", file=sys.stderr)
            return 1
        if not args.benchmark:
            print("ERROR: --hpo requires --benchmark KEY", file=sys.stderr)
            return 1
        if not args.model:
            print("ERROR: --hpo requires --model KEY", file=sys.stderr)
            return 1
        hpo_bm_key = _resolve_benchmarks(
            [args.benchmark[0] if isinstance(args.benchmark, list) else args.benchmark]
        )[0]
        from .hpo import print_hpo_summary, run_benchmark_hpo

        print(f"\nHPO: {hpo_bm_key}  /  {args.model}  ({args.hpo_trials} trials)", file=sys.stderr)
        result, best_params = run_benchmark_hpo(
            hpo_bm_key, args.model,
            n_trials=args.hpo_trials, seed=args.seed,
            metric=args.hpo_metric, n_epochs_cap=args.hpo_epochs_cap,
            save_root=None if args.no_save else args.save_dir,
            verbose=args.verbose,
            mlflow_experiment=args.mlflow_experiment if args.mlflow else None,
            mlflow_tracking_uri=args.mlflow_tracking_uri if args.mlflow else None,
        )
        if result is None:
            print("HPO failed — no successful trials.", file=sys.stderr)
            return 1
        print_hpo_summary(
            result, best_params,
            benchmark_key=hpo_bm_key, model_key=args.model,
            n_trials=args.hpo_trials,
            metric=result.extra.get("hpo_metric", args.hpo_metric or "?"),
        )
        if args.output:
            import json
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
            )
        return 0 if result.passed else 1

    # ── Decide whether we're in leaderboard mode ───────────────────────────────
    # Leaderboard mode = --leaderboard flag  OR  -m all  OR  --compare-models
    use_leaderboard = (
        args.leaderboard
        or args.all_benchmarks
        or _model_is_all(args.model)
        or bool(args.compare_models)
    )

    # ── --leaderboard / -m all ─────────────────────────────────────────────────
    if use_leaderboard:
        # Determine which benchmarks to run.
        if args.all_benchmarks or (not args.benchmark and not args.category):
            bm_keys = list_benchmarks(
                category=args.category,
                task_type=args.task_type,
                include_smoke=_include_smoke,
                tier=args.tier if args.tier and str(args.tier).isdigit() else None,
            )
        elif args.category and not args.benchmark:
            bm_keys = list_benchmarks(
                category=args.category,
                task_type=args.task_type,
                include_smoke=_include_smoke,
                tier=args.tier if args.tier and str(args.tier).isdigit() else None,
            )
        else:
            raw = args.benchmark or []
            bm_keys = _resolve_benchmarks(raw)

        if not bm_keys:
            print("No benchmarks match the specified filters.", file=sys.stderr)
            return 1

        custom_models = (
            [m.strip() for m in args.compare_models.split(",") if m.strip()]
            if args.compare_models
            else None
        )

        print(f"\nLeaderboard: {len(bm_keys)} benchmark(s) — seed={args.seed}\n")

        lb_results = run_leaderboard(
            bm_keys,
            model_keys=custom_models,
            seed=args.seed,
            n_seeds=args.seeds,
            save_root=None if args.no_save else args.save_dir,
            use_hpo_cache=args.use_hpo_cache,
        )

        print_leaderboard(lb_results)

        if args.plot:
            _save_leaderboard_plots(lb_results, save_dir=args.save_dir)

        if args.mlflow:
            log_leaderboard_to_mlflow(
                lb_results,
                experiment_name=args.mlflow_experiment,
                tracking_uri=args.mlflow_tracking_uri,
                save_tables=True,
            )
            print(f"\nMLflow → experiment={args.mlflow_experiment!r}", file=sys.stderr)

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

    # ── --all (default model, no leaderboard) ─────────────────────────────────
    if args.run_all or (args.category and not args.benchmark):
        keys = list_benchmarks(
            category=args.category,
            task_type=args.task_type,
            include_smoke=_include_smoke,
            tier=args.tier if args.tier and str(args.tier).isdigit() else None,
        )
        if not keys:
            print("No benchmarks match the specified filters.", file=sys.stderr)
            return 1
        results = []
        any_failed = False
        print(f"Running {len(keys)} benchmark(s) — seed={args.seed}\n")
        for k in keys:
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
        return 1 if any_failed else 0

    # ── single / multi benchmark with explicit model ───────────────────────────
    if not args.benchmark:
        ap.error(
            "provide --benchmark/-b KEY [KEY...], --all/-a, "
            "--category/-c CAT, --leaderboard, --list, or --list-models"
        )

    bm_key_list = _resolve_benchmarks(
        args.benchmark if isinstance(args.benchmark, list) else [args.benchmark]
    )

    # Multiple benchmarks without -m all → run sequentially with default model.
    if len(bm_key_list) > 1:
        results = []
        any_failed = False
        for bk in bm_key_list:
            try:
                r = run_benchmark(bk, seed=args.seed, model_key=args.model)
                _print_result_banner(r)
                _persist(r, save_root=args.save_dir, no_save=args.no_save)
                results.append(r)
                if not r.passed:
                    any_failed = True
            except Exception as exc:
                print(f"Error running {bk}: {exc}", file=sys.stderr)
                any_failed = True
        return 1 if any_failed else 0

    single_bm_key = bm_key_list[0]

    # Build monitoring kwargs if requested.
    _monitor_kwargs: dict = {}
    if args.verbose:
        _monitor_kwargs["verbose"] = True
    if getattr(args, "train_log_file", None):
        _monitor_kwargs["log_file"] = args.train_log_file

    try:
        if _monitor_kwargs and args.model:
            from .leaderboard import _run_with_adapter
            from surge.model.registry import MODEL_REGISTRY
            adapter = MODEL_REGISTRY.create(args.model, **_monitor_kwargs)
            result = _run_with_adapter(single_bm_key, adapter, seed=args.seed)
            if result is None:
                print("Error: benchmark run failed (see stderr above).", file=sys.stderr)
                return 1
        else:
            result = run_benchmark(single_bm_key, seed=args.seed, model_key=args.model)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _print_result_banner(result)
    rp = _persist(result, save_root=args.save_dir, no_save=args.no_save)

    if args.mlflow:
        _mlflow_log(result, result_path=rp,
                    experiment=args.mlflow_experiment,
                    tracking_uri=args.mlflow_tracking_uri)

    if args.output is not None:
        import json
        text = json.dumps(result.to_dict(), indent=2, default=str)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
