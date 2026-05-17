"""
Cross-model leaderboard for SURGE benchmarks.

Runs every compatible model against one or more benchmarks and renders a
per-benchmark comparison table (rows = models, columns = metrics) with the
best value per column highlighted.  All results are optionally logged to
MLflow so the comparison is browsable in the tracking UI.

Typical usage
-------------
::

    python -m surge.benchmarks.run --leaderboard --tier 1 --task-type classification
    python -m surge.benchmarks.run --leaderboard --benchmark tabular.iris
    python -m surge.benchmarks.run --leaderboard --all-benchmarks --mlflow
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base import BenchmarkResult
from .registry import benchmark_info, list_benchmarks, run_benchmark

# ---------------------------------------------------------------------------
# Model compatibility matrix
# ---------------------------------------------------------------------------

# Models to try for each task type.  Listed in the order they appear in the
# table (roughly: ensemble → boosting → linear → neural).
_REGRESSION_MODELS: list[str] = [
    "sklearn.random_forest",
    "sklearn.mlp",
]

_CLASSIFICATION_MODELS: list[str] = [
    "sklearn.random_forest_classifier",
    "sklearn.gradient_boosting_classifier",
    "sklearn.logistic_regression",
]


def _default_models_for(task_type: str) -> list[str]:
    if task_type == "regression":
        base = list(_REGRESSION_MODELS)
        try:
            from surge.model.pytorch import PYTORCH_AVAILABLE

            if PYTORCH_AVAILABLE:
                base.append("pytorch.mlp")
        except Exception:
            pass
        return base
    return list(_CLASSIFICATION_MODELS)


# Metrics where lower is better (used to decide which direction to highlight).
_LOWER_IS_BETTER: frozenset[str] = frozenset({"runtime_s", "test_rmse"})

# Preferred column order for display (unknown keys appended alphabetically).
_METRIC_ORDER: list[str] = [
    "test_accuracy",
    "test_f1_macro",
    "test_auroc",
    "test_r2",
    "test_rmse",
    "runtime_s",
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_leaderboard(
    benchmark_keys: list[str],
    *,
    model_keys: list[str] | None = None,
    seed: int = 42,
    pytorch_mlp_epochs: int = 100,
    save_root: Path | None = Path("benchmark_reports"),
) -> dict[str, list[BenchmarkResult]]:
    """
    Run every compatible model against each benchmark.

    Parameters
    ----------
    benchmark_keys:
        Benchmarks to run (must be registered).
    model_keys:
        Override the default model list.  If ``None`` the compatible set for
        each benchmark's ``task_type`` is used.
    seed:
        Random seed passed to every task.
    pytorch_mlp_epochs:
        ``n_epochs`` cap for ``pytorch.mlp`` in leaderboard runs (avoids
        very long waits; default 100).
    save_root:
        Auto-save individual results here (``None`` to skip).

    Returns
    -------
    dict mapping benchmark_key → list[BenchmarkResult] (one per model tried).
    """
    from surge.model.registry import MODEL_REGISTRY

    results: dict[str, list[BenchmarkResult]] = {}

    for key in benchmark_keys:
        info = benchmark_info(key)
        task_type = info["task_type"]
        candidates = model_keys if model_keys is not None else _default_models_for(task_type)
        results[key] = []

        for model_key in candidates:
            # Validate the model is actually registered before running.
            if model_key not in MODEL_REGISTRY:
                print(
                    f"  [skip] {model_key} not in MODEL_REGISTRY — skipping",
                    file=sys.stderr,
                )
                continue

            # For pytorch.mlp cap epochs so leaderboard runs don't take too long.
            if model_key == "pytorch.mlp":
                try:
                    adapter = MODEL_REGISTRY.create(model_key, n_epochs=pytorch_mlp_epochs)
                    # Patch: pass a pre-built adapter by temporarily registering a
                    # wrapper.  Simpler: call the task function directly with model_key
                    # and a monkey-patched registry entry — but the cleanest path is
                    # to support passing an adapter instance to the task.  For now
                    # we override n_epochs via the kwargs path by creating a custom
                    # model_key string that resolves to a pre-configured adapter.
                    # The simplest approach: temporarily store the epoch-capped adapter
                    # and replace the lookup.
                    from surge.model.registry import MODEL_REGISTRY as _MR
                    from surge.benchmarks.tasks import _fit_predict_regression, _reg_metrics
                    import time, numpy as np
                    from sklearn.datasets import (
                        load_diabetes, fetch_california_housing,
                    )
                    from sklearn.model_selection import train_test_split

                    # Run the benchmark, but inject the epoch-capped adapter.
                    res = _run_with_adapter(key, adapter, seed=seed)
                    if res is not None:
                        if save_root is not None:
                            res.save(root=save_root)
                        results[key].append(res)
                    continue
                except Exception as exc:
                    print(f"  [warn] pytorch.mlp leaderboard run failed: {exc}", file=sys.stderr)
                    continue

            try:
                res = run_benchmark(key, seed=seed, model_key=model_key)
                if save_root is not None:
                    res.save(root=save_root)
                results[key].append(res)
            except Exception as exc:
                print(
                    f"  [error] {key} / {model_key}: {exc}",
                    file=sys.stderr,
                )

    return results


def _run_with_adapter(benchmark_key: str, adapter: Any, *, seed: int) -> BenchmarkResult | None:
    """Run a benchmark using a pre-instantiated adapter."""
    import time

    import numpy as np
    from sklearn.model_selection import train_test_split

    from .base import BenchmarkResult
    from .tasks import _clf_metrics, _reg_metrics
    from .registry import benchmark_info

    info = benchmark_info(benchmark_key)
    task_type = info["task_type"]

    try:
        X, y = _load_dataset(benchmark_key)
    except Exception as exc:
        print(f"  [error] could not load {benchmark_key}: {exc}", file=sys.stderr)
        return None

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )

    t0 = time.perf_counter()
    try:
        adapter.fit(X_train, y_train)
        y_pred = np.asarray(adapter.predict(X_test))
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"  [error] fit/predict failed for {benchmark_key}: {exc}", file=sys.stderr)
        return None

    if task_type == "regression":
        metrics = _reg_metrics(y_test, y_pred.ravel())
    else:
        y_prob = None
        if hasattr(adapter, "predict_proba"):
            try:
                y_prob = adapter.predict_proba(X_test)
            except Exception:
                pass
        metrics = _clf_metrics(y_test, y_pred, y_prob)

    metrics["runtime_s"] = elapsed
    passed = _check_pass(benchmark_key, metrics)

    return BenchmarkResult(
        benchmark_key=benchmark_key,
        model_key=adapter.name,
        tier=info["tier"],
        task_type=task_type,
        metrics=metrics,
        passed=passed,
        message=f"leaderboard run via {adapter.name}",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def _load_dataset(benchmark_key: str):
    """Load the raw (X, y) arrays for a benchmark key."""
    from sklearn.datasets import (
        fetch_california_housing,
        load_breast_cancer,
        load_diabetes,
        load_digits,
        load_iris,
        load_wine,
    )

    loaders = {
        "synthetic.regression_1d": lambda: _synthetic_regression_1d(),
        "synthetic.classification_binary": lambda: _synthetic_classification_binary(),
        "tabular.diabetes": lambda: load_diabetes(return_X_y=True),
        "tabular.california_housing": lambda: fetch_california_housing(return_X_y=True),
        "tabular.iris": lambda: load_iris(return_X_y=True),
        "tabular.breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "tabular.wine": lambda: load_wine(return_X_y=True),
        "tabular.digits": lambda: load_digits(return_X_y=True),
    }
    if benchmark_key not in loaders:
        raise KeyError(f"No dataset loader for {benchmark_key!r}")
    return loaders[benchmark_key]()


def _synthetic_regression_1d():
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.uniform(-1.0, 1.0, size=(400, 1))
    y = 3.0 * X.ravel() + 1.5 + 0.15 * rng.standard_normal(400)
    return X, y


def _synthetic_classification_binary():
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 20))
    logits = X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(500)
    y = (logits > 0).astype(int)
    return X, y


def _check_pass(benchmark_key: str, metrics: dict) -> bool:
    """Best-effort pass check using known thresholds."""
    _THRESHOLDS: dict[str, tuple[str, float]] = {
        "synthetic.regression_1d": ("test_r2", 0.85),
        "tabular.diabetes": ("test_r2", 0.35),
        "tabular.california_housing": ("test_r2", 0.75),
        "tabular.concrete_strength": ("test_r2", 0.80),
        "synthetic.classification_binary": ("test_accuracy", 0.75),
        "tabular.iris": ("test_accuracy", 0.88),
        "tabular.breast_cancer": ("test_accuracy", 0.93),
        "tabular.wine": ("test_accuracy", 0.90),
        "tabular.digits": ("test_accuracy", 0.95),
    }
    if benchmark_key not in _THRESHOLDS:
        return True
    metric_key, threshold = _THRESHOLDS[benchmark_key]
    val = metrics.get(metric_key)
    if val is None:
        return True
    return float(val) >= threshold


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def format_leaderboard_table(
    benchmark_key: str,
    results: list[BenchmarkResult],
    *,
    highlight_char: str = "*",
) -> str:
    """
    Return a formatted comparison table string for one benchmark.

    Rows are models; columns are metrics.  The best value per column is
    marked with ``highlight_char``.
    """
    if not results:
        return f"  (no results for {benchmark_key})"

    info = benchmark_info(benchmark_key)

    # Collect all metric keys in preferred order.
    all_keys: list[str] = []
    seen: set[str] = set()
    for mk in _METRIC_ORDER:
        if any(mk in r.metrics for r in results):
            all_keys.append(mk)
            seen.add(mk)
    for r in results:
        for mk in sorted(r.metrics):
            if mk not in seen:
                all_keys.append(mk)
                seen.add(mk)

    # Build value matrix and find best per column.
    model_names = [r.model_key for r in results]
    passed_flags = [r.passed for r in results]
    matrix: list[list[float | None]] = [
        [r.metrics.get(k) for k in all_keys] for r in results
    ]

    best_idx: list[int | None] = []
    for col_idx, metric_key in enumerate(all_keys):
        col_vals = [matrix[row][col_idx] for row in range(len(results))]
        numeric = [(i, v) for i, v in enumerate(col_vals) if v is not None]
        if not numeric:
            best_idx.append(None)
            continue
        if metric_key in _LOWER_IS_BETTER:
            best_i = min(numeric, key=lambda x: x[1])[0]
        else:
            best_i = max(numeric, key=lambda x: x[1])[0]
        best_idx.append(best_i)

    # Column widths.
    col_w = max(len(k) + 3 for k in all_keys) if all_keys else 12
    col_w = max(col_w, 12)
    model_w = max((len(n) for n in model_names), default=10) + 4

    # Header.
    lines: list[str] = []
    lines.append(
        f"\nBenchmark : {benchmark_key}  "
        f"(task={info['task_type']}, tier={info['tier']}, shape={info['shape']})"
    )
    lines.append("─" * (model_w + col_w * len(all_keys) + 8))

    header = f"{'Model':<{model_w}}  {'Pass':4}  "
    header += "".join(f"{k:>{col_w}}" for k in all_keys)
    lines.append(header)
    lines.append("─" * (model_w + col_w * len(all_keys) + 8))

    for row_idx, (model_name, passed) in enumerate(zip(model_names, passed_flags)):
        status = "PASS" if passed else "FAIL"
        row = f"{model_name:<{model_w}}  {status:4}  "
        for col_idx, metric_key in enumerate(all_keys):
            val = matrix[row_idx][col_idx]
            is_best = best_idx[col_idx] == row_idx
            if val is None:
                cell = "—"
            elif metric_key == "runtime_s":
                cell = f"{val:.2f}s"
            else:
                cell = f"{val:.4f}"
            if is_best:
                cell = f"{cell}{highlight_char}"
            row += f"{cell:>{col_w}}"
        lines.append(row)

    lines.append("─" * (model_w + col_w * len(all_keys) + 8))
    lines.append(
        f"  {highlight_char} = best  "
        f"(↑ higher-is-better: accuracy, f1, auroc, r2 | ↓ lower-is-better: rmse, runtime)"
    )
    return "\n".join(lines)


def print_leaderboard(results_by_benchmark: dict[str, list[BenchmarkResult]]) -> None:
    """Print all per-benchmark tables to stdout."""
    for bk, res_list in results_by_benchmark.items():
        print(format_leaderboard_table(bk, res_list))
        print()


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_leaderboard_to_mlflow(
    results_by_benchmark: dict[str, list[BenchmarkResult]],
    *,
    experiment_name: str = "surge_benchmarks",
    tracking_uri: str | None = None,
    save_tables: bool = True,
    tables_dir: Path | None = None,
) -> bool:
    """
    Log all leaderboard results to MLflow.

    Each (benchmark, model) pair becomes one MLflow run tagged with
    ``benchmark_key``, ``model_key``, ``tier``, ``passed``, and
    ``surge_version``.  All numeric metrics are logged so MLflow's
    "Compare runs" panel becomes an instant leaderboard.

    Parameters
    ----------
    save_tables:
        If True, a formatted text table for each benchmark is saved as a
        plain-text artifact attached to the *first* run for that benchmark,
        so it is accessible from the MLflow UI.
    tables_dir:
        Directory to write temporary table text files before uploading.
        Defaults to ``benchmark_reports/.leaderboard_tables/``.

    Returns
    -------
    bool — True on success, False if MLflow unavailable or logging fails.
    """
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE

    if not MLFLOW_AVAILABLE:
        print(
            "[warn] MLflow not installed. pip install 'surge-ml[mlflow]'",
            file=sys.stderr,
        )
        return False

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    if save_tables and tables_dir is None:
        tables_dir = Path("benchmark_reports") / ".leaderboard_tables"

    try:
        for bk, res_list in results_by_benchmark.items():
            if not res_list:
                continue

            # Write the table as a text artifact once per benchmark.
            table_path: Path | None = None
            if save_tables and tables_dir is not None:
                tables_dir.mkdir(parents=True, exist_ok=True)
                table_text = format_leaderboard_table(bk, res_list)
                safe_key = bk.replace(".", "_")
                table_path = tables_dir / f"{safe_key}_leaderboard.txt"
                table_path.write_text(table_text + "\n", encoding="utf-8")

            for result_idx, result in enumerate(res_list):
                run_name = f"{result.benchmark_key}__{result.model_key}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({
                        "benchmark_key": result.benchmark_key,
                        "model_key": result.model_key,
                        "tier": result.tier,
                        "task_type": result.task_type,
                        "passed": str(result.passed),
                        "surge_version": result.surge_version or "",
                        "timestamp": result.timestamp or "",
                    })
                    mlflow.log_params({
                        "benchmark_key": result.benchmark_key,
                        "model_key": result.model_key,
                        "tier": result.tier,
                        "task_type": result.task_type,
                        "n_train": result.extra.get("n_train", ""),
                        "n_test": result.extra.get("n_test", ""),
                    })
                    numeric = {
                        k: float(v)
                        for k, v in result.metrics.items()
                        if isinstance(v, (int, float))
                    }
                    if numeric:
                        mlflow.log_metrics(numeric)

                    # Attach the leaderboard table to the first run for this benchmark.
                    if result_idx == 0 and table_path is not None and table_path.exists():
                        mlflow.log_artifact(str(table_path), artifact_path="leaderboard")

        return True
    except Exception as exc:
        print(f"[warn] MLflow leaderboard logging failed: {exc}", file=sys.stderr)
        return False
