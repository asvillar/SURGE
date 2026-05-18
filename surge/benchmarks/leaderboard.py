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

import numpy as np

from .base import BenchmarkResult
from .registry import benchmark_info, list_benchmarks, run_benchmark

# ---------------------------------------------------------------------------
# Model compatibility matrix
# ---------------------------------------------------------------------------

# Models to try for each task type.  Listed in the order they appear in the
# table (roughly: ensemble → boosting → linear → neural).
_REGRESSION_MODELS: list[str] = [
    "sklearn.random_forest",
    "sklearn.gradient_boosting_regressor",
    "sklearn.mlp",
]

_CLASSIFICATION_MODELS: list[str] = [
    "sklearn.random_forest_classifier",
    "sklearn.gradient_boosting_classifier",
    "sklearn.logistic_regression",
]


# Per-benchmark overrides: map a benchmark key to a specific model list.
# PDEBench benchmarks only use neural-operator / deep-learning models.
# Tabular/sklearn/XGBoost models are not viable at PDEBench spatial scales
# and are marked N/A in the leaderboard instead.
_PDEBENCH_OPERATOR_MODELS: list[str] = []
try:
    from surge.model.pytorch import PYTORCH_AVAILABLE as _PT
    if _PT:
        _PDEBENCH_OPERATOR_MODELS = [
            "pytorch.fno1d",
            "pytorch.deeponet",
            "pytorch.mlp",
            "pytorch.residual_mlp",
            "pytorch.cnn1d",
        ]
except Exception:
    pass

_BENCHMARK_MODEL_OVERRIDES: dict[str, list[str]] = {
    # Inline PDE benchmark (64-pt grid, no download) — operator models only
    "pde.burgers_1d":            _PDEBENCH_OPERATOR_MODELS,
    # Real PDEBench HDF5 data — operator models only, tabular models are N/A
    "pdebench.burgers_1d":       _PDEBENCH_OPERATOR_MODELS,
    "pdebench.darcy_2d":         _PDEBENCH_OPERATOR_MODELS,
    "pdebench.shallow_water_2d": _PDEBENCH_OPERATOR_MODELS,
}


def _default_models_for(task_type: str, benchmark_key: str | None = None) -> list[str]:
    # Per-benchmark override takes precedence.
    if benchmark_key is not None and benchmark_key in _BENCHMARK_MODEL_OVERRIDES:
        return list(_BENCHMARK_MODEL_OVERRIDES[benchmark_key])

    if task_type == "regression":
        base = list(_REGRESSION_MODELS)
        try:
            from surge.model.backends.xgboost import XGBOOST_AVAILABLE

            if XGBOOST_AVAILABLE:
                base.append("xgboost.xgbregressor")
        except Exception:
            pass
        try:
            from surge.model.pytorch import PYTORCH_AVAILABLE

            if PYTORCH_AVAILABLE:
                base.append("pytorch.mlp")
                base.append("pytorch.residual_mlp")
                # Sequence/temporal models handle flat input internally — valid
                # for tabular and sequence benchmarks alike.
                base.extend(["pytorch.cnn1d", "pytorch.lstm", "pytorch.gru"])
                # NOTE: pytorch.fno1d and pytorch.deeponet are NOT added here.
                # They are PDE spatial-field operators that require y.ndim >= 2
                # (shape (N, nx)).  They are only valid for pdebench.* and
                # pde.* benchmarks, and appear via _BENCHMARK_MODEL_OVERRIDES.
        except Exception:
            pass
        return base
    # classification
    base = list(_CLASSIFICATION_MODELS)
    try:
        from surge.model.backends.xgboost import XGBOOST_AVAILABLE

        if XGBOOST_AVAILABLE:
            base.append("xgboost.xgbclassifier")
    except Exception:
        pass
    try:
        from surge.model.pytorch import PYTORCH_AVAILABLE

        if PYTORCH_AVAILABLE:
            base.append("pytorch.mlp_classifier")
    except Exception:
        pass
    return base


# Metrics where lower is better (used to decide which direction to highlight).
_LOWER_IS_BETTER: frozenset[str] = frozenset({"runtime_s", "test_rmse", "test_nrmse", "test_relative_l2"})

# Preferred column order for display (unknown keys appended alphabetically).
_METRIC_ORDER: list[str] = [
    "test_accuracy",
    "test_f1_macro",
    "test_auroc",
    "test_r2",
    "test_rmse",
    "test_nrmse",
    "test_relative_l2",
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
        candidates = model_keys if model_keys is not None else _default_models_for(task_type, key)
        results[key] = []

        for model_key in candidates:
            # Validate the model is actually registered before running.
            if model_key not in MODEL_REGISTRY:
                print(
                    f"  [skip] {model_key} not in MODEL_REGISTRY — skipping",
                    file=sys.stderr,
                )
                continue

            # Cap epochs for all pytorch adapters so leaderboard runs don't take too long.
            _PYTORCH_EPOCH_CAP_MODELS = {
                "pytorch.mlp", "pytorch.residual_mlp", "pytorch.mlp_classifier",
                "pytorch.cnn1d", "pytorch.lstm", "pytorch.gru",
                "pytorch.fno1d", "pytorch.deeponet",
                "pytorch.lenet5", "pytorch.resnet20", "pytorch.resnet56",
            }
            if model_key in _PYTORCH_EPOCH_CAP_MODELS:
                try:
                    adapter = MODEL_REGISTRY.create(model_key, n_epochs=pytorch_mlp_epochs)
                    res = _run_with_adapter(key, adapter, seed=seed)
                    if res is not None:
                        if save_root is not None:
                            res.save(root=save_root)
                        results[key].append(res)
                    continue
                except Exception as exc:
                    print(f"  [warn] {model_key} leaderboard run failed: {exc}", file=sys.stderr)
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
        # Sequence benchmarks: compute NRMSE instead of / in addition to R².
        if benchmark_key.startswith("sequence.") or benchmark_key.startswith("pde.") or benchmark_key.startswith("pdebench."):
            nrmse = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))
            metrics = {"test_nrmse": nrmse}
            # Also compute R² if single-output, skip for multi-output PDE fields.
            try:
                from sklearn.metrics import r2_score
                r2 = float(r2_score(y_test, y_pred.ravel() if y_test.ndim == 1 else y_pred))
                metrics["test_r2"] = r2
            except Exception:
                pass
        else:
            # Support multi-output: only ravel for 1-D targets.
            metrics = _reg_metrics(y_test, y_pred.ravel() if y_test.ndim == 1 else y_pred)
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
        "synthetic.multioutput_2d": lambda: _synthetic_multioutput_2d(),
        "tabular.diabetes": lambda: load_diabetes(return_X_y=True),
        "tabular.california_housing": lambda: fetch_california_housing(return_X_y=True),
        "tabular.concrete_strength": lambda: _load_concrete_strength(),
        "tabular.energy_efficiency": lambda: _load_energy_efficiency(),
        "tabular.iris": lambda: load_iris(return_X_y=True),
        "tabular.breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "tabular.wine": lambda: load_wine(return_X_y=True),
        "tabular.digits": lambda: load_digits(return_X_y=True),
        "sequence.lorenz63": lambda: _load_lorenz63(),
        "pde.burgers_1d": lambda: _load_burgers_1d(),
        "classification.flow_regime": lambda: _load_flow_regime(),
        "tabular.airfoil_noise": lambda: _load_airfoil_noise(),
        "tabular.yacht_dynamics": lambda: _load_yacht_dynamics(),
        "classification.plasma_stability": lambda: _load_plasma_stability(),
        "tabular.superconductor": lambda: _load_superconductor(),
        "multioutput.scm20d": lambda: _load_scm20d(),
        "classification.covertype": lambda: _load_covertype(),
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


def _synthetic_multioutput_2d():
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.standard_normal((600, 8))
    A = rng.standard_normal((8, 2)) * 0.5
    noise = 0.1 * rng.standard_normal((600, 2))
    Y = X @ A + noise
    return X, Y


def _load_concrete_strength():
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="concrete-compressive-strength", version=1, as_frame=True, parser="auto")
    return data.data.values.astype(float), data.target.values.astype(float)


def _load_lorenz63():
    from .tasks import _generate_lorenz_trajectories

    return _generate_lorenz_trajectories(n_trajectories=1200, T_in=20, T_out=20, dt=0.01, warmup=500, seed=42)


def _load_burgers_1d():
    from .tasks import _generate_burgers_dataset

    return _generate_burgers_dataset(n_samples=1024, n_x=64, nt=100, dt=1e-3, nu=0.01, seed=42)


def _load_flow_regime(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 800
    mach = rng.uniform(0.1, 3.0, n)
    log_re = rng.uniform(4.0, 8.0, n)
    aoa = rng.uniform(-5.0, 25.0, n)
    labels = np.zeros(n, dtype=int)
    labels[(mach < 0.8) & (log_re < 5.5)] = 0
    labels[(mach < 0.8) & (log_re >= 5.5)] = 1
    labels[(mach >= 0.8) & (mach < 1.2)] = 2
    labels[(mach >= 1.2)] = 3
    noise_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    labels[noise_idx] = rng.integers(0, 4, size=len(noise_idx))
    X = np.column_stack([mach, log_re, aoa])
    return X, labels


def _load_airfoil_noise():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="airfoil_self_noise", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if hasattr(data.target, "values") else np.asarray(data.target, dtype=float)
    return X, y


def _load_yacht_dynamics():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if hasattr(data.target, "values") else np.asarray(data.target, dtype=float)
    return X, y


def _load_plasma_stability():
    import io
    import urllib.request
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    _UCI_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"
    )
    with urllib.request.urlopen(_UCI_URL, timeout=30) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))
    feature_cols = [c for c in df.columns if c not in ("stab", "stabf")]
    X = df[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["stabf"].values)
    return X, y


def _load_energy_efficiency():
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="energy-efficiency", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if data.target.ndim == 1 else data.target.iloc[:, 0].values.astype(float)
    return X, y


def _load_superconductor():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="superconduct", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    return X, y


def _load_scm20d():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="scm20d", version=2, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    return X, y


def _load_covertype():
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    data = fetch_openml(name="covertype", version=3, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(data.target.values if hasattr(data.target, "values") else data.target)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(20_000, len(y)), replace=False)
    return X[idx], y[idx]


def _check_pass(benchmark_key: str, metrics: dict) -> bool:
    """Best-effort pass check using known thresholds."""
    _THRESHOLDS: dict[str, tuple[str, float]] = {
        "synthetic.regression_1d": ("test_r2", 0.85),
        "synthetic.multioutput_2d": ("test_r2", 0.75),
        "tabular.diabetes": ("test_r2", 0.35),
        "tabular.california_housing": ("test_r2", 0.75),
        "tabular.concrete_strength": ("test_r2", 0.80),
        "tabular.energy_efficiency": ("test_r2", 0.90),
        "synthetic.classification_binary": ("test_accuracy", 0.75),
        "tabular.iris": ("test_accuracy", 0.88),
        "tabular.breast_cancer": ("test_accuracy", 0.93),
        "tabular.wine": ("test_accuracy", 0.90),
        "tabular.digits": ("test_accuracy", 0.95),
        "sequence.lorenz63": ("test_nrmse", 0.30),  # lower is better
        "pde.burgers_1d": ("test_relative_l2", 0.10),  # lower is better
        "classification.flow_regime": ("test_accuracy", 0.85),
        "tabular.airfoil_noise": ("test_r2", 0.80),
        "tabular.yacht_dynamics": ("test_r2", 0.80),
        "classification.plasma_stability": ("test_accuracy", 0.92),
        "tabular.superconductor": ("test_r2", 0.90),
        "multioutput.scm20d": ("test_r2", 0.60),
        "classification.covertype": ("test_accuracy", 0.85),
    }
    if benchmark_key not in _THRESHOLDS:
        return True
    metric_key, threshold = _THRESHOLDS[benchmark_key]
    val = metrics.get(metric_key)
    if val is None:
        return True
    # For lower-is-better metrics, pass if val <= threshold.
    if metric_key in _LOWER_IS_BETTER or metric_key == "test_nrmse":
        return float(val) <= threshold
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
