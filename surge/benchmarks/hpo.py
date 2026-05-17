"""Optuna-based hyperparameter optimisation for SURGE benchmark models.

Typical usage
-------------
::

    # CLI
    python -m surge.benchmarks.run \\
        --benchmark tabular.california_housing \\
        --model xgboost.xgbregressor \\
        --hpo --hpo-trials 40

    # Programmatic
    from surge.benchmarks.hpo import run_benchmark_hpo
    result, best_params = run_benchmark_hpo(
        "tabular.california_housing",
        "xgboost.xgbregressor",
        n_trials=40,
    )

Design
------
Each trial:
  1. ``suggest_params(model_key, trial)`` draws hyperparameters from the
     search space for that model.
  2. An adapter is instantiated with those params.
  3. ``_run_with_adapter`` from the leaderboard module fits and evaluates
     the adapter on the benchmark dataset.
  4. The primary metric (auto-selected by task type, or user-specified) is
     returned as the Optuna objective value.

The best ``BenchmarkResult`` is returned alongside the best hyperparameter
dict.  If ``save_root`` is set the best result is auto-saved.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("surge.benchmarks.hpo")

# ---------------------------------------------------------------------------
# Search spaces
# Each entry: param_name → (type, *args)
#   "float"      → (lo, hi, log=False)
#   "float_log"  → (lo, hi, log=True)
#   "int"        → (lo, hi, step=1)
#   "int_log"    → (lo, hi, log=True)
#   "categorical"→ ([choices],)
# ---------------------------------------------------------------------------

_SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    # ------------------------------------------------------------------
    # sklearn
    # ------------------------------------------------------------------
    "sklearn.random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("categorical", [None, 5, 10, 15, 20]),
        "min_samples_leaf": ("int", 1, 8),
        "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.8]),
    },
    "sklearn.gradient_boosting_regressor": {
        "n_estimators": ("int", 50, 400),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "max_depth": ("int", 2, 8),
        "subsample": ("float", 0.5, 1.0),
        "min_samples_leaf": ("int", 1, 8),
    },
    "sklearn.mlp": {
        "hidden_layer_sizes": ("categorical", [
            (64,), (128,), (256,),
            (64, 64), (128, 64), (256, 128),
            (128, 128), (256, 128, 64),
        ]),
        "learning_rate_init": ("float_log", 1e-4, 1e-1),
        "alpha": ("float_log", 1e-5, 1e-1),
        "batch_size": ("categorical", [32, 64, 128, 256, "auto"]),
    },
    "sklearn.random_forest_classifier": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("categorical", [None, 5, 10, 15, 20]),
        "min_samples_leaf": ("int", 1, 8),
        "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.8]),
    },
    "sklearn.gradient_boosting_classifier": {
        "n_estimators": ("int", 50, 400),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "max_depth": ("int", 2, 8),
        "subsample": ("float", 0.5, 1.0),
        "min_samples_leaf": ("int", 1, 8),
    },
    "sklearn.logistic_regression": {
        "C": ("float_log", 1e-3, 1e3),
        "solver": ("categorical", ["lbfgs", "saga"]),
        "max_iter": ("int", 200, 2000),
    },
    "sklearn.gpr": {
        "alpha": ("float_log", 1e-6, 1e-1),
    },
    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    "xgboost.xgbregressor": {
        "n_estimators": ("int", 50, 600),
        "learning_rate": ("float_log", 1e-3, 0.5),
        "max_depth": ("int", 2, 10),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.4, 1.0),
        "reg_alpha": ("float_log", 1e-5, 10.0),
        "reg_lambda": ("float_log", 1e-2, 10.0),
    },
    "xgboost.xgbclassifier": {
        "n_estimators": ("int", 50, 600),
        "learning_rate": ("float_log", 1e-3, 0.5),
        "max_depth": ("int", 2, 10),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.4, 1.0),
        "reg_alpha": ("float_log", 1e-5, 10.0),
        "reg_lambda": ("float_log", 1e-2, 10.0),
    },
    # ------------------------------------------------------------------
    # PyTorch
    # ------------------------------------------------------------------
    "pytorch.mlp": {
        "hidden_layers": ("categorical", [
            [64], [128], [256],
            [64, 64], [128, 64], [128, 128], [256, 128],
            [256, 128, 64], [512, 256, 128],
        ]),
        "learning_rate": ("float_log", 1e-4, 1e-1),
        "dropout_rate": ("float", 0.0, 0.5),
        "batch_size": ("categorical", [32, 64, 128, 256]),
    },
    "pytorch.residual_mlp": {
        "hidden_layers": ("categorical", [
            [64, 64], [128, 128], [256, 256],
            [128, 128, 128], [256, 128, 128],
            [256, 256, 128], [512, 256, 128],
        ]),
        "learning_rate": ("float_log", 1e-4, 1e-1),
        "dropout_rate": ("float", 0.0, 0.4),
        "batch_size": ("categorical", [32, 64, 128]),
        "patience": ("int", 10, 40),
    },
    "pytorch.mlp_classifier": {
        "hidden_layers": ("categorical", [
            [64, 32], [128, 64], [256, 128],
            [128, 64, 32], [256, 128, 64],
        ]),
        "learning_rate": ("float_log", 1e-4, 1e-1),
        "dropout_rate": ("float", 0.0, 0.5),
        "batch_size": ("categorical", [32, 64, 128]),
        "patience": ("int", 10, 30),
    },
}

# Primary metric to maximise/minimise per task type.
_PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "regression": ("test_r2", "maximize"),
    "classification": ("test_accuracy", "maximize"),
}


def suggest_params(model_key: str, trial: Any) -> dict[str, Any]:
    """
    Draw a hyperparameter dict for *model_key* from an Optuna ``trial``.

    Parameters
    ----------
    model_key:
        Registered model key (e.g. ``"xgboost.xgbregressor"``).
    trial:
        ``optuna.trial.Trial`` instance.

    Returns
    -------
    Dict of ``{param_name: value}`` ready to pass to ``MODEL_REGISTRY.create``.

    Raises
    ------
    KeyError
        If ``model_key`` has no registered search space.
    """
    if model_key not in _SEARCH_SPACES:
        raise KeyError(
            f"No HPO search space for {model_key!r}. "
            f"Available: {', '.join(sorted(_SEARCH_SPACES))}"
        )
    space = _SEARCH_SPACES[model_key]
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float":
            lo, hi = spec[1], spec[2]
            log = len(spec) > 3 and spec[3]
            params[name] = trial.suggest_float(name, lo, hi, log=log)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "int":
            step = spec[3] if len(spec) > 3 else 1
            params[name] = trial.suggest_int(name, spec[1], spec[2], step=step)
        elif kind == "int_log":
            params[name] = trial.suggest_int(name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            choices = spec[1]
            chosen = trial.suggest_categorical(name, list(range(len(choices))))
            params[name] = choices[chosen]
        else:
            raise ValueError(f"Unknown param type {kind!r} for {name!r}")
    return params


def list_hpo_models() -> list[str]:
    """Return model keys that have a registered HPO search space."""
    return sorted(_SEARCH_SPACES)


def run_benchmark_hpo(
    benchmark_key: str,
    model_key: str,
    *,
    n_trials: int = 20,
    seed: int = 42,
    metric: str | None = None,
    direction: str | None = None,
    n_epochs_cap: int | None = None,
    save_root: Path | None = Path("benchmark_reports"),
    verbose: bool = False,
    mlflow_experiment: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Run Optuna HPO for *model_key* on *benchmark_key*.

    Parameters
    ----------
    benchmark_key:
        Registered benchmark (e.g. ``"tabular.california_housing"``).
    model_key:
        Registered model (e.g. ``"xgboost.xgbregressor"``).
    n_trials:
        Number of Optuna trials.
    seed:
        Random seed (both numpy and Optuna sampler).
    metric:
        Metric to optimise.  Defaults to the task-type primary metric
        (``test_r2`` for regression, ``test_accuracy`` for classification).
    direction:
        ``"maximize"`` or ``"minimize"``.  Auto-derived from *metric* if
        ``None``.
    n_epochs_cap:
        For PyTorch adapters, cap ``n_epochs`` to this value to speed up
        each trial.  Defaults to ``50``.
    save_root:
        Auto-save the best result here (``None`` to skip).
    verbose:
        Print every trial result to stderr.
    mlflow_experiment:
        If set, log every trial and the best result to MLflow.
    mlflow_tracking_uri:
        MLflow tracking server URI.

    Returns
    -------
    ``(best_result, best_params)``
    """
    import optuna

    from .leaderboard import _run_with_adapter
    from .registry import benchmark_info
    from surge.model.registry import MODEL_REGISTRY

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )

    info = benchmark_info(benchmark_key)
    task_type = info["task_type"]

    # Auto-select metric + direction.
    if metric is None:
        metric, default_dir = _PRIMARY_METRIC.get(task_type, ("test_r2", "maximize"))
    else:
        default_dir = "minimize" if metric in {"test_rmse", "runtime_s"} else "maximize"
    if direction is None:
        direction = default_dir

    _PYTORCH_KEYS = {"pytorch.mlp", "pytorch.residual_mlp", "pytorch.mlp_classifier"}
    epochs_cap = n_epochs_cap if n_epochs_cap is not None else 50

    # Pre-load the dataset once so every trial reuses the same in-memory arrays.
    from .leaderboard import _load_dataset
    from sklearn.model_selection import train_test_split

    try:
        X_all, y_all = _load_dataset(benchmark_key)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load dataset for {benchmark_key!r}: {exc}\n"
            "For Tier-1 benchmarks that require internet access, ensure the "
            "dataset has been cached first by running the benchmark once normally."
        ) from exc

    stratify_split = y_all if info["task_type"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=stratify_split
    )

    best_result: list[Any] = [None]
    best_value: list[float] = [float("-inf") if direction == "maximize" else float("inf")]
    trial_results: list[dict] = []

    def objective(trial: Any) -> float:
        params = suggest_params(model_key, trial)
        if model_key in _PYTORCH_KEYS:
            params.setdefault("n_epochs", epochs_cap)
            params["n_epochs"] = min(params.get("n_epochs", epochs_cap), epochs_cap)

        try:
            adapter = MODEL_REGISTRY.create(model_key, **params)
        except Exception as exc:
            LOG.warning("Trial %d: create failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned()

        # Use pre-loaded data instead of reloading each trial.
        result = _run_with_adapter_data(
            benchmark_key, adapter,
            X_train, X_test, y_train, y_test,
            task_type=info["task_type"],
            tier=info["tier"],
        )
        if result is None:
            raise optuna.exceptions.TrialPruned()

        value = result.metrics.get(metric)
        if value is None or not np.isfinite(value):
            raise optuna.exceptions.TrialPruned()

        trial_results.append({"trial": trial.number, "params": params, "value": value})

        is_better = (
            value > best_value[0] if direction == "maximize" else value < best_value[0]
        )
        if is_better:
            best_value[0] = value
            best_result[0] = result

        if verbose:
            arrow = "↑" if direction == "maximize" else "↓"
            best_marker = " ← best" if is_better else ""
            print(
                f"  Trial {trial.number:3d}:  {metric}={value:.4f}{arrow}{best_marker}  "
                f"params={params}",
                file=sys.stderr,
            )

        return float(value)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params: dict[str, Any] = {}
    try:
        best_trial = study.best_trial
        raw = best_trial.params
        # Decode categorical indices back to actual values.
        space = _SEARCH_SPACES.get(model_key, {})
        for name, spec in space.items():
            if spec[0] == "categorical" and name in raw:
                best_params[name] = spec[1][raw[name]]
            elif name in raw:
                best_params[name] = raw[name]
    except ValueError:
        # All trials were pruned — no best trial available.
        best_trial = None

    # If all trials pruned, run once with default params as fallback.
    if best_result[0] is None:
        LOG.warning("All HPO trials were pruned; running with default params.")
        adapter = MODEL_REGISTRY.create(model_key)
        best_result[0] = _run_with_adapter_data(
            benchmark_key, adapter,
            X_train, X_test, y_train, y_test,
            task_type=info["task_type"],
            tier=info["tier"],
        )

    result = best_result[0]

    # Attach HPO metadata to the result's extra dict.
    if result is not None:
        result.extra["hpo_n_trials"] = len(trial_results)
        result.extra["hpo_best_params"] = best_params
        result.extra["hpo_metric"] = metric
        result.extra["hpo_direction"] = direction

    # Save best result.
    if save_root is not None and result is not None:
        result.save(root=save_root)

    # MLflow logging.
    if mlflow_experiment and result is not None:
        _log_hpo_to_mlflow(
            study,
            result,
            best_params=best_params,
            trial_results=trial_results,
            benchmark_key=benchmark_key,
            model_key=model_key,
            metric=metric,
            experiment_name=mlflow_experiment,
            tracking_uri=mlflow_tracking_uri,
        )

    return result, best_params


def _run_with_adapter_data(
    benchmark_key: str,
    adapter: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    task_type: str,
    tier: str,
) -> Any:
    """
    Fit *adapter* on pre-split data and return a :class:`BenchmarkResult`.

    Avoids re-loading the dataset on every HPO trial.
    """
    import time

    from .base import BenchmarkResult
    from .tasks import _clf_metrics, _reg_metrics
    from .leaderboard import _check_pass

    try:
        t0 = time.perf_counter()
        adapter.fit(X_train, y_train)
        y_pred = np.asarray(adapter.predict(X_test))
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        LOG.debug("fit/predict failed: %s", exc)
        return None

    if task_type == "regression":
        metrics = _reg_metrics(y_test, y_pred.ravel() if np.asarray(y_test).ndim == 1 else y_pred)
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
        tier=tier,
        task_type=task_type,
        metrics=metrics,
        passed=passed,
        message=f"HPO trial via {adapter.name}",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def _log_hpo_to_mlflow(
    study: Any,
    best_result: Any,
    *,
    best_params: dict,
    trial_results: list[dict],
    benchmark_key: str,
    model_key: str,
    metric: str,
    experiment_name: str,
    tracking_uri: str | None,
) -> None:
    """Log HPO summary + every trial to MLflow."""
    try:
        import mlflow
    except ImportError:
        LOG.warning("mlflow not installed; skipping HPO MLflow logging.")
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Parent run: best result summary.
    with mlflow.start_run(run_name=f"HPO/{benchmark_key}/{model_key}") as parent:
        mlflow.set_tags({
            "benchmark_key": benchmark_key,
            "model_key": model_key,
            "hpo_n_trials": len(trial_results),
            "hpo_metric": metric,
        })
        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.log_metrics(best_result.metrics)

        parent_id = parent.info.run_id

    # Child runs: one per trial.
    for tr in trial_results:
        with mlflow.start_run(
            run_name=f"trial_{tr['trial']:03d}",
            tags={"mlflow.parentRunId": parent_id,
                  "benchmark_key": benchmark_key,
                  "model_key": model_key},
        ):
            mlflow.log_params({k: str(v) for k, v in tr["params"].items()})
            mlflow.log_metric(metric, tr["value"])


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def print_hpo_summary(
    result: Any,
    best_params: dict[str, Any],
    *,
    benchmark_key: str,
    model_key: str,
    n_trials: int,
    metric: str,
) -> None:
    """Print a formatted HPO summary to stdout."""
    w = 72
    print("═" * w)
    print(f"  HPO Summary — {benchmark_key}  /  {model_key}")
    print(f"  Trials: {n_trials}   Metric: {metric}")
    print("─" * w)
    print("  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k:<35s} = {v}")
    print("─" * w)
    print("  Best metrics:")
    for k, v in result.metrics.items():
        marker = "  ★" if k == metric else ""
        print(f"    {k:<35s} = {v:.4f}{marker}")
    passed = "PASS" if result.passed else "FAIL"
    print(f"  Status: [{passed}]")
    print("═" * w)
