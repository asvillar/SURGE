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
    # ------------------------------------------------------------------
    # Phase C — Temporal (CNN1D, LSTM, GRU)
    # ------------------------------------------------------------------
    "pytorch.cnn1d": {
        "hidden_channels": ("categorical", [32, 64, 128, 256]),
        "n_layers": ("int", 2, 6),
        "kernel_size": ("categorical", [3, 5, 7, 9]),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "pytorch.lstm": {
        "hidden_size": ("categorical", [64, 128, 256, 512]),
        "n_layers": ("int", 1, 3),
        "dropout": ("float", 0.0, 0.4),
        "learning_rate": ("float_log", 1e-4, 5e-3),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "pytorch.gru": {
        "hidden_size": ("categorical", [64, 128, 256, 512]),
        "n_layers": ("int", 1, 3),
        "dropout": ("float", 0.0, 0.4),
        "learning_rate": ("float_log", 1e-4, 5e-3),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    # ------------------------------------------------------------------
    # Phase D — PDE operators (FNO1d, DeepONet)
    # ------------------------------------------------------------------
    "pytorch.fno1d": {
        "n_modes": ("categorical", [8, 12, 16, 24]),
        "hidden_channels": ("categorical", [32, 64, 128]),
        "n_layers": ("int", 2, 5),
        "learning_rate": ("float_log", 1e-4, 5e-3),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "pytorch.deeponet": {
        "branch_width": ("categorical", [64, 128, 256]),
        "trunk_width": ("categorical", [64, 128, 256]),
        "n_basis": ("categorical", [32, 64, 128]),
        "learning_rate": ("float_log", 1e-4, 5e-3),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    # ------------------------------------------------------------------
    # Phase E — Vision (LeNet-5, ResNet-20/56)
    # ------------------------------------------------------------------
    "pytorch.lenet5": {
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [64, 128, 256]),
        "dropout": ("float", 0.0, 0.5),
    },
    "pytorch.resnet20": {
        "learning_rate": ("float_log", 1e-4, 1e-1),
        "batch_size": ("categorical", [64, 128, 256]),
        "weight_decay": ("float_log", 1e-5, 1e-2),
    },
    "pytorch.resnet56": {
        "learning_rate": ("float_log", 1e-4, 1e-1),
        "batch_size": ("categorical", [64, 128, 256]),
        "weight_decay": ("float_log", 1e-5, 1e-2),
    },
    # ------------------------------------------------------------------
    # Phase 1 — BoTorch GP
    # ------------------------------------------------------------------
    "botorch.gp": {
        "kernel": ("categorical", ["rbf", "matern", "rbf_matern"]),
        "n_train_iter": ("int", 50, 300),
        "learning_rate": ("float_log", 1e-3, 0.5),
        "noise_init": ("float_log", 1e-3, 1.0),
    },
    "botorch.sparse_gp": {
        "n_inducing": ("categorical", [100, 200, 500]),
        "n_train_iter": ("int", 50, 300),
        "learning_rate": ("float_log", 1e-4, 0.1),
        "batch_size": ("categorical", [128, 256, 512]),
    },
    # ------------------------------------------------------------------
    # Phase 2 — KAN
    # ------------------------------------------------------------------
    "pytorch.kan": {
        "hidden_dims": ("categorical", [
            [32, 32], [64, 64], [128, 128],
            [64, 64, 64], [128, 64, 64],
        ]),
        "grid_size": ("categorical", [3, 5, 8, 10]),
        "spline_order": ("categorical", [2, 3, 4]),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [64, 128, 256]),
    },
    "pytorch.kan_classifier": {
        "hidden_dims": ("categorical", [
            [32, 32], [64, 64], [128, 128],
            [64, 64, 64], [128, 64, 64],
        ]),
        "grid_size": ("categorical", [3, 5, 8]),
        "spline_order": ("categorical", [2, 3, 4]),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [64, 128, 256]),
    },
    # ------------------------------------------------------------------
    # Phase 3 — FT-Transformer
    # ------------------------------------------------------------------
    "pytorch.ft_transformer": {
        "d_model": ("categorical", [64, 128, 256]),
        "n_heads": ("categorical", [4, 8]),
        "n_layers": ("int", 1, 6),
        "ffn_factor": ("float", 2.0, 6.0),
        "dropout": ("float", 0.0, 0.4),
        "learning_rate": ("float_log", 1e-5, 1e-3),
        "batch_size": ("categorical", [64, 128, 256]),
    },
    "pytorch.ft_transformer_classifier": {
        "d_model": ("categorical", [64, 128, 256]),
        "n_heads": ("categorical", [4, 8]),
        "n_layers": ("int", 1, 6),
        "ffn_factor": ("float", 2.0, 6.0),
        "dropout": ("float", 0.0, 0.4),
        "learning_rate": ("float_log", 1e-5, 1e-3),
        "batch_size": ("categorical", [64, 128, 256]),
    },
    # ------------------------------------------------------------------
    # Phase 4 — Vision (ViT, AlexNet)
    # ------------------------------------------------------------------
    "pytorch.vit": {
        "d_model": ("categorical", [64, 128, 192]),
        "n_heads": ("categorical", [4, 8]),
        "n_layers": ("int", 2, 8),
        "patch_size": ("categorical", [2, 4, 7]),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("float_log", 1e-5, 1e-3),
        "batch_size": ("categorical", [64, 128]),
    },
    "pytorch.alexnet": {
        "dropout_fc": ("float", 0.2, 0.7),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "weight_decay": ("float_log", 1e-5, 1e-3),
        "batch_size": ("categorical", [64, 128, 256]),
    },
    # ------------------------------------------------------------------
    # Phase 5 — 2D PDE (FNO2d, U-Net)
    # ------------------------------------------------------------------
    "pytorch.fno2d": {
        "n_modes": ("categorical", [8, 12, 16, 20]),
        "hidden_channels": ("categorical", [16, 32, 64]),
        "n_layers": ("int", 2, 6),
        "learning_rate": ("float_log", 1e-4, 5e-3),
        "batch_size": ("categorical", [4, 8, 16]),
    },
    "pytorch.unet": {
        "base_channels": ("categorical", [16, 32, 64]),
        "depth": ("int", 2, 4),
        "learning_rate": ("float_log", 1e-4, 5e-3),
        "batch_size": ("categorical", [4, 8, 16]),
    },
    # ------------------------------------------------------------------
    # Phase 6 — VAE
    # ------------------------------------------------------------------
    "pytorch.vae": {
        "latent_dim": ("categorical", [8, 16, 32, 64]),
        "hidden_dim": ("categorical", [64, 128, 256]),
        "beta": ("float_log", 0.01, 10.0),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [64, 128, 256]),
    },
    # ------------------------------------------------------------------
    # Phase 7 — DDPM
    # ------------------------------------------------------------------
    "pytorch.ddpm": {
        "n_timesteps": ("categorical", [100, 200, 500]),
        "hidden_channels": ("categorical", [32, 64, 128]),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    # ------------------------------------------------------------------
    # Phase 8 — CGAN
    # ------------------------------------------------------------------
    "pytorch.cgan": {
        "latent_dim": ("categorical", [16, 32, 64]),
        "hidden_channels": ("categorical", [64, 128, 256]),
        "learning_rate_g": ("float_log", 1e-4, 1e-2),
        "learning_rate_d": ("float_log", 1e-4, 1e-2),
        "n_gen_layers": ("int", 2, 4),
        "n_disc_layers": ("int", 2, 4),
        "batch_size": ("categorical", [32, 64, 128]),
    },
}

# Primary metric to maximise/minimise per task type.
_PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "regression": ("test_r2", "maximize"),
    "classification": ("test_accuracy", "maximize"),
}

# Per-benchmark metric overrides (for benchmarks where task_type alone is insufficient).
_BENCHMARK_PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "sequence.lorenz63": ("test_nrmse", "minimize"),
    "pde.burgers_1d": ("test_relative_l2", "minimize"),
    "pdebench.burgers_1d": ("test_relative_l2", "minimize"),
    "pdebench.darcy_2d": ("test_relative_l2", "minimize"),
    "pdebench.shallow_water_2d": ("test_relative_l2", "minimize"),
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

    # Auto-select metric + direction (per-benchmark override first).
    if metric is None:
        metric, default_dir = _BENCHMARK_PRIMARY_METRIC.get(
            benchmark_key, _PRIMARY_METRIC.get(task_type, ("test_r2", "maximize"))
        )
    else:
        _lower_metrics = {"test_rmse", "test_nrmse", "test_relative_l2", "runtime_s"}
        default_dir = "minimize" if metric in _lower_metrics else "maximize"
    if direction is None:
        direction = default_dir

    _PYTORCH_KEYS = {
        "pytorch.mlp", "pytorch.residual_mlp", "pytorch.mlp_classifier",
        "pytorch.cnn1d", "pytorch.lstm", "pytorch.gru",
        "pytorch.fno1d", "pytorch.deeponet",
        "pytorch.lenet5", "pytorch.resnet20", "pytorch.resnet56",
    }
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

    # Persist best HP to a human-readable cache file for leaderboard reuse.
    if save_root is not None and best_params:
        save_hpo_cache(benchmark_key, model_key, best_params, root=Path(save_root))

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


# ---------------------------------------------------------------------------
# HPO cache helpers
# ---------------------------------------------------------------------------

def _hpo_cache_path(benchmark_key: str, model_key: str, root: Path) -> Path:
    safe_bm = benchmark_key.replace(".", "_")
    safe_model = model_key.replace(".", "_")
    return root / "hpo_cache" / f"{safe_bm}__{safe_model}.json"


def save_hpo_cache(
    benchmark_key: str,
    model_key: str,
    best_params: dict,
    root: Path = Path("benchmark_reports"),
) -> Path:
    """Persist *best_params* to a JSON cache file.

    Returns the path where it was written.
    """
    import json

    path = _hpo_cache_path(benchmark_key, model_key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_key": benchmark_key,
        "model_key": model_key,
        "best_params": best_params,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_hpo_cache(
    benchmark_key: str,
    model_key: str,
    root: Path = Path("benchmark_reports"),
) -> dict | None:
    """Load cached HP for a (benchmark, model) pair.

    Returns ``None`` when no cache exists.
    """
    import json

    path = _hpo_cache_path(benchmark_key, model_key, root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("best_params")
    except Exception:
        return None
