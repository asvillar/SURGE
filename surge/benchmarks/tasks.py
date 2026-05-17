"""Tier 0 and Tier 1 benchmark implementations (CPU, sklearn-only)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import BenchmarkResult

# Default model keys per benchmark (used when the caller passes no model_key).
_DEFAULTS: dict[str, str] = {
    "synthetic.regression_1d": "sklearn.random_forest",
    "synthetic.classification_binary": "sklearn.random_forest_classifier",
    "tabular.diabetes": "sklearn.random_forest",
    "tabular.california_housing": "sklearn.random_forest",
    "tabular.concrete_strength": "sklearn.random_forest",
    "tabular.iris": "sklearn.random_forest_classifier",
    "tabular.breast_cancer": "sklearn.random_forest_classifier",
    "tabular.wine": "sklearn.random_forest_classifier",
    "tabular.digits": "sklearn.random_forest_classifier",
}

# Which task type each benchmark expects of its model.
_TASK_TYPE: dict[str, str] = {
    "synthetic.regression_1d": "regression",
    "synthetic.classification_binary": "classification",
    "tabular.diabetes": "regression",
    "tabular.california_housing": "regression",
    "tabular.concrete_strength": "regression",
    "tabular.iris": "classification",
    "tabular.breast_cancer": "classification",
    "tabular.wine": "classification",
    "tabular.digits": "classification",
}


def _resolve_model(model_key: str | None, benchmark_key: str) -> Any:
    """Return a fitted-ready adapter instance from MODEL_REGISTRY."""
    from surge.model.registry import MODEL_REGISTRY

    key = model_key or _DEFAULTS.get(benchmark_key)
    if key is None:
        raise ValueError(f"No default model for {benchmark_key!r}. Pass --model KEY.")
    return MODEL_REGISTRY.create(key)


def _fit_predict_regression(adapter: Any, X_train, y_train, X_test):
    """Fit adapter and return predictions. Returns (y_pred, elapsed_s)."""
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test)).ravel()
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed


def _fit_predict_classification(adapter: Any, X_train, y_train, X_test):
    """Fit adapter, return (y_pred_labels, y_proba_or_None, elapsed_s)."""
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = adapter.predict(X_test)
    y_prob = None
    if hasattr(adapter, "predict_proba"):
        try:
            y_prob = adapter.predict_proba(X_test)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    return y_pred, y_prob, elapsed


def _clf_metrics(y_test, y_pred, y_prob) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    metrics: dict[str, float] = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score

            n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
            if n_classes == 2:
                scores = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics["test_auroc"] = float(roc_auc_score(y_test, scores))
            else:
                metrics["test_auroc"] = float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics


def _reg_metrics(y_test, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_squared_error, r2_score

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"test_r2": r2, "test_rmse": rmse}


# ---------------------------------------------------------------------------
# Tier 0
# ---------------------------------------------------------------------------


def run_synthetic_regression_1d(*, seed: int = 42, n_samples: int = 400, model_key: str | None = None) -> BenchmarkResult:
    """Hermetic 1→1 regression (Tier 0)."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = 3.0 * X.ravel() + 1.5 + 0.15 * rng.standard_normal(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    adapter = _resolve_model(model_key, "synthetic.regression_1d")
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="synthetic.regression_1d",
        model_key=adapter.name,
        tier="0",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.85,
        message="Linear 1-D signal with small Gaussian noise",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_synthetic_classification_binary(*, seed: int = 42, n_samples: int = 500, model_key: str | None = None) -> BenchmarkResult:
    """20 → 2 binary labels (Tier 0)."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 20))
    logits = X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(n_samples)
    y = (logits > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    adapter = _resolve_model(model_key, "synthetic.classification_binary")
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="synthetic.classification_binary",
        model_key=adapter.name,
        tier="0",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] > 0.75,
        message="Random linear combo of first 3 features",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Tier 1 — regression
# ---------------------------------------------------------------------------


def run_tabular_diabetes(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """sklearn Diabetes dataset — 10→1 regression (Tier 1)."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.diabetes")
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.diabetes",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.35,
        message="UCI Diabetes / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_california_housing(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """California Housing dataset — 8→1 regression (Tier 1)."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.california_housing")
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.california_housing",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.75,
        message="California Housing / sklearn.datasets (Pace & Barry 1997)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_concrete_strength(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Concrete Compressive Strength — 8→1 regression (Tier 1)."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    data = fetch_openml(name="concrete-strength", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.concrete_strength")
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.concrete_strength",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.80,
        message="UCI Concrete Compressive Strength (Yeh 1998)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Tier 1 — classification
# ---------------------------------------------------------------------------


def run_tabular_iris(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Iris — 4→3 multiclass classification (Tier 1)."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    adapter = _resolve_model(model_key, "tabular.iris")
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.iris",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.88,
        message="UCI Iris / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_breast_cancer(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """Wisconsin Breast Cancer — 30→2 binary classification (Tier 1)."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    adapter = _resolve_model(model_key, "tabular.breast_cancer")
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.breast_cancer",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.93,
        message="Wisconsin Breast Cancer (UCI / WDBC)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_wine(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Wine — 13→3 multiclass classification (Tier 1)."""
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    adapter = _resolve_model(model_key, "tabular.wine")
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.wine",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.90,
        message="UCI Wine / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_digits(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """Optical digits — 64→10 multiclass classification (Tier 1)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    adapter = _resolve_model(model_key, "tabular.digits")
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.digits",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.95,
        message="Optical digits / sklearn.datasets (Alpaydin 1998)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )
