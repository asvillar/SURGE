"""Tier 0 and Tier 1 benchmark implementations (CPU, sklearn-only)."""

from __future__ import annotations

import time

import numpy as np

from .base import BenchmarkResult


def run_synthetic_regression_1d(*, seed: int = 42, n_samples: int = 400) -> BenchmarkResult:
    """Hermetic 1→1 regression (Tier 0)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = 3.0 * X.ravel() + 1.5 + 0.15 * rng.standard_normal(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    t0 = time.perf_counter()
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    r2 = float(r2_score(y_test, y_pred))
    return BenchmarkResult(
        benchmark_key="synthetic.regression_1d",
        tier="0",
        task_type="regression",
        metrics={"test_r2": r2, "runtime_s": float(elapsed)},
        passed=r2 > 0.85,
        message="Linear 1-D signal with small Gaussian noise",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_synthetic_classification_binary(*, seed: int = 42, n_samples: int = 500) -> BenchmarkResult:
    """20 → 2 binary labels (Tier 0)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 20))
    logits = X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(n_samples)
    y = (logits > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    t0 = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=30, max_depth=8, random_state=seed
    ).fit(X_train, y_train)
    pred = clf.predict(X_test)
    elapsed = time.perf_counter() - t0
    acc = float(accuracy_score(y_test, pred))
    return BenchmarkResult(
        benchmark_key="synthetic.classification_binary",
        tier="0",
        task_type="classification",
        metrics={"test_accuracy": acc, "runtime_s": float(elapsed)},
        passed=acc > 0.75,
        message="Random linear combo of first 3 features",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_iris(*, seed: int = 42) -> BenchmarkResult:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    t0 = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=40, max_depth=6, random_state=seed
    ).fit(X_train, y_train)
    pred = clf.predict(X_test)
    elapsed = time.perf_counter() - t0
    acc = float(accuracy_score(y_test, pred))
    return BenchmarkResult(
        benchmark_key="tabular.iris",
        tier="1",
        task_type="classification",
        metrics={"test_accuracy": acc, "runtime_s": float(elapsed)},
        passed=acc >= 0.88,
        message="UCI Iris / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_diabetes(*, seed: int = 42) -> BenchmarkResult:
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    t0 = time.perf_counter()
    reg = RandomForestRegressor(
        n_estimators=80, max_depth=5, random_state=seed
    ).fit(X_train, y_train)
    pred = reg.predict(X_test)
    elapsed = time.perf_counter() - t0
    r2 = float(r2_score(y_test, pred))
    return BenchmarkResult(
        benchmark_key="tabular.diabetes",
        tier="1",
        task_type="regression",
        metrics={"test_r2": r2, "runtime_s": float(elapsed)},
        passed=r2 > 0.35,
        message="UCI Diabetes / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )
