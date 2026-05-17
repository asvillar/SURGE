"""Dispatch table for standard benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BenchmarkResult
from .tasks import (
    run_synthetic_classification_binary,
    run_synthetic_multioutput_2d,
    run_synthetic_regression_1d,
    run_tabular_breast_cancer,
    run_tabular_california_housing,
    run_tabular_concrete_strength,
    run_tabular_diabetes,
    run_tabular_digits,
    run_tabular_energy_efficiency,
    run_tabular_iris,
    run_tabular_wine,
)

# ─── Benchmark metadata ───────────────────────────────────────────────────────
# Each entry: (runner_fn, tier, task_type, shape_desc, description)
_META: dict[str, tuple[Callable, str, str, str, str]] = {
    "synthetic.regression_1d": (
        run_synthetic_regression_1d, "0", "regression", "1→1",
        "Linear 1-D signal with Gaussian noise (inline fixture)",
    ),
    "synthetic.multioutput_2d": (
        run_synthetic_multioutput_2d, "0", "regression", "8→2",
        "Multi-output 8→2 linear regression with Gaussian noise (inline fixture)",
    ),
    "synthetic.classification_binary": (
        run_synthetic_classification_binary, "0", "classification", "20→2",
        "Binary labels from linear combo of features (inline fixture)",
    ),
    "tabular.diabetes": (
        run_tabular_diabetes, "1", "regression", "10→1",
        "UCI Diabetes / sklearn.datasets (Efron et al. 2004)",
    ),
    "tabular.california_housing": (
        run_tabular_california_housing, "1", "regression", "8→1",
        "California Housing / sklearn.datasets (Pace & Barry 1997)",
    ),
    "tabular.concrete_strength": (
        run_tabular_concrete_strength, "1", "regression", "8→1",
        "UCI Concrete Compressive Strength (Yeh 1998) [requires internet on first run]",
    ),
    "tabular.energy_efficiency": (
        run_tabular_energy_efficiency, "1", "regression", "8→1",
        "UCI Energy Efficiency — Heating Load (Tsanas & Xifara 2012) [requires internet on first run]",
    ),
    "tabular.iris": (
        run_tabular_iris, "1", "classification", "4→3",
        "UCI Iris / sklearn.datasets (Fisher 1936)",
    ),
    "tabular.breast_cancer": (
        run_tabular_breast_cancer, "1", "classification", "30→2",
        "Wisconsin Breast Cancer / sklearn.datasets (UCI WDBC)",
    ),
    "tabular.wine": (
        run_tabular_wine, "1", "classification", "13→3",
        "UCI Wine / sklearn.datasets",
    ),
    "tabular.digits": (
        run_tabular_digits, "1", "classification", "64→10",
        "Optical digits / sklearn.datasets (Alpaydin 1998)",
    ),
}

# Flat runner registry (key → callable) used by run_benchmark().
REGISTRY: dict[str, Callable[..., BenchmarkResult]] = {k: v[0] for k, v in _META.items()}


def list_benchmarks(*, tier: str | None = None, task_type: str | None = None) -> list[str]:
    """Return sorted benchmark keys, optionally filtered by tier or task_type."""
    keys = []
    for k, (_, t, tt, _, _) in _META.items():
        if tier is not None and t != tier:
            continue
        if task_type is not None and tt != task_type:
            continue
        keys.append(k)
    return sorted(keys)


def benchmark_info(key: str) -> dict[str, str]:
    """Return metadata dict for a registered benchmark key."""
    if key not in _META:
        raise KeyError(f"Unknown benchmark {key!r}. Use list_benchmarks().")
    _, tier, task_type, shape, description = _META[key]
    return {
        "key": key,
        "tier": tier,
        "task_type": task_type,
        "shape": shape,
        "description": description,
    }


def run_benchmark(key: str, **kwargs: Any) -> BenchmarkResult:
    """
    Run a registered benchmark by key.

    Parameters
    ----------
    key:
        e.g. ``synthetic.regression_1d``, ``tabular.iris``.
    **kwargs:
        Passed to the underlying task (e.g. ``seed=``, ``model_key=``).
    """
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown benchmark {key!r}. Choose one of: {', '.join(list_benchmarks())}"
        )
    return REGISTRY[key](**kwargs)
