"""Dispatch table for standard benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BenchmarkResult
from .tasks import (
    run_synthetic_classification_binary,
    run_synthetic_regression_1d,
    run_tabular_diabetes,
    run_tabular_iris,
)

# Keys follow benchmark plan taxonomy (Tier 0 / Tier 1 subset).
REGISTRY: dict[str, Callable[..., BenchmarkResult]] = {
    "synthetic.regression_1d": run_synthetic_regression_1d,
    "synthetic.classification_binary": run_synthetic_classification_binary,
    "tabular.iris": run_tabular_iris,
    "tabular.diabetes": run_tabular_diabetes,
}


def list_benchmarks() -> list[str]:
    return sorted(REGISTRY)


def run_benchmark(key: str, **kwargs: Any) -> BenchmarkResult:
    """
    Run a registered benchmark by key.

    Parameters
    ----------
    key:
        e.g. ``synthetic.regression_1d``, ``tabular.iris``.
    **kwargs:
        Passed to the underlying task (e.g. ``seed=``).
    """
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown benchmark {key!r}. Choose one of: {', '.join(list_benchmarks())}"
        )
    return REGISTRY[key](**kwargs)
