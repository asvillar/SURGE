"""Physics-agnostic registry of named input-feature transforms.

A transform maps a benchmark's raw input matrix ``X`` to an augmented or
otherwise modified ``X`` (e.g. appending closed-form geometry features for a
benchmark whose inputs are Fourier boundary coefficients). The core stays
physics-agnostic: this module only knows how to look a transform up by name
and gate it by benchmark key. Domain-specific transform *implementations*
(e.g. :func:`surge.benchmarks.loaders.constellaration.geometry_features`)
live with their loaders and are registered here, not reimplemented.

Usage::

    from surge.benchmarks.features import apply_input_features
    X = apply_input_features("stellarator_geometry", X, "plasma.constellaration_multioutput")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

InputFeatureFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _Entry:
    fn: InputFeatureFn
    benchmarks: tuple[str, ...] | None  # None = allowed for every benchmark


_REGISTRY: dict[str, _Entry] = {}


def register_input_features(
    name: str,
    fn: InputFeatureFn,
    benchmarks: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Register a named input-feature transform.

    Parameters
    ----------
    name:
        Lookup key, e.g. ``"stellarator_geometry"``.
    fn:
        ``X -> X'`` callable, ``X`` a ``(n, d)`` array.
    benchmarks:
        Benchmark-key *prefixes* this transform is allowed for (matched with
        ``benchmark_key.startswith(prefix)``). ``None`` allows every
        benchmark.
    """
    _REGISTRY[name] = _Entry(fn=fn, benchmarks=tuple(benchmarks) if benchmarks else None)


def list_input_features() -> list[str]:
    """Return the sorted list of registered transform names."""
    return sorted(_REGISTRY)


def _is_allowed(entry: _Entry, benchmark_key: str) -> bool:
    if entry.benchmarks is None:
        return True
    return any(benchmark_key.startswith(prefix) for prefix in entry.benchmarks)


def apply_input_features(name: str, X: np.ndarray, benchmark_key: str) -> np.ndarray:
    """Apply a registered input-feature transform to ``X``.

    Raises
    ------
    ValueError
        If ``name`` is not registered, or is registered but not allowed for
        ``benchmark_key``.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown input-feature transform {name!r}. "
            f"Available: {list_input_features()}"
        )
    if not _is_allowed(entry, benchmark_key):
        raise ValueError(
            f"Input-feature transform {name!r} is not allowed for benchmark "
            f"{benchmark_key!r} (allowed prefixes: {list(entry.benchmarks or [])})"
        )
    return entry.fn(np.asarray(X))


def _stellarator_geometry(X: np.ndarray) -> np.ndarray:
    from .loaders.constellaration import geometry_features

    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X, geometry_features(X)], axis=1)


register_input_features(
    "stellarator_geometry",
    _stellarator_geometry,
    benchmarks=["plasma.constellaration"],
)
