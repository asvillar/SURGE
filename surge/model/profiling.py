# Copyright (c) 2026, SURGE authors. Licensed under BSD-3-Clause.
"""
Lightweight model profiling for SURGE runs.

Every trained model carries a small "profile" dict into ``ModelRunResult.extra``
and onward into ``workflow_summary.json`` / ``metrics.json``:

    {
      "model_size_bytes": 12345,
      "parameter_count": 6890,                # None if the adapter can't report it
      "inference_ms_per_sample": 0.034,
      "inference_throughput_samples_per_s": 29411.8,
      "probe_batch_size": 128,
      "probe_trials": 5
    }

This gives every release-grade run an honest answer to "how big and how fast
is this surrogate?" without requiring a separate benchmarking pass. Failures
(unpicklable estimator, predict() errors, empty sample) are caught and
reduced to ``None`` so profiling never blocks a training run.
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np

LOGGER = logging.getLogger("surge.model.profiling")


def _count_parameters(adapter: Any) -> Optional[int]:
    """Best-effort parameter count.

    * PyTorch modules: sum over ``parameters()``.
    * sklearn RandomForest: sum over ``tree_.node_count`` per estimator
      (upper bound; each internal tree node is one split).
    * sklearn MLP: sum sizes of ``coefs_`` + ``intercepts_``.
    * Others: ``None`` (no universal definition).
    """
    model = getattr(adapter, "_model", None)
    if model is None:
        return None

    # PyTorch path
    try:
        import torch.nn as nn  # noqa: F401

        # The MLP implementation lives at adapter._model._model (impl holds the nn.Module)
        for candidate in (getattr(model, "_model", None), getattr(model, "module", None), model):
            if candidate is None:
                continue
            if hasattr(candidate, "parameters") and callable(candidate.parameters):
                try:
                    params = list(candidate.parameters())
                except TypeError:
                    continue
                if params and all(hasattr(p, "numel") for p in params):
                    return int(sum(p.numel() for p in params))
    except ImportError:
        pass

    # sklearn RandomForest
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        try:
            return int(sum(getattr(est, "tree_").node_count for est in estimators))
        except Exception:  # noqa: BLE001
            pass

    # sklearn MLPRegressor
    coefs = getattr(model, "coefs_", None)
    if coefs is not None:
        try:
            n = sum(int(np.asarray(c).size) for c in coefs)
            intercepts = getattr(model, "intercepts_", []) or []
            n += sum(int(np.asarray(b).size) for b in intercepts)
            return n
        except Exception:  # noqa: BLE001
            pass

    return None


def _pickled_size_bytes(adapter: Any) -> Optional[int]:
    """Return the pickled size of the underlying estimator, or ``None`` if
    the estimator is unpicklable (e.g. some GPflow objects with TF state).
    """
    model = getattr(adapter, "_model", None)
    if model is None:
        return None
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("pickle of adapter %r failed: %s", getattr(adapter, "name", "?"), exc)
        return None


def measure_model_profile(
    adapter: Any,
    X_sample: np.ndarray,
    *,
    probe_batch_size: int = 128,
    probe_trials: int = 5,
) -> Dict[str, Any]:
    """Measure model size and inference latency on a small sample.

    ``X_sample`` must match the feature layout the adapter was trained on
    (i.e. already scaled by the engine's input scaler when applicable).
    The function is defensive: any individual measurement that fails is
    reported as ``None`` rather than raising.

    Parameters
    ----------
    adapter
        Fitted ``BaseModelAdapter``.
    X_sample
        2-D array of validation/test features to time predictions on.
    probe_batch_size
        Capped at ``len(X_sample)``. Default 128 — large enough to amortize
        per-call overhead, small enough to finish quickly.
    probe_trials
        Number of independent timing runs; the median is reported.
    """
    profile: Dict[str, Any] = {
        "model_size_bytes": _pickled_size_bytes(adapter),
        "parameter_count": _count_parameters(adapter),
        "inference_ms_per_sample": None,
        "inference_throughput_samples_per_s": None,
        "probe_batch_size": None,
        "probe_trials": probe_trials,
    }

    X = np.asarray(X_sample)
    if X.ndim != 2 or X.shape[0] == 0:
        return profile

    n = int(min(probe_batch_size, X.shape[0]))
    probe = X[:n]
    profile["probe_batch_size"] = n

    # Warmup (import, graph compile, kernel launch, ...)
    try:
        adapter.predict(probe[:1])
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("profiling warmup failed: %s", exc)
        return profile

    trials_ms: list[float] = []
    for _ in range(max(1, probe_trials)):
        try:
            t0 = time.perf_counter()
            adapter.predict(probe)
            elapsed = time.perf_counter() - t0
            trials_ms.append(1000.0 * elapsed / n)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("profiling trial failed: %s", exc)
            continue

    if trials_ms:
        median_ms = float(np.median(trials_ms))
        profile["inference_ms_per_sample"] = median_ms
        if median_ms > 0:
            profile["inference_throughput_samples_per_s"] = 1000.0 / median_ms

    return profile


__all__ = ["measure_model_profile"]
