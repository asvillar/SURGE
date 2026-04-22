# Copyright (c) 2026, SURGE authors. Licensed under BSD-3-Clause.
"""
Tests for surge.model.profiling (T1.4 of docs/RELEASE_SPRINT.md).

Verifies that every engine run attaches a profile dict containing
model_size_bytes + inference_ms_per_sample + parameter_count, that the
numbers are finite and positive, and that the helper degrades gracefully
on unpicklable / failing adapters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surge.model.profiling import measure_model_profile


# ---------------------------------------------------------------------------
# Unit: measure_model_profile on a trained sklearn adapter
# ---------------------------------------------------------------------------
def test_measure_model_profile_on_trained_random_forest():
    from surge.model import MODEL_REGISTRY  # noqa: F401 - side-effect

    from surge import EngineRunConfig, ModelSpec, SurrogateEngine

    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y1": rng.normal(size=n),
            "y2": rng.normal(size=n),
        }
    )
    engine = SurrogateEngine(
        run_config=EngineRunConfig(test_fraction=0.2, val_fraction=0.1, random_state=0)
    )
    engine.configure_dataframe(df, input_columns=["x1", "x2"], output_columns=["y1", "y2"])
    results = engine.run([
        ModelSpec(key="sklearn.random_forest", params={"n_estimators": 10, "random_state": 0}),
    ])
    res = results[0]

    profile = res.extra.get("profile")
    assert profile is not None, "engine must attach profile to result.extra"

    # Size: pickled RandomForest is always > 0 bytes.
    assert isinstance(profile["model_size_bytes"], int)
    assert profile["model_size_bytes"] > 0

    # Parameter count: RF -> sum of tree node counts; must be a positive int.
    assert isinstance(profile["parameter_count"], int)
    assert profile["parameter_count"] > 0

    # Latency must be finite, > 0 ms/sample, and throughput the reciprocal.
    ms = profile["inference_ms_per_sample"]
    assert ms is not None and np.isfinite(ms) and ms > 0
    tput = profile["inference_throughput_samples_per_s"]
    assert tput is not None and np.isfinite(tput) and tput > 0
    assert abs(tput - 1000.0 / ms) < 1e-6

    assert profile["probe_trials"] >= 1
    assert profile["probe_batch_size"] >= 1


def test_measure_model_profile_returns_nones_on_failing_adapter():
    """Unpicklable / failing adapters must not raise: profile fields degrade to None."""

    class BrokenAdapter:
        name = "broken"

        class _Model:
            def predict(self, X):
                raise RuntimeError("boom")

        def __init__(self) -> None:
            # A lambda closure is unpicklable, which also exercises the
            # _pickled_size_bytes defensive branch.
            self._model = self._Model()
            self._model.closure = lambda x: x

        def predict(self, X):
            return self._model.predict(X)

    X = np.zeros((16, 3))
    profile = measure_model_profile(BrokenAdapter(), X, probe_trials=2)

    # Pickle of a closure-containing instance should fail -> None
    assert profile["model_size_bytes"] is None
    # predict raises -> timing gracefully None
    assert profile["inference_ms_per_sample"] is None
    assert profile["inference_throughput_samples_per_s"] is None
    # Parameter count not derivable -> None
    assert profile["parameter_count"] is None
    # Header fields still reported
    assert profile["probe_trials"] == 2


def test_measure_model_profile_handles_empty_sample():
    class NoopAdapter:
        name = "noop"
        _model = None  # pickled_size -> None

        def predict(self, X):  # pragma: no cover - shouldn't be called
            raise AssertionError("predict must not be called on empty sample")

    profile = measure_model_profile(NoopAdapter(), np.zeros((0, 3)), probe_trials=3)
    assert profile["inference_ms_per_sample"] is None
    assert profile["probe_batch_size"] is None
    assert profile["probe_trials"] == 3
