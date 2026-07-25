"""Smoke tests for the GPflow adapters (optional TensorFlow stack).

First-ever verification of gpflow.gpr / gpflow.multi_kernel: these were
registered but untested since their introduction. Skips (never fails)
without the [gpflow] extra installed.
"""
from __future__ import annotations

import numpy as np

from tests.conftest import optional_backend


def _toy(n=60, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.standard_normal(n)
    return X, y


def test_gpflow_gpr_fit_predict_uncertainty():
    optional_backend("tensorflow")
    optional_backend("gpflow")
    from surge.model import MODEL_REGISTRY

    X, y = _toy()
    adapter = MODEL_REGISTRY.create("gpflow.gpr")
    adapter.fit(X, y)
    pred = np.asarray(adapter.predict(X)).ravel()
    assert pred.shape == (len(y),)
    assert np.corrcoef(pred, y)[0, 1] > 0.9

    out = adapter.predict_with_uncertainty(X)
    if isinstance(out, tuple):          # current adapters return (mean, var)
        mean, var = out
    else:                               # future UQResult-style mapping
        mean, var = out["mean"], out.get("variance", out.get("std"))
    mean = np.asarray(mean).ravel()
    var = np.asarray(var).ravel()
    assert mean.shape == (len(y),)
    assert np.all(np.asarray(var) >= 0)


def test_gpflow_registered_or_transparent():
    from surge.model import registration_report

    recs = {r.key: r for r in registration_report()}
    # gpflow adapters are core-registered (lazy TF import), so they must
    # always be present in the registry
    from surge.model import list_models
    assert "gpflow.gpr" in list_models()
    assert "gpflow.multi_kernel" in list_models()
    assert recs["gpflow.gpr"].status == "registered"
