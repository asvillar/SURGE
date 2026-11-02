"""Kumo Tabular adapters (optional dep ``structured-data-models``)."""
from __future__ import annotations

import numpy as np
import pytest

try:
    pytest.importorskip("sdm")
except OSError:  # pragma: no cover - native lib broken
    pytest.skip("sdm native dependency broken", allow_module_level=True)

from surge.model.registry import MODEL_REGISTRY

pytestmark = pytest.mark.slow

KW = dict(size="small", device="cpu", num_estimators=1)


def _make(key, **kw):
    return MODEL_REGISTRY.create(key, **{**KW, **kw})


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (OSError, ConnectionError) as exc:  # Hub unreachable / weights missing
        pytest.skip(f"Kumo weights unavailable: {exc}")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = np.exp(0.5 * X[:, 0] + 0.1 * rng.normal(size=200))
    return X, y


def test_registration_and_aliases():
    for alias in ("kumo", "kumo_tabular", "kumo_regressor", "kumo.tabular_regressor"):
        assert type(MODEL_REGISTRY.create(alias, **KW)).name == "kumo.tabular_regressor"
    assert type(MODEL_REGISTRY.create("kumo_classifier", **KW)).name == "kumo.tabular_classifier"


def test_regression_1d_and_uncertainty(data):
    X, y = data
    m = _make("kumo.tabular_regressor")
    _guard(m.fit, X, y)
    p = m.predict(X[:20])
    assert p.shape == (20,) and np.isfinite(p).all()
    mean, std = m.predict_with_uncertainty(X[:20])
    assert mean.shape == std.shape == (20,) and (std >= 0).all()


def test_regression_2d(data):
    X, y = data
    Y = np.stack([y, -y], axis=1)
    m = _make("kumo.tabular_regressor")
    _guard(m.fit, X, Y)
    assert m.predict(X[:10]).shape == (10, 2)
    mean, std = m.predict_with_uncertainty(X[:10])
    assert mean.shape == std.shape == (10, 2) and (std >= 0).all()


def test_context_subsampling(data):
    X, y = data
    m = _make("kumo.tabular_regressor", max_context_rows=50)
    _guard(m.fit, X, y)
    assert len(m.get_estimator().X_context_) == 50
    assert np.isfinite(m.predict(X[:5])).all()


def test_log10_transform(data):
    X, y = data
    m = _make("kumo.tabular_regressor", target_transform="log10")
    _guard(m.fit, X, y)
    p = m.predict(X[:50])
    assert (p > 0).all()
    assert np.corrcoef(np.log(p), np.log(y[:50]))[0, 1] > 0.5
    with pytest.raises(ValueError):
        _make("kumo.tabular_regressor", target_transform="log10").fit(X, -y)


def test_save_load_identical(data, tmp_path):
    X, y = data
    m = _make("kumo.tabular_regressor")
    _guard(m.fit, X, y)
    ref = m.predict(X[:10])
    path = tmp_path / "kumo.npz"
    m.save(path)
    m2 = _make("kumo.tabular_regressor")
    m2.load(path)
    np.testing.assert_allclose(m2.predict(X[:10]), ref, rtol=1e-4, atol=1e-5)


def test_classification():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = np.digitize(X[:, 0], [-0.5, 0.5]) * 10  # labels 0, 10, 20
    m = _make("kumo.tabular_classifier")
    _guard(m.fit, X, y)
    pred = m.predict(X[:15])
    proba = m.predict_proba(X[:15])
    assert pred.shape == (15,) and set(np.unique(pred)) <= {0, 10, 20}
    assert proba.shape == (15, 3)
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-5)
    assert (pred == y[:15]).mean() > 0.6


def test_too_many_columns():
    m = _make("kumo.tabular_regressor")
    with pytest.raises(ValueError, match="100"):
        m.fit(np.zeros((10, 101)), np.zeros(10))
