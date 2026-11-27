"""The custom-adapter template must work against the current base class and registry."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from surge.model.base import BaseModelAdapter
from surge.model.registry import MODEL_REGISTRY

TEMPLATE = Path(__file__).resolve().parents[2] / "examples" / "custom_cnn_adapter_template.py"


@pytest.fixture()
def template():
    spec = importlib.util.spec_from_file_location("_cnn_adapter_template", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    before_reg = dict(MODEL_REGISTRY._registry)
    before_alias = dict(MODEL_REGISTRY._alias_map)
    spec.loader.exec_module(mod)  # registers under mod.MODEL_KEY
    try:
        yield mod
    finally:
        MODEL_REGISTRY._registry.clear()
        MODEL_REGISTRY._registry.update(before_reg)
        MODEL_REGISTRY._alias_map.clear()
        MODEL_REGISTRY._alias_map.update(before_alias)


def test_registration_and_instantiation(template):
    assert template.MODEL_KEY in MODEL_REGISTRY
    assert "custom_cnn" in MODEL_REGISTRY
    assert MODEL_REGISTRY.get(template.MODEL_KEY).adapter_cls is template.CNNAdapter
    model = MODEL_REGISTRY.create(template.MODEL_KEY, channels=4, n_epochs=2)
    assert isinstance(model, BaseModelAdapter)
    assert model.get_estimator().channels == 4
    assert model.get_estimator().n_epochs == 2


def test_registry_clean_after_fixture():
    assert "custom.cnn" not in MODEL_REGISTRY


def test_fit_predict_save_load(template, tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    y = X[:, :2].sum(axis=1)
    model = MODEL_REGISTRY.create(template.MODEL_KEY, channels=4, n_epochs=3, batch_size=16)
    model.fit(X, y, X_val=X[:5], y_val=y[:5])
    pred = model.predict(X)
    assert pred.shape == (40, 1)
    assert np.isfinite(pred).all()
    assert len(model.training_history) == 3

    path = tmp_path / "cnn.pt"
    model.save(path)
    clone = MODEL_REGISTRY.create(template.MODEL_KEY, channels=4)
    clone.load(path)
    np.testing.assert_allclose(clone.predict(X), pred, rtol=1e-5, atol=1e-6)
