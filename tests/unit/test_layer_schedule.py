"""Tests for geometric / flexible MLP layer schedules."""

from __future__ import annotations

import numpy as np
import pytest

from surge.model.layer_schedule import (
    clamp_layer_widths,
    geometric_hidden_widths,
    resolve_hidden_layers,
)


def test_clamp_layer_widths():
    assert clamp_layer_widths([0, 2000, 50], max_width=1024) == [1, 1024, 50]


def test_geometric_hidden_widths_90_to_12():
    layers = geometric_hidden_widths(90, 12, 4, max_width=1024)
    assert len(layers) == 4
    assert all(1 <= w <= 1024 for w in layers)
    # Monotonic decrease toward output dim 12
    assert layers[0] > layers[-1]


def test_geometric_ratio_consistency():
    widths = geometric_hidden_widths(100, 25, 3, min_width=1, max_width=10000)
    # r = (25/100)^(1/4) = 0.707...
    assert widths[0] == pytest.approx(71, abs=2)
    assert widths[2] == pytest.approx(35, abs=2)


def test_resolve_explicit_arbitrary():
    out = resolve_hidden_layers(
        n_in=10,
        n_out=3,
        schedule="explicit",
        hidden_layers=[2, 139, 205, 125],
        max_width=1024,
    )
    assert out == [2, 139, 205, 125]


def test_resolve_geometric():
    out = resolve_hidden_layers(
        n_in=90,
        n_out=12,
        schedule="geometric",
        n_hidden_layers=5,
        max_width=512,
    )
    assert len(out) == 5


@pytest.mark.skipif(
    not __import__("surge.model.pytorch", fromlist=["PYTORCH_AVAILABLE"]).PYTORCH_AVAILABLE,
    reason="PyTorch not installed",
)
def test_geom_residual_mlp_fit_smoke():
    from surge.model import create_model

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 20))
    y = rng.standard_normal((200, 4))
    model = create_model(
        "pytorch.geom_residual_mlp",
        n_hidden_layers=3,
        max_hidden_width=256,
        n_epochs=2,
        verbose=False,
    )
    model.fit(X, y)
    pred = model.predict(X[:10])
    assert pred.shape == (10, 4)
    assert hasattr(model._model, "_resolved_hidden_layers")
    assert len(model._model._resolved_hidden_layers) == 3


@pytest.mark.skipif(
    not __import__("surge.model.pytorch", fromlist=["PYTORCH_AVAILABLE"]).PYTORCH_AVAILABLE,
    reason="PyTorch not installed",
)
def test_residual_mlp_explicit_custom_widths():
    from surge.model import create_model

    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 8))
    y = rng.standard_normal(100)
    model = create_model(
        "pytorch.residual_mlp",
        hidden_layers=[302, 230, 510, 24, 125, 20],
        max_hidden_width=1024,
        n_epochs=2,
        verbose=False,
    )
    model.fit(X, y)
    assert model._model._resolved_hidden_layers == [302, 230, 510, 24, 125, 20]
