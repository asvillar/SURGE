"""Tests for ``pytorch.dino_linear_probe`` (frozen DINOv2 + LogisticRegression).

No network access or real DINOv2 download happens here: ``hub_loader`` is
monkeypatched to a tiny random conv that mimics the backbone's interface
(``(N, 3, H, W) -> (N, D)``). This covers the adapter contract, feature
caching, and registry listing cheaply and deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from surge.model.registry import MODEL_REGISTRY  # noqa: E402


class _FakeBackbone(nn.Module):
    """Tiny random conv standing in for DINOv2 ViT-S/14 in tests."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.pool(self.conv(x)).flatten(1)


def _fake_loader() -> _FakeBackbone:
    return _FakeBackbone()


@pytest.fixture()
def img_data():
    """30 samples of 32x32x3 flat images, 10 classes (mirrors tests/test_models.py)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3072)).astype("float32")
    y = rng.integers(0, 10, 30)
    return X[:24], y[:24], X[24:], y[24:]


def _make_adapter(cache_path, **extra):
    return MODEL_REGISTRY.create(
        "pytorch.dino_linear_probe",
        hub_loader=_fake_loader,
        cache_path=str(cache_path),
        batch_size=8,
        device="cpu",
        max_iter=200,
        **extra,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_dino_linear_probe_registered():
    assert "pytorch.dino_linear_probe" in MODEL_REGISTRY


def test_dino_linear_probe_aliases_registered():
    assert "dino" in MODEL_REGISTRY
    assert "dinov2" in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Adapter contract: fit / predict / predict_proba
# ---------------------------------------------------------------------------


def test_fit_predict_shapes(tmp_path, img_data):
    X_tr, y_tr, X_te, _ = img_data
    adapter = _make_adapter(tmp_path / "feat.npz")
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape[0] == len(X_te)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_fit_accepts_X_val_y_val(tmp_path, img_data):
    """Registry/leaderboard path probes fit's signature for X_val/y_val."""
    import inspect

    X_tr, y_tr, X_te, y_te = img_data
    adapter = _make_adapter(tmp_path / "feat.npz")
    sig = inspect.signature(adapter.fit)
    assert "X_val" in sig.parameters and "y_val" in sig.parameters
    adapter.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert adapter.predict(X_te).shape == (len(X_te),)


def test_prepare_for_fit_tags_pretrained_backbone(tmp_path, img_data):
    X_tr, y_tr, _, _ = img_data
    adapter = _make_adapter(tmp_path / "feat.npz")
    info = adapter.prepare_for_fit(X_shape=X_tr.shape, y_shape=y_tr.shape)
    assert info["banner"]["pretrained_backbone"] == "dinov2_vits14"
    assert info["banner"]["probe"] == "sklearn.LogisticRegression"


# ---------------------------------------------------------------------------
# Feature cache round-trip
# ---------------------------------------------------------------------------


def test_feature_cache_round_trip(tmp_path, img_data):
    X_tr, y_tr, X_te, _ = img_data
    cache_path = tmp_path / "feat.npz"
    adapter = _make_adapter(cache_path)
    adapter.fit(X_tr, y_tr)
    assert cache_path.exists()

    # A fresh adapter instance reusing the same cache path must not need the
    # backbone again: break the loader and confirm predict() still works
    # because the exact-same X (same fingerprint) hits the cache.
    def _broken_loader():
        raise AssertionError("backbone should not be reloaded — cache miss")

    adapter2 = MODEL_REGISTRY.create(
        "pytorch.dino_linear_probe", hub_loader=_broken_loader,
        cache_path=str(cache_path), batch_size=8, device="cpu", max_iter=200,
    )
    # Reuse adapter's fitted classifier directly on cached features for X_tr.
    feats = adapter2._model._extract_features(X_tr)
    assert feats.shape[0] == len(X_tr)


def test_feature_cache_different_inputs_get_different_keys(tmp_path):
    cache_path = tmp_path / "feat.npz"
    adapter = _make_adapter(cache_path)
    rng = np.random.default_rng(0)
    X1 = rng.standard_normal((12, 3072)).astype("float32")
    X2 = rng.standard_normal((12, 3072)).astype("float32")
    f1 = adapter._model._extract_features(X1)
    f2 = adapter._model._extract_features(X2)
    assert f1.shape == f2.shape
    assert not np.allclose(f1, f2)
    with np.load(cache_path) as data:
        assert len(data.files) == 2


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path, img_data):
    X_tr, y_tr, X_te, _ = img_data
    cache_path = tmp_path / "feat.npz"
    adapter = _make_adapter(cache_path)
    adapter.fit(X_tr, y_tr)
    preds_before = adapter.predict(X_te)

    model_path = tmp_path / "model.joblib"
    adapter.save(model_path)

    adapter2 = _make_adapter(cache_path)
    adapter2.load(model_path)
    preds_after = adapter2.predict(X_te)

    assert np.array_equal(preds_before, preds_after)
