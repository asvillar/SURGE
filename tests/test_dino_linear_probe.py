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
        with torch.random.fork_rng():  # deterministic weights across instances
            torch.manual_seed(0)
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
        **{"extractor_id": "test-fake-conv", **extra},
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
        extractor_id="test-fake-conv",
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


# ---------------------------------------------------------------------------
# Fingerprint / cache identity
# ---------------------------------------------------------------------------


def test_fingerprint_covers_last_image_and_is_dtype_stable():
    from surge.model.backends.dino_linear_probe import _fingerprint

    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 3072)).astype("float32")
    X2 = X.copy()
    X2[-1, -1] += 1.0
    assert _fingerprint(X) != _fingerprint(X2)
    assert _fingerprint(X) == _fingerprint(X.astype("float64"))


def test_cache_identity_separates_resize(tmp_path):
    cache = tmp_path / "feat.npz"
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3072)).astype("float32")
    a16 = _make_adapter(cache, resize_to=16)
    a32 = _make_adapter(cache, resize_to=32)
    assert a16._model._cache_identity() != a32._model._cache_identity()
    a16._model._extract_features(X)
    a32._model._extract_features(X)
    with np.load(cache) as data:
        assert len(data.files) == 2
        assert all(f.startswith("feat_") for f in data.files)


def test_custom_loader_without_extractor_id_disables_disk_cache(tmp_path):
    cache = tmp_path / "feat.npz"
    calls = {"n": 0}

    class _Counting(_FakeBackbone):
        def forward(self, x):
            calls["n"] += 1
            return super().forward(x)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3072)).astype("float32")
    y = rng.integers(0, 3, 8)
    adapter = MODEL_REGISTRY.create(
        "pytorch.dino_linear_probe", hub_loader=_Counting, cache_path=str(cache),
        batch_size=8, device="cpu", max_iter=50,
    )
    adapter.fit(X, y)
    assert not cache.exists()
    n_after_fit = calls["n"]
    adapter.predict(X)
    assert calls["n"] == n_after_fit


def test_load_restores_saved_params(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 64)).astype("float32")
    y = rng.integers(0, 3, 20)
    kw = dict(img_size=8, in_channels=1, resize_to=16)
    a1 = _make_adapter(tmp_path / "c1.npz", **kw)
    a1.fit(X, y)
    expected = a1.predict(X)
    path = tmp_path / "m.joblib"
    a1.save(path)

    a2 = _make_adapter(tmp_path / "c2.npz")
    a2.load(path)
    m = a2._model
    assert (m.img_size, m.in_channels, m.resize_to) == (8, 1, 16)
    assert np.array_equal(a2.predict(X), expected)


def test_load_model_compat_reconstructs_dino(tmp_path, monkeypatch):
    from surge.io.load_compat import load_model_compat

    from surge.model.backends import dino_linear_probe as backend

    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path / "data"))
    # The artifact names its extractor; the compat loader must resolve that
    # id to the matching backbone, never to the default DINOv2 hub model.
    monkeypatch.setitem(backend._EXTRACTOR_LOADERS, "test-fake-conv", _fake_loader)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 64)).astype("float32")
    y = rng.integers(0, 3, 20)
    a1 = _make_adapter(tmp_path / "c1.npz", img_size=8, in_channels=1, resize_to=16)
    a1.fit(X, y)
    path = tmp_path / "m.joblib"
    a1.save(path)

    model = load_model_compat(
        str(path), {"backend": "pytorch", "name": "pytorch.dino_linear_probe"}
    )
    assert hasattr(model, "predict")
    assert model._model._hub_loader is _fake_loader
    assert np.array_equal(model.predict(X), a1.predict(X))


def test_load_model_compat_rejects_unregistered_extractor(tmp_path, monkeypatch):
    from surge.io.load_compat import load_model_compat
    from surge.model.backends import dino_linear_probe as backend

    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path / "data"))
    rng = np.random.default_rng(1)
    X = rng.standard_normal((16, 64)).astype("float32")
    y = rng.integers(0, 2, 16)
    a1 = _make_adapter(
        tmp_path / "c.npz", img_size=8, in_channels=1, resize_to=16,
        extractor_id="unregistered-custom-backbone",
    )
    a1.fit(X, y)
    path = tmp_path / "m.joblib"
    a1.save(path)
    assert "unregistered-custom-backbone" not in backend._EXTRACTOR_LOADERS

    with pytest.raises(ValueError, match="unregistered-custom-backbone"):
        load_model_compat(str(path), {"backend": "pytorch"})


def test_resolve_extractor_default_and_registered():
    from surge.model.backends import dino_linear_probe as backend

    assert backend.resolve_extractor(None) is backend.load_hub_backbone
    assert backend.resolve_extractor(backend.DEFAULT_EXTRACTOR_ID) is backend.load_hub_backbone
    backend.register_extractor("test-resolver-id", _fake_loader)
    try:
        assert backend.resolve_extractor("test-resolver-id") is _fake_loader
    finally:
        backend._EXTRACTOR_LOADERS.pop("test-resolver-id", None)
