"""Tests for the Keras adapter and PCA characterization (both optional-dep aware)."""
from __future__ import annotations

import numpy as np
import pytest

from surge.preprocessing import pca_summary
from tests.conftest import optional_backend

# ---------------------------------------------------------------- PCA


def test_pca_summary_recovers_low_rank_structure():
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((400, 2))
    mix = rng.standard_normal((2, 6))
    X = latent @ mix + 0.01 * rng.standard_normal((400, 6))

    out = pca_summary(X)
    assert out["n_components_90"] <= 2
    assert out["explained_variance_ratio"][0] > 0.3
    assert np.isclose(out["cumulative_variance"][-1], 1.0, atol=1e-6)
    assert out["scores"].shape == (400, 2)
    assert set(out["top_features"]) <= {"PC1", "PC2", "PC3"}


def test_pca_summary_dataframe_and_names():
    import pandas as pd

    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    out = pca_summary(df, n_components=2)
    assert out["feature_names"] == ["a", "b", "c"]
    assert len(out["explained_variance_ratio"]) == 2
    assert out["components"].shape == (2, 3)


def test_pca_summary_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pca_summary(np.zeros(5))


# ------------------------------------------------------------- Keras


def _toy_regression(n=160, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = X @ np.arange(1, d + 1, dtype=np.float32) + 0.05 * rng.standard_normal(n)
    return X, y.astype(np.float32)


def test_keras_registered_or_transparently_skipped():
    from surge.model import registration_report

    recs = {r.key: r for r in registration_report()}
    assert "keras.mlp" in recs
    rec = recs["keras.mlp"]
    assert rec.status in ("registered", "skipped")
    if rec.status == "skipped":
        assert "tensorflow" in rec.reason


def test_keras_mlp_fit_predict_save_load(tmp_path):
    optional_backend("tensorflow")
    from surge.model import MODEL_REGISTRY

    X, y = _toy_regression()
    adapter = MODEL_REGISTRY.create(
        "keras.mlp", hidden_layers=(16,), epochs=60, batch_size=32,
        random_state=0, early_stopping_patience=0, validation_fraction=0.0)
    adapter.fit(X, y)
    pred = adapter.predict(X)
    assert pred.shape == y.shape
    # near-linear toy target: a small MLP must correlate strongly
    assert np.corrcoef(pred, y)[0, 1] > 0.9

    path = tmp_path / "model"
    adapter.save(path)
    reloaded = MODEL_REGISTRY.create("keras.mlp").load(path)
    np.testing.assert_allclose(reloaded.predict(X), pred, rtol=1e-5, atol=1e-5)


def test_keras_custom_build_fn():
    optional_backend("tensorflow")
    from surge.model.adapters.keras import KerasMLPAdapter, _keras

    keras = _keras()

    def build(n_in, n_out):
        m = keras.Sequential([
            keras.layers.Input(shape=(n_in,)),
            keras.layers.Dense(8, activation="tanh"),
            keras.layers.Dense(n_out),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    X, y = _toy_regression(n=80, d=3, seed=1)
    adapter = KerasMLPAdapter(build_fn=build, epochs=20, verbose=0,
                              early_stopping_patience=0,
                              validation_fraction=0.0)
    adapter.fit(X, y)
    assert adapter.predict(X).shape == y.shape


def test_pod_roundtrip_and_shapes():
    """Full-rank POD reconstructs exactly; truncation degrades gracefully."""
    import numpy as np

    from surge.preprocessing import pod_fit, pod_inverse, pod_transform

    rng = np.random.default_rng(0)
    # rank-5 fields + noise, 40 samples x 60 dof
    U = rng.standard_normal((40, 5))
    V = rng.standard_normal((5, 60))
    F = U @ V + 0.01 * rng.standard_normal((40, 60))

    full = pod_fit(F, n_modes=39)
    rec = pod_inverse(full, pod_transform(full, F))
    assert np.allclose(rec, F, atol=1e-8)

    b16 = pod_fit(F, n_modes=16)
    assert b16["components"].shape == (16, 60)
    coeffs = pod_transform(b16, F)
    assert coeffs.shape == (40, 16)
    err16 = np.linalg.norm(pod_inverse(b16, coeffs) - F)
    b2 = pod_fit(F, n_modes=2)
    err2 = np.linalg.norm(pod_inverse(b2, pod_transform(b2, F)) - F)
    assert err16 < err2                       # more modes -> better recon
    assert b16["explained_variance_ratio"][:5].sum() > 0.95

    with __import__("pytest").raises(ValueError):
        pod_fit(F, n_modes=0)
