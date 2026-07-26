"""All model adapter tests.

Covers: sklearn baselines, BoTorch GP, FT-Transformer, KAN, Vision,
PDE/operator models, Generative models (VAE/DDPM/CGAN), MLP Ensemble.

Each section:
  1. Registration check
  2. fit/predict shape test (tiny data, ≤3 epochs)
  3. predict_with_uncertainty where available
  4. edge-case / error tests where relevant

Per-test `pytest.importorskip` keeps sklearn tests runnable when torch
is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import optional_backend

from surge.model.registry import MODEL_REGISTRY


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def reg_data():
    """60-train / 20-test 1-D regression."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6)).astype("float32")
    y = X[:, 0] * 2 - X[:, 1] + 0.1 * rng.standard_normal(80)
    return X[:60], y[:60].astype("float32"), X[60:], y[60:].astype("float32")


@pytest.fixture()
def clf_data():
    """75-train / 25-test binary classification."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 6)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:75], y[:75], X[75:], y[75:]


@pytest.fixture()
def multiout_data():
    """60-train / 20-test 2-column regression."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 6)).astype("float32")
    A = np.array([[1, -1], [0.5, 0.5], [-1, 0], [0, 1], [0.3, -0.3], [0.2, 0.8]])
    Y = (X @ A + 0.05 * rng.standard_normal((80, 2))).astype("float32")
    return X[:60], Y[:60], X[60:], Y[60:]


# ── Sklearn baselines (Group C) ───────────────────────────────────────────────


def test_ridge_registered():
    assert "sklearn.ridge" in MODEL_REGISTRY


def test_ridge_fit_predict(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create("sklearn.ridge")
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_lgbm_regressor_registered():
    optional_backend("lightgbm")
    assert "lgbm.regressor" in MODEL_REGISTRY


def test_lgbm_regressor_fit_predict(reg_data):
    optional_backend("lightgbm")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create("lgbm.regressor", n_estimators=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_lgbm_classifier_registered():
    optional_backend("lightgbm")
    assert "lgbm.classifier" in MODEL_REGISTRY


def test_lgbm_classifier_fit_predict_proba(clf_data):
    optional_backend("lightgbm")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create("lgbm.classifier", n_estimators=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_catboost_regressor_registered():
    pytest.importorskip("catboost")
    assert "catboost.regressor" in MODEL_REGISTRY


def test_catboost_regressor_fit_predict(reg_data):
    pytest.importorskip("catboost")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create("catboost.regressor", iterations=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_catboost_classifier_registered():
    pytest.importorskip("catboost")
    assert "catboost.classifier" in MODEL_REGISTRY


def test_catboost_classifier_fit_predict_proba(clf_data):
    pytest.importorskip("catboost")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create("catboost.classifier", iterations=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 2)


# ── BoTorch GP (Group D) ──────────────────────────────────────────────────────


def test_botorch_gp_registered():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    assert "botorch.gp" in MODEL_REGISTRY


def test_botorch_gp_fit_predict():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype("float32")
    y = (X[:, 0] * 2).astype("float32")
    X_tr, y_tr, X_te = X[:24], y[:24], X[24:]

    adapter = MODEL_REGISTRY.create("botorch.gp", n_train_iter=5)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (6,)
    assert np.isfinite(preds).all()


def test_botorch_gp_predict_with_uncertainty():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype("float32")
    y = (X[:, 0] * 2).astype("float32")
    X_tr, y_tr, X_te = X[:24], y[:24], X[24:]

    adapter = MODEL_REGISTRY.create("botorch.gp", n_train_iter=5)
    adapter.fit(X_tr, y_tr)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert mean.shape == (6,)
    assert std.shape == (6,)
    assert (std >= 0).all()


def test_botorch_gp_size_guard():
    """ExactGP raises ValueError for n > 5000."""
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6001, 4)).astype("float32")
    y = rng.standard_normal(6001).astype("float32")
    adapter = MODEL_REGISTRY.create("botorch.gp")
    with pytest.raises(ValueError):
        adapter.fit(X, y)


def test_botorch_sparse_gp_registered():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    assert "botorch.sparse_gp" in MODEL_REGISTRY


def test_botorch_sparse_gp_fit_predict():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4)).astype("float32")
    y = (X[:, 0]).astype("float32")
    X_tr, y_tr, X_te = X[:60], y[:60], X[60:]
    adapter = MODEL_REGISTRY.create(
        "botorch.sparse_gp", n_train_iter=5, n_inducing=10
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (20,)


# ── FT-Transformer (Group E) ──────────────────────────────────────────────────


def test_ft_transformer_registered():
    pytest.importorskip("torch")
    assert "pytorch.ft_transformer" in MODEL_REGISTRY


def test_ft_transformer_classifier_registered():
    pytest.importorskip("torch")
    assert "pytorch.ft_transformer_classifier" in MODEL_REGISTRY


def test_ft_transformer_fit_predict(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.ft_transformer",
        n_epochs=2, d_model=16, n_heads=2, n_layers=1, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_ft_transformer_classifier_fit_predict_proba(clf_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.ft_transformer_classifier",
        n_epochs=2, d_model=16, n_heads=2, n_layers=1, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── KAN (Group E) ─────────────────────────────────────────────────────────────


def test_kan_registered():
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    assert "pytorch.kan" in MODEL_REGISTRY


def test_kan_classifier_registered():
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    assert "pytorch.kan_classifier" in MODEL_REGISTRY


def test_kan_fit_predict(reg_data):
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.kan",
        n_epochs=2, hidden_dims=[16, 16], grid_size=3, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_kan_classifier_fit_predict(clf_data):
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.kan_classifier",
        n_epochs=2, hidden_dims=[16, 16], grid_size=3, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset({0, 1})


# ── Vision models (Group F) ───────────────────────────────────────────────────


@pytest.fixture()
def img_data():
    """30 samples of 32×32×3 flat images, 10 classes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3072)).astype("float32")
    y = rng.integers(0, 10, 30)
    return X[:24], y[:24], X[24:], y[24:]


def test_alexnet_registered():
    pytest.importorskip("torch")
    assert "pytorch.alexnet" in MODEL_REGISTRY


def test_alexnet_fit_predict(img_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.alexnet", n_epochs=1, n_classes=10, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset(set(range(10)))
    assert proba.shape == (len(X_te), 10)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_vit_registered():
    pytest.importorskip("torch")
    assert "pytorch.vit" in MODEL_REGISTRY


def test_vit_fit_predict(img_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.vit",
        n_epochs=1, n_classes=10, d_model=16, n_heads=2, n_layers=1,
        patch_size=4, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 10)


def test_resnet20_registered():
    pytest.importorskip("torch")
    assert "pytorch.resnet20" in MODEL_REGISTRY


def test_resnet20_fit_predict(img_data):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.resnet20", n_epochs=1, n_classes=10, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 10)


def test_resnet56_registered():
    pytest.importorskip("torch")
    assert "pytorch.resnet56" in MODEL_REGISTRY


def test_resnet56_fit_predict(img_data):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.resnet56", n_epochs=1, n_classes=10, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


# ── PDE / operator models (Group G) ──────────────────────────────────────────


def test_fno2d_registered():
    pytest.importorskip("torch")
    assert "pytorch.fno2d" in MODEL_REGISTRY


def test_fno2d_fit_predict():
    """FNO2D on 2D spatial fields (B, nx, ny)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8, 8)).astype("float32")
    y = X + 0.1 * rng.standard_normal((20, 8, 8)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:16], y[:16], X[16:], y[16:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.fno2d",
        n_epochs=2, hidden_channels=8, n_modes=4, n_layers=1, batch_size=4,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape[0] == len(X_te)
    assert np.isfinite(preds).all()


def test_unet_registered():
    pytest.importorskip("torch")
    assert "pytorch.unet" in MODEL_REGISTRY


def test_unet_fit_predict():
    """U-Net on 2D fields (B, C, H, W)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 1, 16, 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((20, 1, 16, 16)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:16], y[:16], X[16:], y[16:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.unet",
        n_epochs=2, base_channels=8, depth=2, batch_size=4,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape[0] == len(X_te)
    assert np.isfinite(preds).all()


def test_unet_flat_input():
    """U-Net accepts flat (B, nx*ny) fields like FNO-2D — same 2D contract."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 16 * 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((20, 16 * 16)).astype("float32")

    adapter = MODEL_REGISTRY.create(
        "pytorch.unet",
        n_epochs=2, base_channels=8, depth=2, batch_size=4,
    )
    adapter.fit(X[:16], y[:16])
    preds = adapter.predict(X[16:])
    assert preds.shape[0] == 4
    assert np.isfinite(preds).all()

    with pytest.raises(ValueError, match="square"):
        adapter.predict(rng.standard_normal((2, 15)).astype("float32"))


# ── Generative / conditional models (Group H) ─────────────────────────────────


def test_vae_registered():
    pytest.importorskip("torch")
    assert "pytorch.vae" in MODEL_REGISTRY


def test_vae_fit_predict(reg_data):
    """VAE: fit(X, y) → predict(X) returns ŷ (regression surrogate)."""
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.vae",
        latent_dim=4, hidden_dim=16, n_epochs=2, batch_size=8, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_vae_predict_with_uncertainty(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.vae",
        latent_dim=4, hidden_dim=16, n_epochs=2, batch_size=8, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert mean.shape == (len(X_te),)
    assert std.shape == (len(X_te),)


def test_ddpm_registered():
    pytest.importorskip("torch")
    assert "pytorch.ddpm" in MODEL_REGISTRY


def test_ddpm_fit_predict():
    """DDPM: conditional field-to-field diffusion model."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((40, 16)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:30], y[:30], X[30:], y[30:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.ddpm",
        n_timesteps=5, hidden_channels=8, n_epochs=2, batch_size=8, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == y_te.shape
    assert np.isfinite(preds).all()


def test_cgan_registered():
    pytest.importorskip("torch")
    assert "pytorch.cgan" in MODEL_REGISTRY


def test_cgan_fit_predict():
    """CGAN: conditional GAN field-to-field surrogate."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((40, 16)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:30], y[:30], X[30:], y[30:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.cgan",
        latent_dim=4, hidden_channels=16, n_gen_layers=2, n_disc_layers=2,
        n_epochs=2, batch_size=8, n_predict_samples=2,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == y_te.shape
    assert np.isfinite(preds).all()


# ── MLP Ensemble (Group I) ────────────────────────────────────────────────────


def test_mlp_ensemble_registered():
    pytest.importorskip("torch")
    assert "pytorch.mlp_ensemble" in MODEL_REGISTRY


def test_mlp_ensemble_fit_predict(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble",
        n_ensembles=2, n_epochs=3, hidden_dim=16, n_layers=2,
        batch_size=16, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_mlp_ensemble_predict_with_uncertainty(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble",
        n_ensembles=2, n_epochs=3, hidden_dim=16, n_layers=2,
        batch_size=16, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert mean.shape == (len(X_te),)
    assert std.shape == (len(X_te),)
    assert (std >= 0).all()


def test_mlp_ensemble_multioutput(multiout_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = multiout_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble",
        n_ensembles=2, n_epochs=3, hidden_dim=16, n_layers=2,
        batch_size=16, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert preds.shape == (len(X_te), 2)
    assert mean.shape == (len(X_te), 2)
    assert std.shape == (len(X_te), 2)


# ── Benchmark smoke tests ─────────────────────────────────────────────────────


def test_ridge_benchmark_smoke():
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "tabular.california_housing", model_key="sklearn.ridge", seed=0
    )
    assert result is not None
    assert "test_r2" in result.metrics


def test_lgbm_classifier_benchmark_smoke():
    optional_backend("lightgbm")
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark("tabular.iris", model_key="lgbm.classifier", seed=0)
    assert result is not None
    assert "test_accuracy" in result.metrics


@pytest.mark.slow
def test_ft_transformer_benchmark_smoke():
    pytest.importorskip("torch")
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "tabular.diabetes", model_key="pytorch.ft_transformer", seed=0
    )
    assert result is not None
    assert "test_r2" in result.metrics
