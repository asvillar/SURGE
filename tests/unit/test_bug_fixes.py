"""Group A — bug-fix regression tests.

Verifies specific bugs that were fixed:
- CNN1D: scalar-target error message
- CNN1D: sequence-target fit works
- RNN: predict() returns (n,) not (n,1) for scalar targets
- HPC: detect_compute_resources returns correct ComputeResources
- Adapters: training_history is a property, not a method
"""

from __future__ import annotations

import numpy as np
import pytest


# ── CNN1D shape fix ────────────────────────────────────────────────────────────


def test_cnn1d_rejects_scalar_target():
    """CNN1DModel.fit(X, y_1d) must raise ValueError with helpful message."""
    pytest.importorskip("torch")
    from surge.model.backends.cnn import CNN1DModel

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8)).astype("float32")
    y = rng.standard_normal(20).astype("float32")  # 1-D scalar target

    model = CNN1DModel(n_epochs=1, hidden_channels=8, n_layers=1)
    with pytest.raises(ValueError, match="pytorch.mlp"):
        model.fit(X, y)


def test_cnn1d_accepts_sequence_target():
    """CNN1DModel.fit with X and y having same spatial dim (20 samples, 8 steps)."""
    pytest.importorskip("torch")
    from surge.model.backends.cnn import CNN1DModel

    rng = np.random.default_rng(0)
    n_samples, n_steps = 20, 8
    X = rng.standard_normal((n_samples, n_steps)).astype("float32")
    y = rng.standard_normal((n_samples, n_steps)).astype("float32")

    model = CNN1DModel(n_epochs=2, hidden_channels=8, n_layers=1, batch_size=8)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5, n_steps)
    assert np.isfinite(preds).all()


# ── RNN shape fix ─────────────────────────────────────────────────────────────


def test_lstm_1d_target_predict_shape():
    """LSTMModel.predict returns (n,) not (n,1) for T_out==1, n_state==1."""
    pytest.importorskip("torch")
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5)).astype("float32")  # 5 features = T_in*1
    y = rng.standard_normal((60, 1)).astype("float32")

    adapter = MODEL_REGISTRY.create(
        "pytorch.lstm", n_epochs=2, batch_size=16, n_state=1, T_in=5, T_out=1,
    )
    adapter.fit(X, y)
    preds = adapter.predict(X[:10])
    assert preds.ndim == 1, f"Expected 1D, got shape {preds.shape}"
    assert preds.shape == (10,)


def test_gru_1d_target_predict_shape():
    """GRUModel.predict returns (n,) not (n,1) for T_out==1, n_state==1."""
    pytest.importorskip("torch")
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5)).astype("float32")
    y = rng.standard_normal((60, 1)).astype("float32")

    adapter = MODEL_REGISTRY.create(
        "pytorch.gru", n_epochs=2, batch_size=16, n_state=1, T_in=5, T_out=1,
    )
    adapter.fit(X, y)
    preds = adapter.predict(X[:10])
    assert preds.ndim == 1, f"Expected 1D, got shape {preds.shape}"
    assert preds.shape == (10,)


# ── HPC resources ─────────────────────────────────────────────────────────────


def test_detect_compute_resources():
    from surge.hpc.resources import ComputeResources, detect_compute_resources

    res = detect_compute_resources()
    assert isinstance(res, ComputeResources)


def test_compute_resources_has_required_fields():
    from surge.hpc.resources import detect_compute_resources

    res = detect_compute_resources()
    assert isinstance(res.n_cpus, int) and res.n_cpus >= 1
    assert isinstance(res.n_gpus, int) and res.n_gpus >= 0
    assert res.device in ("cpu", "cuda")
    assert isinstance(res.hostname, str)
    # scheduler may be None on a laptop
    assert res.scheduler is None or isinstance(res.scheduler, str)


def test_compute_resources_instantiation():
    """ComputeResources can be constructed on any Python version (slots guard)."""
    from surge.hpc.resources import ComputeResources

    cr = ComputeResources(
        scheduler=None,
        n_cpus=4,
        n_gpus=0,
        gpu_type=None,
        device="cpu",
        hostname="testhost",
        extras={"platform": "test"},
    )
    assert cr.n_cpus == 4
    assert cr.device == "cpu"
    d = cr.to_dict()
    assert d["n_cpus"] == 4


# ── Adapter: training_history is a property ───────────────────────────────────


def test_residual_mlp_training_history_is_property():
    """training_history must be a list (property), not a callable."""
    pytest.importorskip("torch")
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4)).astype("float32")
    y = (X[:, 0] + 0.1 * rng.standard_normal(60)).astype("float32")

    adapter = MODEL_REGISTRY.create("pytorch.residual_mlp", n_epochs=3)
    adapter.fit(X, y)

    hist = adapter.training_history
    assert isinstance(hist, list), (
        f"training_history should be a list, got {type(hist)}"
    )
    assert len(hist) > 0
    assert "train_loss" in hist[0]


def test_mlp_classifier_training_history_is_property():
    """MLPClassifierAdapter.training_history must be a list (property)."""
    pytest.importorskip("torch")
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    adapter = MODEL_REGISTRY.create("pytorch.mlp_classifier", n_epochs=3)
    adapter.fit(X, y)

    hist = adapter.training_history
    assert isinstance(hist, list), (
        f"training_history should be a list, got {type(hist)}"
    )
    assert len(hist) > 0
