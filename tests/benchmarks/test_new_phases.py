"""Smoke tests for Phases C, D1, D2, E, F, G additions.

These tests verify registration, shapes, and quick benchmark runs.  They
do NOT train full models or download large datasets — they only check that
the infrastructure is wired correctly.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed (optional dependency)")

# ---------------------------------------------------------------------------
# Phase C — Temporal adapters
# ---------------------------------------------------------------------------


@requires_torch
def test_pytorch_cnn1d_registered():
    from surge.model import list_models
    assert "pytorch.cnn1d" in list_models()


@requires_torch
def test_pytorch_lstm_registered():
    from surge.model import list_models
    assert "pytorch.lstm" in list_models()


@requires_torch
def test_pytorch_gru_registered():
    from surge.model import list_models
    assert "pytorch.gru" in list_models()


@pytest.mark.parametrize("model_key", ["pytorch.lstm", "pytorch.gru"])
@requires_torch
def test_rnn_fit_predict(model_key):
    from surge.model.registry import MODEL_REGISTRY

    n_state, T_in, T_out = 3, 10, 5
    n_samples = 80
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, T_in * n_state))
    y = rng.standard_normal((n_samples, T_out * n_state))

    adapter = MODEL_REGISTRY.create(model_key, n_epochs=3, batch_size=32)
    adapter.fit(X, y)
    y_pred = adapter.predict(X[:10])
    assert y_pred.shape == (10, T_out * n_state)


@requires_torch
def test_cnn1d_fit_predict():
    from surge.model.registry import MODEL_REGISTRY

    n_x = 32
    n_samples = 60
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n_samples, n_x))
    y = rng.standard_normal((n_samples, n_x))

    adapter = MODEL_REGISTRY.create("pytorch.cnn1d", n_epochs=3, batch_size=16)
    adapter.fit(X, y)
    y_pred = adapter.predict(X[:10])
    assert y_pred.shape == (10, n_x)


def test_lorenz63_benchmark_registered():
    from surge.benchmarks.registry import list_benchmarks
    assert "sequence.lorenz63" in list_benchmarks()


def test_lorenz63_benchmark_runs():
    """Smoke-sized run of the upgraded Lyapunov-time rollout protocol."""
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "sequence.lorenz63", seed=42, model_key="sklearn.ridge",
        n_trajectories=80, n_eval_ics=5, rollout_lyapunov_times=1.0,
    )
    assert result.benchmark_key == "sequence.lorenz63"
    assert "test_nrmse" in result.metrics
    assert result.metrics["test_nrmse"] < 1.0
    for key in ("test_vpt_median", "test_vpt_p25", "test_vpt_p75", "test_climate_error"):
        assert key in result.metrics
    assert 0.0 <= result.metrics["test_vpt_median"] <= 1.0  # rollout_lyapunov_times=1.0
    assert result.metrics["test_climate_error"] >= 0.0
    assert len(result.extra["per_ic_vpt"]) == 5
    assert result.extra["n_rollout_steps"] > 0


def test_lorenz_rk4_integrator():
    from surge.benchmarks.tasks import _lorenz_rk4_step

    state = np.array([1.0, 1.0, 1.0])
    state_new = _lorenz_rk4_step(state, dt=0.01)
    assert state_new.shape == (3,)
    assert not np.allclose(state, state_new)


# ---------------------------------------------------------------------------
# Phase D1 — FNO + DeepONet + Burgers
# ---------------------------------------------------------------------------


@requires_torch
def test_pytorch_fno1d_registered():
    from surge.model import list_models
    assert "pytorch.fno1d" in list_models()


@requires_torch
def test_pytorch_deeponet_registered():
    from surge.model import list_models
    assert "pytorch.deeponet" in list_models()


@requires_torch
def test_fno1d_fit_predict():
    from surge.model.registry import MODEL_REGISTRY

    n_x = 16
    n_samples = 40
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n_samples, n_x))
    y = rng.standard_normal((n_samples, n_x))

    adapter = MODEL_REGISTRY.create("pytorch.fno1d", n_epochs=3, batch_size=16, n_modes=4)
    adapter.fit(X, y)
    y_pred = adapter.predict(X[:8])
    assert y_pred.shape == (8, n_x)


@requires_torch
def test_deeponet_fit_predict():
    from surge.model.registry import MODEL_REGISTRY

    n_sensors = 16
    n_query = 16
    n_samples = 40
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n_samples, n_sensors))
    y = rng.standard_normal((n_samples, n_query))

    adapter = MODEL_REGISTRY.create("pytorch.deeponet", n_epochs=3, batch_size=16)
    adapter.fit(X, y)
    y_pred = adapter.predict(X[:8])
    assert y_pred.shape == (8, n_query)


def test_burgers_benchmark_registered():
    from surge.benchmarks.registry import list_benchmarks
    assert "pde.burgers_1d" in list_benchmarks()


def test_burgers_inline_solver():
    from surge.benchmarks.tasks import _burgers_solve_fd

    n_x = 64
    import numpy as np
    u0 = np.sin(np.linspace(0, 2 * np.pi, n_x, endpoint=False))
    u_T = _burgers_solve_fd(u0, n_x=n_x, dt=1e-4, nt=10, nu=0.01)
    assert u_T.shape == (n_x,)
    assert np.isfinite(u_T).all()


def test_relative_l2_metric():
    from surge.metrics import relative_l2

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert relative_l2(y_true, y_pred) == pytest.approx(0.0)

    y_pred_bad = y_true * 2
    assert relative_l2(y_true, y_pred_bad) > 0.5


# ---------------------------------------------------------------------------
# Phase D2 — PDEBench loader (import/guard only, no download)
# ---------------------------------------------------------------------------


def test_pdebench_loader_import():
    from surge.benchmarks.loaders.pdebench import (
        H5PY_AVAILABLE,
        list_pdebench_datasets,
    )
    datasets = list_pdebench_datasets()
    assert "burgers_1d" in datasets
    assert "darcy_2d" in datasets
    assert "shallow_water_2d" in datasets


def test_pdebench_benchmarks_registered():
    from surge.benchmarks.registry import list_benchmarks
    keys = list_benchmarks()
    assert "pdebench.burgers_1d" in keys
    assert "pdebench.darcy_2d" in keys
    assert "pdebench.shallow_water_2d" in keys


def test_pdebench_raises_without_h5py_or_file():
    """Trying to load without the HDF5 file should raise cleanly."""
    from surge.benchmarks.loaders.pdebench import H5PY_AVAILABLE, load_pdebench

    if not H5PY_AVAILABLE:
        with pytest.raises(ImportError, match="h5py"):
            load_pdebench("burgers_1d", download=False)
    else:
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_pdebench("burgers_1d", download=False, cache_dir="/nonexistent/path")


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("h5py"),
    reason="h5py not installed",
)
def test_pdebench_burgers_loader_synthetic(tmp_path):
    """Burgers 1D loader parses synthetic HDF5 correctly (no download)."""
    import h5py
    from surge.benchmarks.loaders.pdebench import _load_burgers_hdf5

    N, T, nx = 50, 11, 64
    rng = np.random.default_rng(0)
    tensor = rng.random((N, T, nx), dtype=np.float32)

    fpath = tmp_path / "1D_Burgers_test.hdf5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("tensor", data=tensor)
        f.create_dataset("x-coordinate", data=np.linspace(0, 1, nx, dtype=np.float32))

    X_tr, y_tr, X_te, y_te = _load_burgers_hdf5(fpath, n_train=40, n_test=10, seed=42)

    assert X_tr.shape == (40, nx), f"X_train shape mismatch: {X_tr.shape}"
    assert y_tr.shape == (40, nx)
    assert X_te.shape == (10, nx)
    assert y_te.shape == (10, nx)
    # IC and final state should differ (drawn independently)
    assert not np.allclose(X_tr, y_tr)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("h5py"),
    reason="h5py not installed",
)
def test_pdebench_darcy_loader_synthetic(tmp_path):
    """Darcy 2D loader parses synthetic HDF5 correctly (no download)."""
    import h5py
    from surge.benchmarks.loaders.pdebench import _load_darcy_hdf5

    N, nx, ny = 50, 16, 16
    rng = np.random.default_rng(1)
    nu_data = rng.random((N, nx, ny), dtype=np.float32)
    tensor_data = rng.random((N, 1, nx, ny), dtype=np.float32)  # (N, nt=1, nx, ny)

    fpath = tmp_path / "2D_Darcy_test.hdf5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("nu", data=nu_data)
        f.create_dataset("tensor", data=tensor_data)
        f.create_dataset("x-coordinate", data=np.linspace(0, 1, nx, dtype=np.float32))
        f.create_dataset("y-coordinate", data=np.linspace(0, 1, ny, dtype=np.float32))

    X_tr, y_tr, X_te, y_te = _load_darcy_hdf5(fpath, n_train=40, n_test=10, seed=42)

    assert X_tr.shape == (40, nx * ny), f"X_train shape mismatch: {X_tr.shape}"
    assert y_tr.shape == (40, nx * ny)
    assert X_te.shape == (10, nx * ny)
    assert y_te.shape == (10, nx * ny)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("h5py"),
    reason="h5py not installed",
)
def test_pdebench_shallow_water_loader_synthetic(tmp_path):
    """Shallow Water 2D loader parses per-trajectory HDF5 groups correctly."""
    import h5py
    from surge.benchmarks.loaders.pdebench import _load_shallow_water_hdf5

    T, nx, ny, nc = 8, 16, 16, 2
    n_traj = 50
    rng = np.random.default_rng(2)

    fpath = tmp_path / "2D_rdb_test.h5"
    with h5py.File(fpath, "w") as f:
        for i in range(1, n_traj + 1):
            key = f"{i:04d}"
            grp = f.create_group(key)
            grp.create_dataset("data", data=rng.random((T, nx, ny, nc), dtype=np.float32))
            grid = grp.create_group("grid")
            grid.create_dataset("x", data=np.linspace(0, 1, nx, dtype=np.float32))
            grid.create_dataset("y", data=np.linspace(0, 1, ny, dtype=np.float32))
            grid.create_dataset("t", data=np.linspace(0, 1, T, dtype=np.float32))

    X_tr, y_tr, X_te, y_te = _load_shallow_water_hdf5(fpath, n_train=40, n_test=10, seed=42)

    assert X_tr.shape == (40, nx * ny), f"X_train shape mismatch: {X_tr.shape}"
    assert y_tr.shape == (40, nx * ny)
    assert X_te.shape == (10, nx * ny)
    assert y_te.shape == (10, nx * ny)
    # IC ≠ final state
    assert not np.allclose(X_tr, y_tr)


# ---------------------------------------------------------------------------
# Phase F — Scientific benchmarks
# ---------------------------------------------------------------------------


def test_flow_regime_benchmark_registered():
    from surge.benchmarks.registry import list_benchmarks
    assert "classification.flow_regime" in list_benchmarks()


def test_flow_regime_benchmark_runs():
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "classification.flow_regime", seed=42,
        model_key="sklearn.random_forest_classifier"
    )
    assert result.benchmark_key == "classification.flow_regime"
    assert "test_accuracy" in result.metrics
    assert result.metrics["test_accuracy"] > 0.5


def test_airfoil_benchmark_registered():
    from surge.benchmarks.registry import list_benchmarks
    assert "tabular.airfoil_noise" in list_benchmarks()


def test_yacht_benchmark_registered():
    from surge.benchmarks.registry import list_benchmarks
    assert "tabular.yacht_dynamics" in list_benchmarks()


def test_plasma_stability_registered():
    from surge.benchmarks.registry import list_benchmarks, resolve_benchmark_key
    assert "tabular.plasma_stability" in list_benchmarks()
    # Legacy key still resolves after rename from classification.* to tabular.*
    assert resolve_benchmark_key("classification.plasma_stability") == "tabular.plasma_stability"


def test_m3dc1_benchmark_registered():
    from surge.benchmarks.registry import list_benchmarks
    assert "fusion.m3dc1_sample" in list_benchmarks()


def test_m3dc1_benchmark_runs_synthetic():
    """fusion.m3dc1_sample should work with the synthetic fallback."""
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "fusion.m3dc1_sample", seed=42, model_key="sklearn.random_forest"
    )
    assert result.benchmark_key == "fusion.m3dc1_sample"
    assert "test_r2" in result.metrics
    assert np.isfinite(result.metrics["test_r2"])


# ---------------------------------------------------------------------------
# Phase E — Vision adapters (registration + tiny synthetic test)
# ---------------------------------------------------------------------------


@requires_torch
def test_lenet5_registered():
    from surge.model import list_models
    assert "pytorch.lenet5" in list_models()


@requires_torch
def test_resnet20_registered():
    from surge.model import list_models
    assert "pytorch.resnet20" in list_models()


@requires_torch
def test_resnet56_registered():
    from surge.model import list_models
    assert "pytorch.resnet56" in list_models()


@requires_torch
def test_lenet5_fit_predict_synthetic():
    """Verify LeNet-5 can fit tiny synthetic image data."""
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(7)
    n_samples = 60
    # Flat MNIST-like input (28*28 = 784)
    X = rng.uniform(0.0, 1.0, (n_samples, 784)).astype("float32")
    y = rng.integers(0, 10, n_samples)

    adapter = MODEL_REGISTRY.create("pytorch.lenet5", n_epochs=2, batch_size=16)
    adapter.fit(X, y)
    y_pred = adapter.predict(X[:10])
    assert y_pred.shape == (10,)
    assert all(0 <= p < 10 for p in y_pred)


@requires_torch
def test_resnet20_fit_predict_synthetic():
    """Verify ResNet-20 can fit tiny synthetic CIFAR-like data."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(8)
    n_samples = 40
    # Flat CIFAR-like input (3*32*32 = 3072)
    X = rng.uniform(0.0, 1.0, (n_samples, 3072)).astype("float32")
    y = rng.integers(0, 10, n_samples)

    adapter = MODEL_REGISTRY.create("pytorch.resnet20", n_epochs=2, batch_size=16, n_classes=10)
    adapter.fit(X, y)
    y_pred = adapter.predict(X[:8])
    assert y_pred.shape == (8,)


def test_vision_benchmarks_registered():
    from surge.benchmarks.registry import list_benchmarks
    keys = list_benchmarks()
    assert "vision.mnist" in keys
    assert "vision.cifar10" in keys


# ---------------------------------------------------------------------------
# Phase G — TheWell (registration + guard check)
# ---------------------------------------------------------------------------


def test_thewell_benchmarks_registered():
    from surge.benchmarks.registry import list_benchmarks
    keys = list_benchmarks()
    assert "thewell.gray_scott" in keys
    assert "thewell.turbulence_2d" in keys
    assert "thewell.mhd" in keys


def test_thewell_loader_import():
    from surge.benchmarks.loaders.thewell import (
        THEWELL_AVAILABLE,
        list_thewell_datasets,
    )
    datasets = list_thewell_datasets()
    assert "gray_scott" in datasets
    assert "turbulence_2d" in datasets
    assert "mhd" in datasets


def test_thewell_raises_without_package():
    from surge.benchmarks.loaders.thewell import THEWELL_AVAILABLE, load_thewell

    if not THEWELL_AVAILABLE:
        with pytest.raises(ImportError, match="the_well"):
            load_thewell("gray_scott")
    else:
        pytest.skip("the_well is installed; skipping guard test")


# ---------------------------------------------------------------------------
# HPO new search spaces
# ---------------------------------------------------------------------------


def test_hpo_has_new_model_spaces():
    from surge.benchmarks.hpo import list_hpo_models

    models = list_hpo_models()
    assert "pytorch.cnn1d" in models
    assert "pytorch.lstm" in models
    assert "pytorch.gru" in models
    assert "pytorch.fno1d" in models
    assert "pytorch.deeponet" in models
    assert "pytorch.lenet5" in models
    assert "pytorch.resnet20" in models
    assert "pytorch.resnet56" in models
