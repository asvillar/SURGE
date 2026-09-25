"""Tests for the ConStellaration closed-form boundary geometry loader and the
physics-agnostic input-features registry that wires it into ``surge bench``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from surge.benchmarks.features import (
    apply_input_features,
    list_input_features,
    register_input_features,
)
from surge.benchmarks.loaders.constellaration import (
    GEOMETRY_FEATURE_NAMES,
    geometry_features,
    geometry_metrics,
    mode_table,
)

_NPZ_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "datasets" / "constellaration" / "paper_nfp3_clip0.05.npz"
)


def _load_cached_rows(n: int = 300, seed: int = 0):
    data = np.load(_NPZ_PATH, allow_pickle=True)
    X, Y = data["X"], data["Y"]
    metric_names = [str(n) for n in data["metric_names"]]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], Y[idx], metric_names


# ---------------------------------------------------------------------------
# (a) analytic aspect_ratio vs. dataset label
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NPZ_PATH.exists(), reason="constellaration npz cache not present")
def test_aspect_ratio_matches_dataset_label():
    X, Y, metric_names = _load_cached_rows(n=300)
    ar_idx = metric_names.index("aspect_ratio")
    y_label = Y[:, ar_idx]

    metrics = geometry_metrics(X)
    ar_pred = metrics["aspect_ratio"]

    r = np.corrcoef(ar_pred, y_label)[0, 1]
    rel_err = np.abs(ar_pred - y_label) / np.abs(y_label)

    assert r > 0.9999, f"Pearson r={r} too low"
    assert np.median(rel_err) < 0.005, f"median rel. err={np.median(rel_err)} too high"


# ---------------------------------------------------------------------------
# grid independence + full-dataset timing
#
# The boundary is band-limited (m<=4, n<=4), so a uniform-grid rectangle
# rule integrates the theta/phi integrands used here exactly once the grid
# exceeds their bandwidth — geometry_metrics should give (near-)identical
# answers at the default 32x24 grid and at a much denser 128x96 grid.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NPZ_PATH.exists(), reason="constellaration npz cache not present")
def test_geometry_metrics_is_grid_independent():
    X, _Y, _names = _load_cached_rows(n=200, seed=1)

    default = geometry_metrics(X)  # default 32x24 grid
    dense = geometry_metrics(X, n_theta=128, n_phi=96)

    np.testing.assert_allclose(default["aspect_ratio"], dense["aspect_ratio"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(default["volume"], dense["volume"], atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not _NPZ_PATH.exists(), reason="constellaration npz cache not present")
def test_geometry_features_full_dataset_under_60s():
    data = np.load(_NPZ_PATH, allow_pickle=True)
    X = data["X"]
    assert len(X) > 20_000  # sanity: this is really the full ~27k-row cache

    t0 = time.perf_counter()
    F = geometry_features(X)
    elapsed = time.perf_counter() - t0

    assert F.shape == (len(X), len(GEOMETRY_FEATURE_NAMES))
    assert elapsed < 60.0, f"geometry_features on {len(X)} rows took {elapsed:.1f}s (budget: 60s)"


# ---------------------------------------------------------------------------
# (b) mode_table index mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NPZ_PATH.exists(), reason="constellaration npz cache not present")
def test_mode_table_column_4_is_m0_n0():
    modes = mode_table()
    assert len(modes) == 90
    assert modes[4] == (0, 0)

    data = np.load(_NPZ_PATH, allow_pickle=True)
    r_cos_4 = data["X"][:, 4]
    # r_cos_4 is the normalized major-radius (m=0, n=0) term — close to 1.0
    # for every row, though not exactly 1.0 in the clipped/filtered cache
    # (observed range ~[0.96, 1.00] in paper_nfp3_clip0.05.npz).
    assert np.all(np.isclose(r_cos_4, 1.0, atol=0.05))
    assert np.median(r_cos_4) == pytest.approx(1.0, abs=1e-3)


def test_mode_table_shape_without_data():
    # Pure index-math check, no dataset required.
    modes = mode_table()
    assert modes[0] == (0, -4)
    assert modes[8] == (0, 4)
    assert modes[9] == (1, -4)
    # z_sin block (columns 45..89) mirrors the r_cos block's (m, n) mapping.
    assert modes[45 + 4] == (0, 0)


# ---------------------------------------------------------------------------
# (c) features registry
# ---------------------------------------------------------------------------


def test_apply_input_features_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown input-feature transform"):
        apply_input_features("not_a_real_transform", np.zeros((5, 90)), "plasma.constellaration_multioutput")


def test_stellarator_geometry_appends_k_columns():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 90))
    X[:, 4] = 1.0  # keep the major-radius term near its normalized value

    X_out = apply_input_features("stellarator_geometry", X, "plasma.constellaration_multioutput")

    assert X_out.shape == (4, 90 + len(GEOMETRY_FEATURE_NAMES))
    np.testing.assert_allclose(X_out[:, :90], X)
    np.testing.assert_allclose(X_out[:, 90:], geometry_features(X))


def test_stellarator_geometry_disallowed_benchmark_raises():
    with pytest.raises(ValueError, match="not allowed for benchmark"):
        apply_input_features("stellarator_geometry", np.zeros((3, 90)), "tabular.iris")


def test_register_input_features_and_list():
    marker = []

    def _noop(X):
        marker.append(True)
        return X

    register_input_features("_test_noop_transform", _noop, benchmarks=None)
    assert "_test_noop_transform" in list_input_features()

    X = np.ones((2, 3))
    out = apply_input_features("_test_noop_transform", X, "anything.at.all")
    assert marker == [True]
    np.testing.assert_array_equal(out, X)


# ---------------------------------------------------------------------------
# (d) CLI smoke test: --input-features is parsed
# ---------------------------------------------------------------------------


def test_cli_parses_input_features_flag(capsys):
    from surge.benchmarks.run import main

    # --list short-circuits before the flag is used, so this only exercises
    # argparse acceptance of --input-features.
    code = main(["--list", "--input-features", "stellarator_geometry"])
    assert code == 0
    out = capsys.readouterr().out
    assert "plasma.constellaration" in out or "tabular.iris" in out
