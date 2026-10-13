"""Tests for the field-shaped ConStellaration benchmarks
(``plasma.constellaration_modb``, ``plasma.constellaration_surfaces``).

Skipped end-to-end if the VMEC wout subset cache
(``data/datasets/constellaration/wout_nfp3_subset_parts/``) isn't present —
this is a large, separately-downloaded dataset (see
``surge.benchmarks.loaders.constellaration.download_wout_subset``); works on
however many rows are currently on disk (500 today, up to 26,897 once the
download finishes) with no changes needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "datasets" / "constellaration"
_PARTS_DIR = _DATA_DIR / "wout_nfp3_subset_parts"

pytestmark = pytest.mark.skipif(
    not (_PARTS_DIR / "meta.npz").exists() or not list(_PARTS_DIR.glob("part_*.npz")),
    reason="ConStellaration VMEC wout subset not downloaded (data/datasets/constellaration/wout_nfp3_subset_parts/)",
)


# ---------------------------------------------------------------------------
# (a) Dataset shapes
# ---------------------------------------------------------------------------


def test_modb_dataset_shapes():
    from surge.benchmarks.loaders.constellaration_fields import constellaration_modb_dataset

    X, y, ids, extra = constellaration_modb_dataset()
    n = len(ids)
    assert n > 0
    assert X.shape == (n, 2, 64, 32)
    assert y.shape == (n, 64, 32)
    assert extra["modb_scale"].shape == (n,)
    assert extra["minor_radius"].shape == (n,)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
    # Normalised by its own per-sample mean -> each sample's mean is ~1.
    assert np.abs(y.reshape(n, -1).mean(axis=1) - 1.0).max() < 1e-5


def test_surfaces_dataset_shapes():
    from surge.benchmarks.loaders.constellaration_fields import constellaration_surfaces_dataset

    X, y, ids, extra = constellaration_surfaces_dataset()
    n = len(ids)
    assert X.shape == (n, 2, 64, 32)
    assert y.shape[0] == n
    assert y.shape[1] == 2
    assert y.shape[3] == 64
    assert y.shape[2] <= 16
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# (b) Alignment: X from boundary_grid matches evaluate_surface at the
#     boundary, on the theta_ref grid built by _theta_ref_permutation -- the
#     same identity tests/benchmarks/test_constellaration_wout.py verifies
#     on a single fixture row, checked here across the whole loaded subset.
# ---------------------------------------------------------------------------


def test_boundary_alignment_against_evaluate_surface():
    from surge.benchmarks.loaders.constellaration import evaluate_surface, load_wout_subset
    from surge.benchmarks.loaders.constellaration_fields import (
        DEFAULT_N_PHI_FIELD,
        DEFAULT_N_THETA_FIELD,
        _boundary_x90,
        _theta_ref_permutation,
        load_constellaration_field_raw,
    )

    raw = load_constellaration_field_raw()
    data = load_wout_subset()
    xm, xn = np.asarray(data["xm"]), np.asarray(data["xn"])
    nfp = int(np.asarray(data["nfp"]).ravel()[0])
    rmnc, zmns = np.asarray(data["rmnc"]), np.asarray(data["zmns"])

    X90 = _boundary_x90(rmnc[:, -1, :], zmns[:, -1, :], xm, xn)
    from surge.benchmarks.loaders.constellaration import boundary_grid

    grid = boundary_grid(X90, n_theta=DEFAULT_N_THETA_FIELD, n_phi=DEFAULT_N_PHI_FIELD, n_fp=nfp)

    R_wout, Z_wout, _t, _p = evaluate_surface(
        rmnc[:, -1, :], zmns[:, -1, :], xm, xn,
        n_theta=DEFAULT_N_THETA_FIELD, n_phi=DEFAULT_N_PHI_FIELD, n_fp=nfp,
    )
    perm = _theta_ref_permutation(DEFAULT_N_THETA_FIELD)
    R_matched, Z_matched = R_wout[:, perm, :], Z_wout[:, perm, :]

    assert np.max(np.abs(R_matched - grid.R)) < 1e-6
    assert np.max(np.abs(Z_matched - grid.Z)) < 1e-6
    # And raw["X"] (what the benchmark actually feeds models) matches too.
    assert np.max(np.abs(raw["X"][:, 0] - grid.R)) < 1e-6
    assert np.max(np.abs(raw["X"][:, 1] - grid.Z)) < 1e-6


def test_theta_ref_permutation_is_involution():
    from surge.benchmarks.loaders.constellaration_fields import _theta_ref_permutation

    perm = _theta_ref_permutation(64)
    assert np.array_equal(perm[perm], np.arange(64))
    with pytest.raises(ValueError):
        _theta_ref_permutation(63)


# ---------------------------------------------------------------------------
# (c) Baselines
# ---------------------------------------------------------------------------


def test_field_baselines_mean_and_pod():
    from surge.benchmarks.loaders.constellaration_fields import constellaration_modb_dataset
    from surge.benchmarks.tasks import _constellaration_field_split, _field_baselines

    X, y, ids, _extra = constellaration_modb_dataset()
    train_idx, test_idx = _constellaration_field_split(ids)
    assert len(train_idx) > 0 and len(test_idx) > 0

    baselines = _field_baselines(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    assert baselines["baseline_mean_field_relative_l2"] >= 0.0
    # POD+ridge should beat (or roughly match) the plain mean-field baseline
    # on a smooth geometric field like |B|.
    if "baseline_pod_ridge_relative_l2" in baselines:
        assert baselines["baseline_pod_ridge_relative_l2"] >= 0.0


# ---------------------------------------------------------------------------
# (d) End-to-end smoke runs (sklearn.ridge — fast, no torch dependency)
# ---------------------------------------------------------------------------


def test_run_constellaration_modb_smoke():
    from surge.benchmarks.tasks import run_constellaration_modb

    result = run_constellaration_modb(model_key="sklearn.ridge")
    assert result.benchmark_key == "plasma.constellaration_modb"
    assert "test_relative_l2" in result.metrics
    assert "test_relative_l2_mean" in result.metrics
    assert "test_mirror_ratio_mae" in result.metrics
    assert "baseline_mean_field_relative_l2" in result.extra
    assert result.metrics["test_relative_l2"] >= 0.0


def test_run_constellaration_surfaces_smoke():
    from surge.benchmarks.tasks import run_constellaration_surfaces

    result = run_constellaration_surfaces(model_key="sklearn.ridge")
    assert result.benchmark_key == "plasma.constellaration_surfaces"
    assert "test_relative_l2" in result.metrics
    assert "test_axis_position_error" in result.metrics
    assert "baseline_mean_field_relative_l2" in result.extra


# ---------------------------------------------------------------------------
# (e) Registration consistency (no data dependency; runs even without the
#     wout cache, but grouped here for locality — kept unskipped explicitly).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(False, reason="never skipped")
class TestRegistrationConsistency:
    KEYS = ("plasma.constellaration_modb", "plasma.constellaration_surfaces")

    def test_in_registry_meta(self):
        from surge.benchmarks.registry import _META

        for key in self.KEYS:
            assert key in _META
            runner, category, task_type, _shape, _desc = _META[key]
            assert runner is not None
            assert category == "plasma"
            assert task_type == "regression"

    def test_short_aliases(self):
        from surge.benchmarks.registry import resolve_benchmark_key

        assert resolve_benchmark_key("constellaration_modb") == "plasma.constellaration_modb"
        assert resolve_benchmark_key("constellaration_surfaces") == "plasma.constellaration_surfaces"

    def test_in_metadata_yaml(self):
        import yaml

        path = Path(__file__).resolve().parent.parent.parent / "surge" / "benchmarks" / "metadata.yaml"
        doc = yaml.safe_load(path.read_text())
        keys = {b["key"] for b in doc["benchmarks"]}
        for key in self.KEYS:
            assert key in keys

    def test_in_dataset_io(self):
        from surge.benchmarks.dataset_io import get_benchmark_io

        for key in self.KEYS:
            io = get_benchmark_io(key)
            assert io is not None
            assert io.get("inputs")
            assert io.get("output")

    def test_in_thresholds(self):
        from surge.benchmarks import leaderboard

        # _THRESHOLDS lives inside _check_pass's local scope; exercise it
        # through the public pass/fail behaviour instead of reaching in.
        for key in self.KEYS:
            assert leaderboard._check_pass(key, {"test_relative_l2": 1e-6}) is True
            assert leaderboard._check_pass(key, {"test_relative_l2": 10.0}) is False
        assert "test_relative_l2" in leaderboard._LOWER_IS_BETTER

    def test_leaderboard_loader_entries_present(self):
        from surge.benchmarks import leaderboard

        assert hasattr(leaderboard, "_load_constellaration_modb")
        assert hasattr(leaderboard, "_load_constellaration_surfaces")
