"""Tests for the ConStellaration VMEC wout equilibrium subset loader.

Uses a tiny fixture (``tests/fixtures/constellaration_wout_row.json``) built
from one real downloaded ``vmecpp_wout`` row (already radially subsampled by
:func:`surge.benchmarks.loaders.constellaration._extract_wout_row`, so it is
small enough to check in), plus the matching 90-feature boundary
(``r_cos``/``z_sin``) and metrics pulled from the same row of the local
``paper_nfp3_clip0.05.npz`` cache (aligned by :func:`paper_filter_ids`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from surge.benchmarks.loaders.constellaration import (
    boundary_grid,
    evaluate_modB,
    evaluate_surface,
    wout_boundary_to_r_cos_z_sin,
)

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "constellaration_wout_row.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


pytestmark = pytest.mark.skipif(
    not _FIXTURE_PATH.exists(), reason="constellaration wout fixture not present"
)


# ---------------------------------------------------------------------------
# (a) JSON -> array extraction shapes
# ---------------------------------------------------------------------------


def test_fixture_shapes():
    d = _load_fixture()

    n_full = len(d["s_full"])
    n_half = len(d["s_half"])
    mnmax = len(d["xm"])
    mnmax_nyq = len(d["xm_nyq"])

    assert n_full <= 16
    assert len(d["xn"]) == mnmax
    assert len(d["xn_nyq"]) == mnmax_nyq

    rmnc = np.asarray(d["rmnc"])
    zmns = np.asarray(d["zmns"])
    lmns = np.asarray(d["lmns"])
    bmnc = np.asarray(d["bmnc"])
    assert rmnc.shape == (n_full, mnmax)
    assert zmns.shape == (n_full, mnmax)
    assert lmns.shape == (n_half, mnmax)  # lmns uses the regular mode set, not Nyquist
    assert bmnc.shape == (n_half, mnmax_nyq)

    ns = d["ns"]
    assert np.asarray(d["iota_full"]).shape == (ns,)
    assert np.asarray(d["pressure_full"]).shape == (ns,)
    assert np.asarray(d["phi"]).shape == (ns,)
    assert np.asarray(d["vp"]).shape == (ns - 1,)

    # Kept surfaces include the axis and the boundary.
    assert d["s_full"][0] == pytest.approx(0.0)
    assert d["s_full"][-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (b) evaluate_modB: max/min mirror ratio at the outermost kept surface is a
#     physically sensible value and is grid-resolution converged (a coarse
#     32x24 grid and a fine 64x48 grid must agree to within 5% -- otherwise
#     the Fourier sum would be aliased/under-resolved).
# ---------------------------------------------------------------------------


def test_evaluate_modb_mirror_ratio():
    d = _load_fixture()
    bmnc = np.asarray(d["bmnc"])
    xm_nyq = np.asarray(d["xm_nyq"])
    xn_nyq = np.asarray(d["xn_nyq"])
    nfp = d["nfp"]

    modB_lo, _t, _p = evaluate_modB(bmnc[-1], xm_nyq, xn_nyq, n_theta=32, n_phi=24, n_fp=nfp)
    modB_hi, _t, _p = evaluate_modB(bmnc[-1], xm_nyq, xn_nyq, n_theta=64, n_phi=48, n_fp=nfp)

    ratio_lo = modB_lo.max() / modB_lo.min()
    ratio_hi = modB_hi.max() / modB_hi.min()

    assert 1.0 < ratio_hi < 5.0, f"mirror ratio {ratio_hi} outside a sane stellarator range"
    rel_err = abs(ratio_lo - ratio_hi) / ratio_hi
    assert rel_err < 0.05, f"mirror ratio not grid-converged: {ratio_lo} vs {ratio_hi} (rel err {rel_err})"


# ---------------------------------------------------------------------------
# (c) evaluate_surface at s=1 (boundary) matches boundary_grid on the same
#     configuration's 90-feature (r_cos, z_sin) representation, to 1e-6.
#
# VMEC's own wout poloidal angle is gauge-related (not identical) to the
# angle underlying boundary.r_cos/z_sin: empirically, theta_ref = pi -
# theta_wout at the same phi (see wout_boundary_to_r_cos_z_sin's docstring).
# With n_theta even, that reflection maps evaluate_surface's regular theta
# grid exactly onto boundary_grid's, so the two can be compared pointwise
# after reindexing.
# ---------------------------------------------------------------------------


def test_wout_boundary_to_r_cos_z_sin_matches_dataset_boundary():
    d = _load_fixture()
    r_cos, z_sin = wout_boundary_to_r_cos_z_sin(
        np.asarray(d["rmnc"])[-1], np.asarray(d["zmns"])[-1], d["xm"], d["xn"]
    )
    assert np.max(np.abs(r_cos - np.asarray(d["r_cos_90"]))) < 1e-6
    assert np.max(np.abs(z_sin - np.asarray(d["z_sin_90"]))) < 1e-6


def test_evaluate_surface_matches_boundary_grid_at_boundary():
    d = _load_fixture()
    rmnc = np.asarray(d["rmnc"])
    zmns = np.asarray(d["zmns"])
    xm = np.asarray(d["xm"])
    xn = np.asarray(d["xn"])
    nfp = d["nfp"]

    n_theta, n_phi = 32, 24  # n_theta even, required for the reflection below
    R_wout, Z_wout, _theta, _phi = evaluate_surface(
        rmnc[-1], zmns[-1], xm, xn, n_theta=n_theta, n_phi=n_phi, n_fp=nfp
    )

    X90 = np.concatenate([d["r_cos_90"], d["z_sin_90"]])[None, :]
    grid = boundary_grid(X90, n_theta=n_theta, n_phi=n_phi, n_fp=nfp)
    R_ref, Z_ref = grid.R[0], grid.Z[0]

    reflect = (n_theta // 2 - np.arange(n_theta)) % n_theta
    R_matched, Z_matched = R_wout[reflect], Z_wout[reflect]

    assert np.max(np.abs(R_matched - R_ref)) < 1e-6
    assert np.max(np.abs(Z_matched - Z_ref)) < 1e-6
