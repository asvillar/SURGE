"""Closed-form boundary geometry for the ConStellaration stellarator dataset.

The ``plasma.constellaration_*`` benchmarks map 90 VMEC boundary Fourier
coefficients (``r_cos_0..44``, ``z_sin_0..44``) to 12 stellarator metrics
computed by an MHD equilibrium solve (Cadena et al. 2025, *ConStellaration: A
dataset of QI-like stellarator plasma boundaries and optimized configurations*,
arXiv:2506.19583; data: ``proxima-fusion/constellaration`` on HuggingFace).

Some of those 12 metrics are pure functions of the boundary shape and need no
equilibrium solve at all. This module evaluates the boundary surface on a
grid and derives those shape metrics analytically:

* ``boundary_grid`` / the ``R, Z`` evaluation — **exact**, direct evaluation
  of the truncated Fourier series that *defines* the boundary.
* ``aspect_ratio`` — **exact / closed-form**: minor radius from the
  toroidally-averaged cross-section area, major radius from the exact VMEC
  surface-volume integral. Verified against the dataset's
  ``aspect_ratio`` column at Pearson r > 0.9999 (see
  ``audit/presentation/review/constellaration_algebraic_check.md``).
* ``volume`` / ``mean_cross_section_area`` — **exact**, closed-form surface
  integrals (same integrals used to derive ``aspect_ratio``).
* ``max_elongation_est`` / ``average_triangularity_est`` — **estimates**.
  These are geometric *by construction* (pure shape descriptors of the
  boundary, no B-field involved), but the specific numeric convention used
  here (per-slice extrema and symmetric top/bottom triangularity) has not
  been matched to ConStellaration's own definition — treat these as
  order-of-magnitude shape descriptors, not reproductions of the dataset's
  ``max_elongation`` / ``average_triangularity`` columns.

Index / formula reference (recovered and verified numerically in the audit
above): flattening of the 45 ``(m, n)`` VMEC modes is row-major over
``m = 0..4`` (slow index) and ``n_idx = 0..8`` (fast index, toroidal mode
``n = n_idx - 4``), i.e. ``r_cos[i] = r_cos[m=i//9, n=i%9-4]`` (same for
``z_sin``). The boundary at one field period is::

    R(theta, phi) = sum_{m,n} r_cos[m, n] * cos(m*theta - n_fp*n*phi)
    Z(theta, phi) = sum_{m,n} z_sin[m, n] * sin(m*theta - n_fp*n*phi)

with ``n_fp = 3`` for this dataset.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

N_MODES_M = 5
N_MODES_N = 9
N_MODES_TOTAL = N_MODES_M * N_MODES_N  # 45
N_INPUT_FEATURES = 2 * N_MODES_TOTAL  # 90 (r_cos block + z_sin block)

# The boundary is a trigonometric polynomial band-limited by m <= 4, n <= 4:
# the theta/phi integrands used below (products of R, Z and their
# theta-derivatives) have frequency <= 2*4 = 8 in theta and well under the
# rectangle-rule Nyquist bound in phi. A uniform-grid rectangle/trapezoid
# rule integrates any trigonometric polynomial *exactly* once the grid has
# more points than the integrand's bandwidth, with no truncation error at
# all beyond floating point — so there is no accuracy reason to use a dense
# grid. Empirically (see tests/benchmarks/test_constellaration_geometry.py),
# 17x13 is already exact to ~1e-14; 32x24 keeps a comfortable safety margin
# while still being ~48x cheaper than the original 128x96 default.
DEFAULT_N_THETA = 32
DEFAULT_N_PHI = 24
# Row-chunk size for geometry_metrics/geometry_features so peak memory stays
# well under ~500 MB even on the full ~27k-row dataset (a chunk's R/Z/dR/dZ
# grids at the default resolution are ~2,048*32*24*8 bytes*4 arrays ~ 50 MB).
DEFAULT_CHUNK_SIZE = 2048

GEOMETRY_FEATURE_NAMES: list[str] = [
    "aspect_ratio",
    "volume",
    "mean_cross_section_area",
    "max_elongation_est",
    "average_triangularity_est",
]


def mode_table() -> list[tuple[int, int]]:
    """Return the ``(m, n)`` VMEC mode for each of the 90 input columns.

    Columns ``0..44`` are the ``r_cos`` block, ``45..89`` the ``z_sin``
    block; both blocks use the same row-major ``(m, n_idx)`` flattening
    (``m = i // 9``, ``n = i % 9 - 4``). E.g. column 4 (``r_cos_4``) is
    ``(m=0, n=0)`` — the normalized major-radius term, which is why
    ``r_cos_4`` is close to (but not exactly) 1.0 for every row in the
    dataset (observed range ~[0.958, 1.002] in ``paper_nfp3_clip0.05.npz``).
    """
    modes: list[tuple[int, int]] = []
    for _block in range(2):
        for i in range(N_MODES_TOTAL):
            modes.append((i // N_MODES_N, i % N_MODES_N - 4))
    return modes


def _split_coeffs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(n, 90)`` input rows into ``r_cos``/``z_sin`` mode grids ``(n, 5, 9)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_INPUT_FEATURES:
        raise ValueError(f"Expected X of shape (n, {N_INPUT_FEATURES}), got {X.shape}")
    r_cos = X[:, :N_MODES_TOTAL].reshape(-1, N_MODES_M, N_MODES_N)
    z_sin = X[:, N_MODES_TOTAL:].reshape(-1, N_MODES_M, N_MODES_N)
    return r_cos, z_sin


class BoundaryGrid(NamedTuple):
    """Boundary surface evaluated on a ``theta x phi`` grid over one field period."""

    R: np.ndarray  # (n, n_theta, n_phi)
    Z: np.ndarray  # (n, n_theta, n_phi)
    dR_dtheta: np.ndarray  # (n, n_theta, n_phi)
    dZ_dtheta: np.ndarray  # (n, n_theta, n_phi)
    theta: np.ndarray  # (n_theta,)
    phi: np.ndarray  # (n_phi,) -- one field period, [0, 2*pi/n_fp)


def boundary_grid(
    X: np.ndarray, n_theta: int = DEFAULT_N_THETA, n_phi: int = DEFAULT_N_PHI, n_fp: int = 3
) -> BoundaryGrid:
    """Evaluate the boundary ``R(theta, phi)``, ``Z(theta, phi)`` and their
    ``theta``-derivatives on a uniform grid over one field period.

    Exact (closed-form) evaluation of the truncated Fourier series — no
    approximation beyond the finite mode set already present in ``X``.
    Vectorised, batch-first: works for ``X`` of shape ``(n, 90)`` for any
    ``n``, including ``n == 1``.

    Uses the angle-sum identities ``cos(a-b) = cos a cos b + sin a sin b``
    and ``sin(a-b) = sin a cos b - cos a sin b`` to factor the double sum
    over ``(m, n)`` into two matrix contractions (over ``n`` then ``m``)
    instead of a dense ``O(n_theta * n_phi * 45)`` sum per sample, which
    keeps this cheap enough to run on the full ~27k-row dataset.

    Parameters
    ----------
    X:
        ``(n, 90)`` array of ``[r_cos_0..44, z_sin_0..44]`` VMEC boundary
        coefficients.
    n_theta, n_phi:
        Grid resolution in the poloidal (``theta``, full ``2*pi``) and
        toroidal (``phi``, one field period ``2*pi/n_fp``) directions.
    n_fp:
        Number of field periods (3 for this dataset).

    Returns
    -------
    :class:`BoundaryGrid` with ``R``, ``Z``, ``dR_dtheta``, ``dZ_dtheta`` of
    shape ``(n, n_theta, n_phi)``, plus the 1-D ``theta``/``phi`` grids.
    """
    r_cos, z_sin = _split_coeffs(X)  # (n, 5, 9) each

    m = np.arange(N_MODES_M, dtype=np.float64)  # (5,)
    n_idx = np.arange(N_MODES_N, dtype=np.float64)
    n_tor = n_idx - 4.0  # (9,)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / n_fp, n_phi, endpoint=False)

    cos_theta = np.cos(np.outer(m, theta))  # (5, n_theta)
    sin_theta = np.sin(np.outer(m, theta))  # (5, n_theta)
    cos_phi = np.cos(n_fp * np.outer(n_tor, phi))  # (9, n_phi)
    sin_phi = np.sin(n_fp * np.outer(n_tor, phi))  # (9, n_phi)

    # Contract over the toroidal mode index n first: A/B[.., m, p].
    A_r = np.einsum("bmn,np->bmp", r_cos, cos_phi)
    B_r = np.einsum("bmn,np->bmp", r_cos, sin_phi)
    A_z = np.einsum("bmn,np->bmp", z_sin, cos_phi)
    B_z = np.einsum("bmn,np->bmp", z_sin, sin_phi)

    # R = sum_m cos(m*theta)*A_r + sin(m*theta)*B_r
    R = np.einsum("mt,bmp->btp", cos_theta, A_r) + np.einsum("mt,bmp->btp", sin_theta, B_r)
    # Z = sum_m sin(m*theta)*A_z - cos(m*theta)*B_z
    Z = np.einsum("mt,bmp->btp", sin_theta, A_z) - np.einsum("mt,bmp->btp", cos_theta, B_z)

    # d/dtheta of the same series (analytic, reuses A/B).
    m_cos_theta = cos_theta * m[:, None]
    m_sin_theta = sin_theta * m[:, None]
    dR_dtheta = -np.einsum("mt,bmp->btp", m_sin_theta, A_r) + np.einsum("mt,bmp->btp", m_cos_theta, B_r)
    dZ_dtheta = np.einsum("mt,bmp->btp", m_cos_theta, A_z) + np.einsum("mt,bmp->btp", m_sin_theta, B_z)

    return BoundaryGrid(R=R, Z=Z, dR_dtheta=dR_dtheta, dZ_dtheta=dZ_dtheta, theta=theta, phi=phi)


def geometry_metrics(
    X: np.ndarray,
    *,
    n_theta: int = DEFAULT_N_THETA,
    n_phi: int = DEFAULT_N_PHI,
    n_fp: int = 3,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Closed-form and estimated boundary-geometry metrics for each row of ``X``.

    Parameters
    ----------
    X:
        ``(n, 90)`` VMEC boundary coefficients.
    n_theta, n_phi, n_fp:
        Passed through to :func:`boundary_grid`. The defaults (32x24) are
        already exact to floating-point precision for this band-limited
        (m<=4, n<=4) boundary — see the module-level comment above
        :data:`DEFAULT_N_THETA`.
    chunk_size:
        Rows are processed in batches of this size to bound peak memory
        (default 2048, ~50 MB per batch at the default grid). ``None``
        evaluates all rows in one batch.

    Returns
    -------
    dict of ``(n,)`` float64 arrays:

    ``aspect_ratio`` (**exact**)
        ``R0 / a`` where ``a = sqrt(mean_cross_section_area / pi)`` and
        ``R0 = volume / (2 * pi**2 * a**2)`` — the standard VMEC
        aspect-ratio definition, reproduces the dataset's ``aspect_ratio``
        column at Pearson r > 0.9999 (see module docstring).
    ``volume`` (**exact**)
        Enclosed volume from the surface integral
        ``V = n_fp * oint dphi oint dtheta (1/2) R^2 dZ/dtheta``.
    ``mean_cross_section_area`` (**exact**)
        Toroidally-averaged poloidal cross-section area, shoelace formula
        ``A(phi) = (1/2) oint dtheta (R dZ/dtheta - Z dR/dtheta)``.
    ``max_elongation_est`` (**estimate**)
        Max over phi-slices of ``(Zmax-Zmin) / (Rmax-Rmin)`` — a
        bounding-box elongation proxy. Convention not matched to the
        dataset's ``max_elongation`` column; see module docstring.
    ``average_triangularity_est`` (**estimate**)
        Mean over phi-slices of the symmetric top/bottom triangularity
        ``((R0_local - R@Zmax) + (R0_local - R@Zmin)) / (2*a_local)``
        (standard tokamak-style convention). Not matched to the dataset's
        ``average_triangularity`` column; see module docstring.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if chunk_size is None or n <= chunk_size:
        return _geometry_metrics_dense(X, n_theta=n_theta, n_phi=n_phi, n_fp=n_fp)

    out = {name: np.empty(n, dtype=np.float64) for name in GEOMETRY_FEATURE_NAMES}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = _geometry_metrics_dense(X[start:end], n_theta=n_theta, n_phi=n_phi, n_fp=n_fp)
        for name in GEOMETRY_FEATURE_NAMES:
            out[name][start:end] = chunk[name]
    return out


def _geometry_metrics_dense(
    X: np.ndarray, *, n_theta: int, n_phi: int, n_fp: int
) -> dict[str, np.ndarray]:
    """Single-batch (no chunking) implementation behind :func:`geometry_metrics`."""
    grid = boundary_grid(X, n_theta=n_theta, n_phi=n_phi, n_fp=n_fp)
    R, Z, dR_dtheta, dZ_dtheta = grid.R, grid.Z, grid.dR_dtheta, grid.dZ_dtheta

    dtheta = 2.0 * np.pi / n_theta
    dphi = (2.0 * np.pi / n_fp) / n_phi

    # Per-(sample, phi) cross-section area (shoelace, theta contour).
    area_phi = 0.5 * np.abs(np.sum(R * dZ_dtheta - Z * dR_dtheta, axis=1) * dtheta)  # (n, n_phi)
    mean_cross_section_area = area_phi.mean(axis=1)  # (n,)

    # Enclosed volume (VMEC surface-integral formula), one field period
    # times n_fp.
    volume = n_fp * np.abs(np.sum(0.5 * R**2 * dZ_dtheta, axis=(1, 2)) * dtheta * dphi)  # (n,)

    a = np.sqrt(mean_cross_section_area / np.pi)
    R0 = volume / (2.0 * np.pi**2 * a**2)
    aspect_ratio = R0 / a

    # Bounding-box elongation / symmetric triangularity per phi-slice.
    R_max = R.max(axis=1)  # (n, n_phi)
    R_min = R.min(axis=1)
    Z_max = Z.max(axis=1)
    Z_min = Z.min(axis=1)
    a_local = (R_max - R_min) / 2.0
    b_local = (Z_max - Z_min) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        elongation_local = np.where(a_local > 0, b_local / a_local, np.nan)

    idx_zmax = Z.argmax(axis=1)  # (n, n_phi)
    idx_zmin = Z.argmin(axis=1)
    R_at_zmax = np.take_along_axis(R, idx_zmax[:, None, :], axis=1)[:, 0, :]
    R_at_zmin = np.take_along_axis(R, idx_zmin[:, None, :], axis=1)[:, 0, :]
    R0_local = (R_max + R_min) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        triangularity_local = np.where(
            a_local > 0,
            ((R0_local - R_at_zmax) + (R0_local - R_at_zmin)) / (2.0 * a_local),
            np.nan,
        )

    max_elongation_est = np.nanmax(elongation_local, axis=1)
    average_triangularity_est = np.nanmean(triangularity_local, axis=1)

    return {
        "aspect_ratio": aspect_ratio,
        "volume": volume,
        "mean_cross_section_area": mean_cross_section_area,
        "max_elongation_est": max_elongation_est,
        "average_triangularity_est": average_triangularity_est,
    }


def geometry_features(
    X: np.ndarray,
    *,
    n_theta: int = DEFAULT_N_THETA,
    n_phi: int = DEFAULT_N_PHI,
    n_fp: int = 3,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Stack :func:`geometry_metrics` into a dense ``(n, k)`` feature array.

    Column order matches :data:`GEOMETRY_FEATURE_NAMES`. Runs in chunks (see
    ``chunk_size``) so it is cheap enough to apply to the full ~27k-row
    dataset (< 60 s, see the timing test).
    """
    metrics = geometry_metrics(X, n_theta=n_theta, n_phi=n_phi, n_fp=n_fp, chunk_size=chunk_size)
    return np.stack([metrics[name] for name in GEOMETRY_FEATURE_NAMES], axis=1).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Paper-filtered row IDs + full VMEC wout equilibria (surfaces, |B|, profiles)
#
# The 90-feature boundary + 12-metric cache used above
# (``paper_nfp3_clip0.05.npz``, built by ``leaderboard._load_constellaration_
# paper`` — surge/benchmarks/leaderboard.py:1491-1635) stores no row IDs, so
# it cannot be joined against the separate ``vmecpp_wout`` HF config (one
# full VMEC equilibrium per row, joined on ``plasma_config_id``). The
# functions below recover those IDs by re-streaming the `default` HF config
# and reproducing that exact filter (byte-for-byte verified against the
# cached npz, see ``paper_filter_ids``), then stream-download a radially
# subsampled slice of the matching wout equilibria.
# ═══════════════════════════════════════════════════════════════════════════

_HF_DATASET_NAME = "proxima-fusion/constellaration"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "constellaration"

# Mirrors leaderboard._CONSTELLARATION_METRICS / METRIC_COLS_SRC
# (surge/benchmarks/leaderboard.py:1475-1488, 1537) exactly. Percentile
# clipping is per-column and order-independent, so only the *set* of columns
# (not their order) needs to match for the recovered filter to be identical.
_PAPER_METRIC_NAMES = [
    "aspect_ratio",
    "aspect_ratio_over_edge_rotational_transform",
    "max_elongation",
    "axis_rotational_transform_over_n_field_periods",
    "edge_rotational_transform_over_n_field_periods",
    "axis_magnetic_mirror_ratio",
    "edge_magnetic_mirror_ratio",
    "average_triangularity",
    "vacuum_well",
    "minimum_normalized_magnetic_gradient_scale_length",
    "flux_compression_in_regions_of_bad_curvature",
    "log_10_qi",  # transformed from metrics.qi
]
_PAPER_METRIC_SRC_COLS = [m for m in _PAPER_METRIC_NAMES if m != "log_10_qi"] + ["qi"]


def _ensure_hf_ssl_env() -> None:
    """Best-effort fix for TLS-inspecting corporate proxies (e.g. Zscaler)
    whose root CA lives in the OS trust store but not in Python's
    ``certifi`` bundle, which otherwise makes ``datasets``/``huggingface_hub``
    streaming fail with ``CERTIFICATE_VERIFY_FAILED``.

    No-op if ``SSL_CERT_FILE`` is already set, if not on macOS, or if
    anything below fails (falls back to certifi's default behaviour).
    """
    if os.environ.get("SSL_CERT_FILE") or platform.system() != "Darwin":
        return
    try:
        import certifi

        bundle = _DEFAULT_DATA_DIR / ".ca_bundle.pem"
        if not bundle.exists():
            _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
            roots = subprocess.run(
                ["security", "find-certificate", "-a", "-p", "/Library/Keychains/System.keychain"],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout
            bundle.write_text(Path(certifi.where()).read_text() + roots)
        os.environ["SSL_CERT_FILE"] = str(bundle)
    except Exception:
        pass


def _verify_ids_against_npz(
    X: np.ndarray, n_field_periods: int, outlier_clip_pct: float, npz_path: "str | Path | None"
) -> None:
    """Assert the recovered boundary rows match ``paper_nfp{n}_clip{c}.npz``
    row-for-row (max abs diff < 1e-12). Silently returns if that npz is not
    present locally (nothing to verify against).
    """
    npz_path = Path(npz_path) if npz_path else (
        _DEFAULT_DATA_DIR / f"paper_nfp{n_field_periods}_clip{outlier_clip_pct}.npz"
    )
    if not npz_path.exists():
        return
    data = np.load(npz_path, allow_pickle=True)
    X_ref = data["X"]
    if X_ref.shape != X.shape:
        raise RuntimeError(
            f"paper_filter_ids: row count/shape mismatch vs {npz_path}: "
            f"recovered {X.shape} vs cached {X_ref.shape}"
        )
    max_diff = float(np.max(np.abs(np.asarray(X_ref, dtype=np.float64) - X)))
    if max_diff >= 1e-12:
        raise RuntimeError(
            f"paper_filter_ids: X mismatch vs {npz_path}: max abs diff {max_diff:.3e} >= 1e-12"
        )


def paper_filter_ids(
    n_field_periods: int = 3,
    outlier_clip_pct: float = 0.05,
    cache_path: "str | Path | None" = None,
    verify_against_npz: bool = True,
    npz_path: "str | Path | None" = None,
) -> dict:
    """Recover the ``plasma_config_id`` / ``misc.vmecpp_wout_id`` for every
    row of the paper-filtered cache, by re-streaming the HF ``default``
    config and reproducing the exact filter used to build it (see the
    module-level docstring above and
    ``leaderboard._load_constellaration_paper``,
    surge/benchmarks/leaderboard.py:1491-1635): ``n_field_periods`` match,
    an optimised (DESC or VMEC) configuration, all 12 metrics finite, then a
    per-metric ``outlier_clip_pct`` percentile clip.

    Returns ``{"plasma_config_id": [...], "vmecpp_wout_id": [...]}`` in the
    same row order as ``X``/``Y`` in ``paper_nfp{n}_clip{clip}.npz``. Cached
    to ``data/datasets/constellaration/paper_nfp{n}_ids.json``.

    If ``verify_against_npz`` and that npz cache exists locally, raises if
    the recovered rows' ``r_cos``/``z_sin`` don't match its ``X`` row-for-row
    to within 1e-12 max abs diff — the only way to be confident the ID
    recovery used exactly the same filter and row order as the cache.
    """
    cache_path = Path(cache_path) if cache_path else (
        _DEFAULT_DATA_DIR / f"paper_nfp{n_field_periods}_ids.json"
    )
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    _ensure_hf_ssl_env()
    import datasets as hf_datasets

    columns = [
        "plasma_config_id",
        "misc.vmecpp_wout_id",
        "boundary.n_field_periods",
        "desc_omnigenous_field_optimization_settings.id",
        "vmec_omnigenous_field_optimization_settings.id",
        "boundary.r_cos",
        "boundary.z_sin",
    ] + [f"metrics.{c}" for c in _PAPER_METRIC_SRC_COLS]

    ds = hf_datasets.load_dataset(_HF_DATASET_NAME, "default", split="train", streaming=True)
    ds = ds.select_columns(columns)

    ids_pcid: list = []
    ids_wid: list = []
    rows_X: list[np.ndarray] = []
    rows_Y: list[list[float]] = []

    for row in ds:
        if row.get("boundary.n_field_periods") != n_field_periods:
            continue
        desc_id = row.get("desc_omnigenous_field_optimization_settings.id")
        vmec_id = row.get("vmec_omnigenous_field_optimization_settings.id")
        if desc_id is None and vmec_id is None:
            continue

        r = row.get("boundary.r_cos")
        z = row.get("boundary.z_sin")
        if r is None or z is None:
            continue
        r_flat = np.asarray(r, dtype=float).ravel()
        z_flat = np.asarray(z, dtype=float).ravel()
        if r_flat.size != 45 or z_flat.size != 45:
            continue

        metric_vals: list[float] = []
        ok = True
        for col in _PAPER_METRIC_SRC_COLS[:-1]:  # all except qi
            v = row.get(f"metrics.{col}")
            if v is None:
                ok = False
                break
            metric_vals.append(float(v))
        if not ok:
            continue

        qi = row.get("metrics.qi")
        if qi is None or float(qi) <= 0:
            continue
        metric_vals.append(math.log10(float(qi)))

        ids_pcid.append(row.get("plasma_config_id"))
        ids_wid.append(row.get("misc.vmecpp_wout_id"))
        rows_X.append(np.concatenate([r_flat, z_flat]))
        rows_Y.append(metric_vals)

    if not rows_X:
        raise RuntimeError("paper_filter_ids: no rows passed the nfp/optimised filter.")

    X = np.array(rows_X, dtype=float)
    Y = np.array(rows_Y, dtype=float)
    ids_pcid_arr = np.array(ids_pcid, dtype=object)
    ids_wid_arr = np.array(ids_wid, dtype=object)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[valid], Y[valid]
    ids_pcid_arr, ids_wid_arr = ids_pcid_arr[valid], ids_wid_arr[valid]

    mask = np.ones(len(X), dtype=bool)
    half = outlier_clip_pct / 2.0
    for j in range(Y.shape[1]):
        lo = np.percentile(Y[:, j], half)
        hi = np.percentile(Y[:, j], 100.0 - half)
        mask &= (Y[:, j] >= lo) & (Y[:, j] <= hi)
    X = X[mask]
    ids_pcid_arr, ids_wid_arr = ids_pcid_arr[mask], ids_wid_arr[mask]

    if verify_against_npz:
        _verify_ids_against_npz(X, n_field_periods, outlier_clip_pct, npz_path)

    result = {
        "plasma_config_id": ids_pcid_arr.tolist(),
        "vmecpp_wout_id": ids_wid_arr.tolist(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


# ── VMEC wout equilibrium subset: download, cache, load, evaluate ──────────

# Fields stored per row (leading dim = row, i.e. one equilibrium) in each
# ``part_*.npz`` chunk file and the finalized subset npz.
_WOUT_ROW_FIELDS = [
    "rmnc", "zmns", "lmns", "bmnc",
    "iota_full", "pressure_full", "phi", "vp",
    "ns", "mpol", "ntor", "nfp",
    "aspect", "volume_p", "b0",
]


def _build_wout_meta(rec: dict, keep: int) -> dict:
    """Radial-subsample plan + shared mode tables, built once from the first
    row's wout JSON and checked against every subsequent row.
    """
    ns = int(rec["ns"])
    mnmax = len(rec["xm"])
    mnmax_nyq = len(rec["xm_nyq"])
    n_full = max(1, min(keep, ns))
    surf_idx_full = np.unique(np.round(np.linspace(0, ns - 1, n_full)).astype(np.int64))
    n_half = max(ns - 1, 0)
    if n_half > 0:
        surf_idx_half = np.unique(np.clip(surf_idx_full[:-1], 0, n_half - 1))
    else:
        surf_idx_half = np.array([], dtype=np.int64)
    return {
        "ns": ns,
        "mpol": int(rec["mpol"]),
        "ntor": int(rec["ntor"]),
        "mnmax": mnmax,
        "mnmax_nyq": mnmax_nyq,
        "xm": np.array(rec["xm"], dtype=np.int32),
        "xn": np.array(rec["xn"], dtype=np.int32),
        "xm_nyq": np.array(rec["xm_nyq"], dtype=np.int32),
        "xn_nyq": np.array(rec["xn_nyq"], dtype=np.int32),
        "surf_idx_full": surf_idx_full.astype(np.int32),
        "surf_idx_half": surf_idx_half.astype(np.int32),
        "s_full": (surf_idx_full / max(ns - 1, 1)).astype(np.float64),
        "s_half": ((surf_idx_half + 0.5) / max(ns - 1, 1)).astype(np.float64),
    }


def _extract_wout_row(rec: dict, meta: dict) -> "dict | None":
    """Extract the float32 radial subsample for one wout JSON record.
    Returns ``None`` (row skipped, caller logs it) if this row's resolution
    doesn't match ``meta`` (defensive: the dataset is expected to use a
    fixed ``ns``/``mpol``/``ntor`` throughout, verified empirically, but a
    mismatch must not corrupt the fixed-shape cache).
    """
    if (
        int(rec["ns"]) != meta["ns"]
        or len(rec["xm"]) != meta["mnmax"]
        or len(rec["xm_nyq"]) != meta["mnmax_nyq"]
    ):
        return None

    fi, hi = meta["surf_idx_full"], meta["surf_idx_half"]
    rmnc_full = np.asarray(rec["rmnc"], dtype=np.float32)
    zmns_full = np.asarray(rec["zmns"], dtype=np.float32)
    lmns_full = np.asarray(rec["lmns"], dtype=np.float32)  # half mesh, (ns-1, mnmax)
    bmnc_full = np.asarray(rec["bmnc"], dtype=np.float32)  # half mesh, (ns-1, mnmax_nyq)

    phi_src = rec.get("toroidal_flux", rec.get("phi"))

    return {
        "rmnc": rmnc_full[fi],
        "zmns": zmns_full[fi],
        "lmns": lmns_full[hi] if len(hi) else lmns_full[:0],
        "bmnc": bmnc_full[hi] if len(hi) else bmnc_full[:0],
        "iota_full": np.asarray(rec["iota_full"], dtype=np.float32),
        "pressure_full": np.asarray(rec["pressure_full"], dtype=np.float32),
        "phi": np.asarray(phi_src, dtype=np.float32),
        "vp": np.asarray(rec["dVds"], dtype=np.float32),
        "ns": np.int32(rec["ns"]),
        "mpol": np.int32(rec["mpol"]),
        "ntor": np.int32(rec["ntor"]),
        "nfp": np.int32(rec["nfp"]),
        "aspect": np.float32(rec["aspect"]),
        "volume_p": np.float32(rec["volume_p"]),
        "b0": np.float32(rec["b0"]),
    }


def _flush_part(parts_dir: Path, part_idx: int, buf: "list[dict]") -> None:
    def stack(key: str) -> np.ndarray:
        return np.stack([b[key] for b in buf], axis=0)

    path = parts_dir / f"part_{part_idx:06d}.npz"
    kwargs = {k: stack(k) for k in _WOUT_ROW_FIELDS}
    kwargs["plasma_config_id"] = np.array([b["plasma_config_id"] for b in buf])
    np.savez_compressed(path, **kwargs)


def _scan_existing_part_ids(parts_dir: Path) -> set:
    """Recursively scan ``parts_dir`` (root-level parts *and* any
    ``worker_<i>/`` subdirectories) for already-stored ``plasma_config_id``s.
    """
    done: set = set()
    if not parts_dir.exists():
        return done
    for p in sorted(parts_dir.glob("**/part_*.npz")):
        try:
            with np.load(p) as d:
                done.update(d["plasma_config_id"].tolist())
        except Exception:
            continue
    return done


def _next_part_index(worker_dir: Path) -> int:
    """Next part-file index *within a single worker's own directory* (part
    numbering is local to each worker so concurrent workers never collide).
    """
    existing = sorted(worker_dir.glob("part_*.npz"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def _collect_parts(parts_dir: Path, id_order: "list[str] | None" = None) -> dict:
    """Concatenate every ``part_*.npz`` under ``parts_dir`` (recursively,
    across any ``worker_<i>/`` subdirectories) plus one shared ``meta.npz``.

    If ``id_order`` is given, rows are reordered (and any ids not found are
    dropped) to match it -- used to align the merged cache to
    ``paper_filter_ids()``'s row order.
    """
    meta_path = next(iter(sorted(parts_dir.glob("**/meta.npz"))), None)
    parts = sorted(parts_dir.glob("**/part_*.npz"))
    if meta_path is None or not parts:
        raise FileNotFoundError(f"no wout subset data in {parts_dir}")
    with np.load(meta_path) as m:
        meta = {k: m[k] for k in m.files}
    collected: dict[str, list[np.ndarray]] = {k: [] for k in _WOUT_ROW_FIELDS}
    pcids = []
    for p in parts:
        with np.load(p) as d:
            pcids.append(d["plasma_config_id"])
            for k in _WOUT_ROW_FIELDS:
                collected[k].append(d[k])
    pcid_arr = np.concatenate(pcids)
    out: dict = {"plasma_config_id": pcid_arr}
    for k in _WOUT_ROW_FIELDS:
        out[k] = np.concatenate(collected[k], axis=0)

    if id_order is not None:
        pos = {pcid: i for i, pcid in enumerate(pcid_arr.tolist())}
        keep_ids = [pcid for pcid in id_order if pcid in pos]
        sel = np.array([pos[pcid] for pcid in keep_ids], dtype=np.int64)
        out["plasma_config_id"] = pcid_arr[sel]
        for k in _WOUT_ROW_FIELDS:
            out[k] = out[k][sel]

    out.update(meta)
    return out


def _finalize_wout_subset(
    parts_dir: Path, out_path: Path, id_order: "list[str] | None" = None
) -> Path:
    out = _collect_parts(parts_dir, id_order=id_order)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out_path


def _download_wout_worker(
    worker_idx: int,
    n_workers: int,
    ids: "list[str]",
    out_path_str: str,
    keep: int,
    chunk_rows: int,
    progress_every: int,
) -> None:
    """One worker's share of the ``vmecpp_wout`` stream.

    When ``n_workers > 1``, splits the streaming ``IterableDataset`` into
    ``n_workers`` contiguous-shard-file partitions via ``.shard(num_shards=
    n_workers, index=worker_idx)`` (each worker reads a disjoint subset of
    the underlying parquet shard files) and runs as its own OS process (see
    :func:`download_wout_subset`); writes its own part files under
    ``<out_path>_parts/worker_<worker_idx>/`` and its own progress file
    ``<out_path>.progress.worker<worker_idx>.json``, so concurrent workers
    never touch the same files. With ``n_workers <= 1`` this *is* the
    (single-process) download loop, writing directly to ``<out_path>_parts/``
    and ``<out_path>.progress.json``.

    The ``plasma_config_id`` column is checked against the target ``ids``
    *before* the (megabyte-scale) ``json`` column is ever parsed.
    """
    id_set = set(ids)
    out_path = Path(out_path_str)
    parts_root = out_path.parent / f"{out_path.stem}_parts"
    worker_dir = parts_root if n_workers <= 1 else (parts_root / f"worker_{worker_idx}")
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = "[wout_subset]" if n_workers <= 1 else f"[wout_subset w{worker_idx}]"
    progress_path = (
        out_path.parent / f"{out_path.stem}.progress.json"
        if n_workers <= 1
        else out_path.parent / f"{out_path.stem}.progress.worker{worker_idx}.json"
    )
    skipped_path = worker_dir / "skipped.json"

    # Snapshot of everything already stored (by any worker, or a prior run)
    # at start-up -- since shards are disjoint (each row belongs to exactly
    # one parquet shard file), a given plasma_config_id can only ever be
    # matched by one worker, so no cross-worker duplicate-write races.
    done_ids = _scan_existing_part_ids(parts_root)
    if len(done_ids & id_set) >= len(id_set):
        return

    _ensure_hf_ssl_env()
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(_HF_DATASET_NAME, "vmecpp_wout", split="train", streaming=True)
    if n_workers > 1:
        ds = ds.shard(num_shards=n_workers, index=worker_idx)

    meta = None
    for meta_candidate in (worker_dir / "meta.npz", parts_root / "meta.npz"):
        if meta_candidate.exists():
            with np.load(meta_candidate) as m:
                meta = {k: m[k] for k in m.files}
                meta["ns"], meta["mpol"], meta["ntor"] = int(meta["ns"]), int(meta["mpol"]), int(meta["ntor"])
                meta["mnmax"], meta["mnmax_nyq"] = int(meta["mnmax"]), int(meta["mnmax_nyq"])
            break

    buf: list = []
    skipped: list = []
    part_idx = _next_part_index(worker_dir)
    t0 = time.time()
    n_done_local = 0

    for row in ds:
        pcid = row.get("plasma_config_id")
        if pcid not in id_set or pcid in done_ids:
            continue  # id column checked before the json column is parsed
        rec = json.loads(row["json"])
        if meta is None:
            meta = _build_wout_meta(rec, keep=keep)
            # Persist immediately (not just at the end): part files are
            # unreadable without it, so a crash before completion must not
            # leave meta.npz missing.
            np.savez(worker_dir / "meta.npz", **meta)
        row_arrays = _extract_wout_row(rec, meta)
        if row_arrays is None:
            skipped.append(pcid)
            done_ids.add(pcid)  # don't retry a structurally incompatible row
            continue
        row_arrays["plasma_config_id"] = pcid
        buf.append(row_arrays)
        done_ids.add(pcid)
        n_done_local += 1

        if len(buf) >= chunk_rows:
            _flush_part(worker_dir, part_idx, buf)
            part_idx += 1
            buf = []

        if n_done_local % progress_every == 0:
            dt = time.time() - t0
            rate = n_done_local / dt if dt > 0 else 0.0
            print(f"{log_prefix} {n_done_local} local rows ({rate:.2f} rows/s)", flush=True)
            progress_path.write_text(json.dumps({
                "worker": worker_idx, "n_workers": n_workers,
                "n_done_local": n_done_local, "rows_per_s_local": rate,
                "elapsed_s": dt, "finished": False,
            }))

    if buf:
        _flush_part(worker_dir, part_idx, buf)
    if meta is not None and not (worker_dir / "meta.npz").exists():
        np.savez(worker_dir / "meta.npz", **meta)
    if skipped:
        prior = json.loads(skipped_path.read_text()) if skipped_path.exists() else []
        skipped_path.write_text(json.dumps(prior + skipped))

    dt = time.time() - t0
    rate = n_done_local / dt if dt > 0 else 0.0
    print(f"{log_prefix} shard exhausted: {n_done_local} local rows ({rate:.2f} rows/s)", flush=True)
    progress_path.write_text(json.dumps({
        "worker": worker_idx, "n_workers": n_workers,
        "n_done_local": n_done_local, "rows_per_s_local": rate,
        "elapsed_s": dt, "finished": True,
    }))


def download_wout_subset(
    ids: "list[str] | None" = None,
    out_path: "str | Path | None" = None,
    keep: int = 16,
    chunk_rows: int = 500,
    n_field_periods: int = 3,
    progress_every: int = 1000,
    n_workers: int = 1,
) -> Path:
    """Stream the HF ``vmecpp_wout`` config, keep only rows whose
    ``plasma_config_id`` is in ``ids`` (default: ``paper_filter_ids()``),
    and store a radial subsample of each matching VMEC equilibrium.

    Per equilibrium, stores float32: ``rmnc``/``zmns`` (boundary-surface
    Fourier coeffs) and ``lmns``/``bmnc`` on <= ``keep`` radial surfaces
    (evenly spaced by index, including the axis and the boundary; see
    ``meta_s_full``/``meta_s_half`` for the kept normalized-radius values),
    the shared mode tables (``meta_xm``/``meta_xn``/``meta_xm_nyq``/
    ``meta_xn_nyq``), the 1-D profiles ``iota_full``, ``pressure_full``,
    ``phi`` (cumulative toroidal flux) and ``vp`` (``dV/ds``, half mesh) at
    full radial resolution, and the scalars ``ns``/``mpol``/``ntor``/``nfp``/
    ``aspect``/``volume_p``/``b0``.

    Writes incrementally to ``<out_path>_parts/part_NNNNNN.npz`` chunks of
    ``chunk_rows`` rows (so a crash keeps all completed chunks) and merges
    them into ``out_path``, row-order-aligned to ``ids``, once every
    requested ID has been found or every shard is exhausted. Resumable:
    rows whose ``plasma_config_id`` is already present in an existing part
    file (root-level, or under any ``worker_<i>/``) are skipped.

    ``n_workers``:
        ``1`` (default): single-process download, as above. ``> 1``: fan
        out across ``n_workers`` OS processes (``multiprocessing``, spawn
        context), each handling a disjoint contiguous partition of the
        underlying parquet shard files (``IterableDataset.shard``) and
        writing its own ``<out_path>_parts/worker_<i>/`` part files and
        ``<out_path>.progress.worker<i>.json`` progress file (aggregate
        throughput = sum of the per-worker ``rows_per_s_local``). Since a
        given ``plasma_config_id`` lives in exactly one shard, workers never
        write conflicting data. Blocks until every worker's shard is
        exhausted, then finalizes.
    """
    if ids is None:
        ids = paper_filter_ids(n_field_periods=n_field_periods)["plasma_config_id"]
    ids = list(ids)
    id_set = set(ids)
    n_total = len(id_set)

    out_path = Path(out_path) if out_path else (_DEFAULT_DATA_DIR / "wout_nfp3_subset.npz")
    parts_dir = out_path.parent / f"{out_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    done_ids = _scan_existing_part_ids(parts_dir)
    if len(done_ids & id_set) >= n_total:
        if not out_path.exists():
            _finalize_wout_subset(parts_dir, out_path, id_order=ids)
        return out_path

    if n_workers <= 1:
        _download_wout_worker(0, 1, ids, str(out_path), keep, chunk_rows, progress_every)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(
                target=_download_wout_worker,
                args=(i, n_workers, ids, str(out_path), keep, chunk_rows, progress_every),
            )
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    done_ids = _scan_existing_part_ids(parts_dir)
    if len(done_ids & id_set) >= n_total:
        _finalize_wout_subset(parts_dir, out_path, id_order=ids)
    return out_path


def load_wout_subset(path: "str | Path | None" = None) -> dict:
    """Load a wout-subset cache written by :func:`download_wout_subset`.

    Returns a dict of arrays keyed by field name (``rmnc``, ``zmns``,
    ``lmns``, ``bmnc``, the profiles/scalars, ``plasma_config_id``) plus the
    shared ``meta_*`` arrays (mode tables ``meta_xm``/``meta_xn``/
    ``meta_xm_nyq``/``meta_xn_nyq``, kept-surface grids ``meta_s_full``/
    ``meta_s_half``). Row ``i`` of every per-row array corresponds to
    ``plasma_config_id[i]``.

    If the finalized ``path`` npz isn't written yet (a background download
    still running), falls back to concatenating the ``<path>_parts/`` chunk
    files on the fly, so progress can be inspected mid-run.
    """
    path = Path(path) if path else (_DEFAULT_DATA_DIR / "wout_nfp3_subset.npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    parts_dir = path.parent / f"{path.stem}_parts"
    return _collect_parts(parts_dir)


def evaluate_surface(
    rmnc_row: np.ndarray,
    zmns_row: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    n_theta: int = DEFAULT_N_THETA,
    n_phi: int = DEFAULT_N_PHI,
    n_fp: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``R(theta, phi)``, ``Z(theta, phi)`` for one VMEC surface from its raw
    wout Fourier coefficients, evaluated over one field period.

    Unlike :func:`boundary_grid` (which is specialised to the band-limited
    90-feature ``r_cos``/``z_sin`` dataset and its raw-``n`` convention),
    this works for any wout mode table. ``xn`` here is this dataset's own
    (vmecpp) wout encoding, empirically verified (see
    tests/benchmarks/test_constellaration_wout.py) to be ``xn = 2 * n``
    -- the toroidal mode index doubled, **independent of** ``n_field_periods``
    (not the classic Fortran-VMEC ``xn = n * nfp`` convention) -- so the
    physical angle is ``m*theta - n_fp*(xn/2)*phi``; ``n_fp`` both scales
    that angle and sets the ``phi`` domain to one field period.

    Note this uses VMEC's own internal poloidal angle, which is gauge-
    related to (not identical to) the angle underlying ``boundary.r_cos``/
    ``z_sin``: empirically, ``theta_ref = pi - theta_wout`` with the same
    ``phi`` (see :func:`wout_boundary_to_r_cos_z_sin`, which converts a
    wout boundary row into the ``r_cos``/``z_sin`` convention directly).

    Parameters
    ----------
    rmnc_row, zmns_row:
        ``(mnmax,)`` (or ``(b, mnmax)`` batched) Fourier coefficients for one
        surface, e.g. ``load_wout_subset(...)["rmnc"][i, k]`` for row ``i``,
        kept-surface index ``k``.
    xm, xn:
        ``(mnmax,)`` mode tables, e.g. ``meta_xm``/``meta_xn``.

    Returns
    -------
    ``R``, ``Z`` of shape ``(b, n_theta, n_phi)`` (squeezed to
    ``(n_theta, n_phi)`` if the input was 1-D), plus the 1-D ``theta``/
    ``phi`` grids.
    """
    squeeze = np.asarray(rmnc_row).ndim == 1
    rmnc_row = np.atleast_2d(np.asarray(rmnc_row, dtype=np.float64))
    zmns_row = np.atleast_2d(np.asarray(zmns_row, dtype=np.float64))
    xm = np.asarray(xm, dtype=np.float64)
    xn = np.asarray(xn, dtype=np.float64)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / n_fp, n_phi, endpoint=False)

    n_idx = 0.5 * xn  # this dataset's wout xn = 2 * n, regardless of n_fp
    angle = xm[:, None, None] * theta[None, :, None] - (n_fp * n_idx)[:, None, None] * phi[None, None, :]
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    R = np.einsum("bm,mtp->btp", rmnc_row, cos_a)
    Z = np.einsum("bm,mtp->btp", zmns_row, sin_a)
    if squeeze:
        R, Z = R[0], Z[0]
    return R, Z, theta, phi


def evaluate_modB(
    bmnc_row: np.ndarray,
    xm_nyq: np.ndarray,
    xn_nyq: np.ndarray,
    n_theta: int = DEFAULT_N_THETA,
    n_phi: int = DEFAULT_N_PHI,
    n_fp: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``|B|(theta, phi)`` on one surface from its Nyquist-mode ``bmnc`` row.

    Same conventions as :func:`evaluate_surface` (``xn_nyq`` is this
    dataset's ``2 * n`` encoding, independent of ``n_fp``; stellarator-
    symmetric cosine series). Returns ``modB`` of shape
    ``(b, n_theta, n_phi)`` (squeezed if the input was 1-D) plus the
    ``theta``/``phi`` grids.
    """
    squeeze = np.asarray(bmnc_row).ndim == 1
    bmnc_row = np.atleast_2d(np.asarray(bmnc_row, dtype=np.float64))
    xm_nyq = np.asarray(xm_nyq, dtype=np.float64)
    xn_nyq = np.asarray(xn_nyq, dtype=np.float64)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / n_fp, n_phi, endpoint=False)

    n_idx_nyq = 0.5 * xn_nyq
    angle = xm_nyq[:, None, None] * theta[None, :, None] - (n_fp * n_idx_nyq)[:, None, None] * phi[None, None, :]
    modB = np.einsum("bm,mtp->btp", bmnc_row, np.cos(angle))
    if squeeze:
        modB = modB[0]
    return modB, theta, phi


def wout_boundary_to_r_cos_z_sin(
    rmnc_boundary_row: np.ndarray, zmns_boundary_row: np.ndarray, xm: np.ndarray, xn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project a wout boundary surface's native Fourier coefficients onto the
    dataset's 90-feature ``(r_cos, z_sin)`` convention (``m=0..4``,
    ``n=-4..4``; see the module docstring and :func:`mode_table`).

    VMEC's internal poloidal angle for the wout representation is gauge-
    related to (not identical to) the angle underlying
    ``boundary.r_cos``/``z_sin``: empirically verified (see
    tests/benchmarks/test_constellaration_wout.py, which cross-checks this
    against the HF `default` config's own ``r_cos``/``z_sin`` for the same
    ``plasma_config_id``, to ~1e-6) to be ``theta_ref = pi - theta_wout``
    with the same ``phi``. Substituting that into the wout Fourier series
    gives, mode by mode (using ``n = xn / 2``, this dataset's toroidal mode
    index -- see :func:`evaluate_surface`):

    * ``m == 0``: ``r_cos[0, n] = rmnc[0, n]``, ``z_sin[0, n] = zmns[0, n]``
      (only ``n >= 0`` is populated in the wout mode table for ``m=0``).
    * ``m >= 1``: ``r_cos[m, -n] = (-1)**m * rmnc[m, n]``,
      ``z_sin[m, -n] = (-1)**(m+1) * zmns[m, n]``.

    Modes outside ``m <= 4``, ``|n| <= 4`` are dropped (the 90-feature
    representation is band-limited to that range; this dataset's boundary
    has negligible power outside it).

    Parameters
    ----------
    rmnc_boundary_row, zmns_boundary_row:
        ``(mnmax,)`` Fourier coefficients at the boundary surface (``s=1``,
        i.e. the last full-mesh surface -- ``load_wout_subset(...)["rmnc"][i, -1]``
        if ``-1`` is the kept boundary surface).
    xm, xn:
        ``(mnmax,)`` wout mode tables.

    Returns
    -------
    ``(r_cos, z_sin)``, each ``(45,)``, in the same row-major
    ``m=0..4``, ``n_idx=0..8`` (``n = n_idx - 4``) flattening as
    :func:`mode_table`.
    """
    xm = np.asarray(xm, dtype=np.int64)
    n_idx = np.asarray(xn, dtype=np.int64) // 2
    rmnc_boundary_row = np.asarray(rmnc_boundary_row, dtype=np.float64)
    zmns_boundary_row = np.asarray(zmns_boundary_row, dtype=np.float64)

    r_cos = np.zeros((N_MODES_M, N_MODES_N), dtype=np.float64)
    z_sin = np.zeros((N_MODES_M, N_MODES_N), dtype=np.float64)

    for i in range(len(xm)):
        m = int(xm[i])
        if m > 4:
            continue
        n = int(n_idx[i]) if m == 0 else -int(n_idx[i])
        if not (-4 <= n <= 4):
            continue
        r_sign = -1.0 if m % 2 else 1.0
        z_sign = -1.0 if (m % 2 == 0 and m > 0) else 1.0
        r_cos[m, n + 4] = r_sign * rmnc_boundary_row[i]
        z_sin[m, n + 4] = z_sign * zmns_boundary_row[i]

    return r_cos.ravel(), z_sin.ravel()
