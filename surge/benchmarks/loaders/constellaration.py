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
