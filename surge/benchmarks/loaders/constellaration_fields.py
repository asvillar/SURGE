"""Field-shaped ConStellaration benchmarks: boundary geometry -> |B| and
flux-surface maps.

Builds the ``plasma.constellaration_modb`` / ``plasma.constellaration_surfaces``
benchmark arrays from the radially-subsampled VMEC wout cache written by
:func:`surge.benchmarks.loaders.constellaration.download_wout_subset`. Uses
only that module's public, documented API (``load_wout_subset``,
``wout_boundary_to_r_cos_z_sin``, ``boundary_grid``, ``evaluate_surface``,
``evaluate_modB``, ``geometry_metrics``) -- see that module's docstring for
the angle/gauge conventions this relies on.

Angle-gauge note
-----------------
``evaluate_surface``/``evaluate_modB`` evaluate wout quantities on VMEC's own
poloidal angle ``theta_wout``; ``boundary_grid`` (and therefore this
benchmark's ``X``, built from the same 90 ``r_cos``/``z_sin`` features as the
other ``plasma.constellaration_*`` benchmarks) uses the dataset's
``r_cos``/``z_sin`` gauge ``theta_ref = pi - theta_wout`` (same ``phi``; see
``constellaration.evaluate_surface`` / ``wout_boundary_to_r_cos_z_sin``
docstrings). For a uniform ``n_theta``-point grid with even ``n_theta``, that
gauge flip is exactly the index permutation
``perm[i] = (n_theta // 2 - i) % n_theta`` -- an involution -- so any
wout-native ``theta_wout``-grid array is re-expressed on the ``theta_ref``
grid used by ``X`` via ``arr[..., perm, :]``, no interpolation needed. This
is the same identity ``tests/benchmarks/test_constellaration_wout.py``
verifies numerically for ``evaluate_surface`` vs. ``boundary_grid``; see
:func:`_theta_ref_permutation` and
``tests/benchmarks/test_constellaration_field.py`` for the equivalent check
on the field arrays built here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constellaration import (
    N_INPUT_FEATURES,
    _DEFAULT_DATA_DIR,
    boundary_grid,
    evaluate_modB,
    evaluate_surface,
    geometry_metrics,
    load_wout_subset,
    wout_boundary_to_r_cos_z_sin,
)

DEFAULT_N_THETA_FIELD = 64
DEFAULT_N_PHI_FIELD = 32


def _theta_ref_permutation(n_theta: int) -> np.ndarray:
    """Index permutation mapping the wout-native ``theta_wout`` grid onto the
    ``theta_ref`` grid used by :func:`boundary_grid` (``r_cos``/``z_sin``
    gauge): ``arr_ref[..., i, :] = arr_wout[..., perm[i], :]``.

    Requires even ``n_theta`` (the reflection ``theta_ref = pi - theta_wout``
    is then an exact grid-index permutation, not an interpolation). See the
    module docstring.
    """
    if n_theta % 2 != 0:
        raise ValueError(f"theta_ref<->theta_wout permutation requires even n_theta, got {n_theta}")
    return (n_theta // 2 - np.arange(n_theta)) % n_theta


def _boundary_x90(rmnc_boundary: np.ndarray, zmns_boundary: np.ndarray, xm: np.ndarray, xn: np.ndarray) -> np.ndarray:
    """Row-by-row :func:`wout_boundary_to_r_cos_z_sin`, stacked to ``(n, 90)``.

    ``wout_boundary_to_r_cos_z_sin`` is not batched, so this loops over rows;
    cheap relative to the equilibrium data it derives from (O(n * mnmax)
    plain numpy ops).
    """
    n = rmnc_boundary.shape[0]
    out = np.empty((n, N_INPUT_FEATURES), dtype=np.float64)
    for i in range(n):
        r_cos, z_sin = wout_boundary_to_r_cos_z_sin(rmnc_boundary[i], zmns_boundary[i], xm, xn)
        out[i, : N_INPUT_FEATURES // 2] = r_cos
        out[i, N_INPUT_FEATURES // 2 :] = z_sin
    return out


def _cache_path(n_rows: int, n_theta: int, n_phi: int, n_fp: int) -> Path:
    return _DEFAULT_DATA_DIR / f"field_nfp{n_fp}_n{n_rows}_t{n_theta}_p{n_phi}.npz"


def load_constellaration_field_raw(
    *,
    n_theta: int = DEFAULT_N_THETA_FIELD,
    n_phi: int = DEFAULT_N_PHI_FIELD,
    path: "str | Path | None" = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the shared boundary/modB/flux-surface arrays behind both field
    benchmarks from whatever rows of the wout subset are present on disk.

    Accepts a partial ``*_parts/`` cache (prints how many rows were used) or
    the finalized merged npz once the download completes; scales unchanged
    from 500 rows today to the full ~26,897-row dataset.

    Returns a dict with (row-aligned, row count ``n``):

    ``plasma_config_id`` (n,), ``X`` (n, 2, 64, 32) boundary ``R``/``Z`` on
    the ``theta_ref`` grid (from the 90 ``r_cos``/``z_sin`` features via
    :func:`boundary_grid`, exact -- same convention as the scalar
    ``plasma.constellaration_*`` benchmarks), ``modb`` (n, 64, 32) ``|B|`` on
    the boundary (last kept half-mesh surface) re-expressed on that same
    ``theta_ref`` grid (dimensionless: divided by its own per-sample mean,
    see ``modb_scale``), ``modb_scale`` (n,) the per-sample mean ``|B|``
    used for that normalisation, ``surfaces`` (n, 2, n_surfaces, 64) the
    poloidal cross-section at ``phi=0`` of every kept flux surface (``R``,
    ``Z`` channels, ``theta_ref`` grid, surface 0 = magnetic axis, last =
    boundary), and ``minor_radius`` (n,) from the exact closed-form
    :func:`geometry_metrics` (``sqrt(mean_cross_section_area / pi)``, same
    definition used by the ``aspect_ratio`` scalar benchmark).
    """
    finalized_path = Path(path) if path else (_DEFAULT_DATA_DIR / "wout_nfp3_subset.npz")
    data = load_wout_subset(path)
    n = len(data["plasma_config_id"])
    if not finalized_path.exists():
        print(
            f"[constellaration_fields] merged {finalized_path.name} not found yet; "
            f"using {n} row(s) from *_parts/ chunks on disk (more will arrive as the "
            "background download continues)",
            flush=True,
        )

    n_fp = int(np.asarray(data["nfp"]).ravel()[0])

    cache_path = _cache_path(n, n_theta, n_phi, n_fp)
    if use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    xm, xn = np.asarray(data["xm"]), np.asarray(data["xn"])
    xm_nyq, xn_nyq = np.asarray(data["xm_nyq"]), np.asarray(data["xn_nyq"])
    rmnc, zmns, bmnc = np.asarray(data["rmnc"]), np.asarray(data["zmns"]), np.asarray(data["bmnc"])
    n_surfaces = rmnc.shape[1]

    perm = _theta_ref_permutation(n_theta)

    # ── X: boundary R,Z from the 90-feature r_cos/z_sin, theta_ref grid ────
    X90 = _boundary_x90(rmnc[:, -1, :], zmns[:, -1, :], xm, xn)
    grid = boundary_grid(X90, n_theta=n_theta, n_phi=n_phi, n_fp=n_fp)
    X = np.stack([grid.R, grid.Z], axis=1).astype(np.float32)  # (n, 2, n_theta, n_phi)

    # ── |B| on the boundary (last kept half-mesh surface), theta_ref grid ──
    modB_wout, _t, _p = evaluate_modB(bmnc[:, -1, :], xm_nyq, xn_nyq, n_theta=n_theta, n_phi=n_phi, n_fp=n_fp)
    modB_ref = modB_wout[:, perm, :]
    modb_scale = modB_ref.reshape(n, -1).mean(axis=1)
    modb = (modB_ref / modb_scale[:, None, None]).astype(np.float32)

    # ── Poloidal cross-section (phi=0) of every kept flux surface ──────────
    R_list, Z_list = [], []
    for k in range(n_surfaces):
        R_k, Z_k, _t, _p = evaluate_surface(rmnc[:, k, :], zmns[:, k, :], xm, xn, n_theta=n_theta, n_phi=1, n_fp=n_fp)
        R_list.append(R_k[:, :, 0])
        Z_list.append(Z_k[:, :, 0])
    R_surf = np.stack(R_list, axis=1)[:, :, perm]  # (n, n_surfaces, n_theta)
    Z_surf = np.stack(Z_list, axis=1)[:, :, perm]
    surfaces = np.stack([R_surf, Z_surf], axis=1).astype(np.float32)  # (n, 2, n_surfaces, n_theta)

    minor_radius = np.sqrt(geometry_metrics(X90)["mean_cross_section_area"] / np.pi).astype(np.float32)

    out = {
        "plasma_config_id": data["plasma_config_id"],
        "X": X,
        "modb": modb,
        "modb_scale": modb_scale.astype(np.float32),
        "surfaces": surfaces,
        "minor_radius": minor_radius,
    }
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    return out


def constellaration_modb_dataset(
    *, n_theta: int = DEFAULT_N_THETA_FIELD, n_phi: int = DEFAULT_N_PHI_FIELD, path: "str | Path | None" = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """``plasma.constellaration_modb`` arrays: boundary (R, Z) -> |B|/mean(|B|).

    Returns ``(X, y, plasma_config_id, extra)``: ``X`` (n, 2, 64, 32),
    ``y`` (n, 64, 32) dimensionless (per-sample-mean-normalised), ``extra``
    with ``modb_scale`` (n,) and ``minor_radius`` (n,).
    """
    raw = load_constellaration_field_raw(n_theta=n_theta, n_phi=n_phi, path=path)
    extra = {"modb_scale": raw["modb_scale"], "minor_radius": raw["minor_radius"]}
    return raw["X"], raw["modb"], raw["plasma_config_id"], extra


def constellaration_surfaces_dataset(
    *, n_theta: int = DEFAULT_N_THETA_FIELD, n_phi: int = DEFAULT_N_PHI_FIELD, path: "str | Path | None" = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """``plasma.constellaration_surfaces`` arrays: boundary (R, Z) -> the 16
    stored flux surfaces' poloidal cross-sections at ``phi=0``.

    Returns ``(X, y, plasma_config_id, extra)``: ``X`` (n, 2, 64, 32),
    ``y`` (n, 2, n_surfaces, 64) (``R``, ``Z`` channels; surface 0 = axis,
    last = boundary), ``extra`` with ``minor_radius`` (n,).
    """
    raw = load_constellaration_field_raw(n_theta=n_theta, n_phi=n_phi, path=path)
    extra = {"minor_radius": raw["minor_radius"]}
    return raw["X"], raw["surfaces"], raw["plasma_config_id"], extra


def mirror_ratio(field: np.ndarray) -> np.ndarray:
    """Per-sample mirror ratio ``(max - min) / (max + min)`` over the last
    two (grid) axes of a ``(n, ny, nx)`` field. Scale-invariant, so it can be
    computed identically on a normalised or raw ``|B|`` array.
    """
    f = np.asarray(field, dtype=np.float64).reshape(field.shape[0], -1)
    fmax, fmin = f.max(axis=1), f.min(axis=1)
    denom = fmax + fmin
    return np.divide(fmax - fmin, denom, out=np.full_like(fmax, np.nan), where=denom != 0)


def axis_centroid(surfaces: np.ndarray) -> np.ndarray:
    """Centroid ``(R, Z)`` of flux-surface index 0 (the magnetic axis),
    averaged over ``theta`` -- ``surfaces`` is ``(n, 2, n_surfaces, n_theta)``.
    Returns ``(n, 2)``.
    """
    axis = np.asarray(surfaces)[:, :, 0, :]  # (n, 2, n_theta)
    return axis.mean(axis=2)


def axis_position_error(surfaces_true: np.ndarray, surfaces_pred: np.ndarray, minor_radius: np.ndarray) -> np.ndarray:
    """Per-sample magnetic-axis centroid displacement, normalised by the
    boundary minor radius. Returns ``(n,)``; aggregate with ``np.median``.
    """
    c_true = axis_centroid(surfaces_true)
    c_pred = axis_centroid(surfaces_pred)
    dist = np.linalg.norm(c_pred - c_true, axis=1)
    a = np.asarray(minor_radius, dtype=np.float64)
    return np.divide(dist, a, out=np.full_like(dist, np.nan), where=a != 0)
