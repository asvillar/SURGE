"""Autoregressive rollout helper for trained field-to-field surrogates.

A single-step surrogate (trained to predict ``B(t + dt)`` from ``B(t)``)
can be pushed to a longer horizon by feeding its own prediction back in as
the next input. This is the standard way to reach a horizon larger than
what one training pair covers, and it is shared across every SURGE example
that reports a rollout (rather than being re-implemented per script) so the
error-accumulation bookkeeping — and its test coverage — lives in one place.

See ``examples/thewell_grayscott_study.py --rollout`` for the reference
driver: train one U-Net at a modest step, then call
:func:`autoregressive_rollout` to reach the README's 160-step headline
horizon and compare against :func:`persistence_rel_l2`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..metrics import relative_l2

__all__ = [
    "autoregressive_rollout", "persistence_rel_l2",
    "sample_rollout_starts", "choose_dt", "regime_masks",
    "layout_case_grid_frames", "stack_case_panel_frames", "match_frame",
    "valid_prediction_time", "climate_histogram_error",
]


def autoregressive_rollout(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    truths: np.ndarray,
    residual: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll ``predict_fn`` forward autoregressively for ``len(truths)`` steps.

    Each application's *own output* (not the ground truth) becomes the next
    input, so error accumulation is honest — this is what distinguishes a
    rollout from ``len(truths)`` independent single-shot predictions.

    Parameters
    ----------
    predict_fn:
        Maps one field (same shape as ``initial_state``) to the next field,
        or to the field *increment* when ``residual=True`` (the state
        update is then ``state + predict_fn(state)``). Any batching the
        underlying model needs is ``predict_fn``'s own responsibility.
    initial_state:
        Starting field for a single sample (no leading batch axis).
    truths:
        Ground-truth fields at horizons ``1..N``, shape
        ``(N, *initial_state.shape)``.

    Returns
    -------
    preds : ``(N, *initial_state.shape)`` ndarray
        The rolled-out state after each application.
    rel_l2 : ``(N,)`` ndarray
        Relative L2 error of ``preds[k]`` vs ``truths[k]`` at each horizon.
    """
    truths = np.asarray(truths)
    n_steps = len(truths)
    state = np.asarray(initial_state, dtype=np.float32)
    preds = np.empty((n_steps, *state.shape), dtype=np.float32)
    rel_l2 = np.empty(n_steps, dtype=np.float64)
    for k in range(n_steps):
        out = np.asarray(predict_fn(state), dtype=np.float32).reshape(state.shape)
        state = state + out if residual else out
        preds[k] = state
        rel_l2[k] = relative_l2(truths[k], state)
    return preds, rel_l2


def persistence_rel_l2(initial_state: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Relative L2 of the "predict no change" baseline at each horizon.

    Anchors a rollout error curve honestly: a surrogate only earns its
    keep at a given horizon if its rollout error is below this line.
    """
    truths = np.asarray(truths)
    init = np.asarray(initial_state, dtype=np.float32)
    return np.array([relative_l2(t, init) for t in truths])


def valid_prediction_time(
    errors: np.ndarray, threshold: float, dt_lyapunov: float,
    horizon_lyapunov: float | None = None,
) -> float:
    """First crossing time (in Lyapunov-time units) of a normalised error curve.

    Chaotic-system rollouts (e.g. Lorenz-63) are conventionally scored by
    "valid prediction time" (VPT): how long an autoregressive rollout tracks
    the truth before a normalised error exceeds a threshold, measured in
    Lyapunov times so the number is comparable across systems with different
    divergence rates (Pathak et al. 2018, PRL 120, 024102).

    Parameters
    ----------
    errors:
        ``(n_steps,)`` normalised error at each rolled-out step (step ``k``
        is ``k + 1`` model applications past the initial condition).
    threshold:
        Crossing threshold (e.g. 0.4).
    dt_lyapunov:
        Lyapunov times per step (``step_dt / lyapunov_time``), converting a
        step index into a Lyapunov-time value.
    horizon_lyapunov:
        Value returned when ``errors`` never crosses ``threshold`` — i.e. the
        rollout's total horizon in Lyapunov times. Defaults to
        ``len(errors) * dt_lyapunov`` (consistent, but an explicit horizon
        avoids rounding drift when the caller already knows it).

    Returns
    -------
    vpt : float, in Lyapunov-time units.
    """
    errors = np.asarray(errors)
    exceed = np.nonzero(errors > threshold)[0]
    if exceed.size == 0:
        return float(
            horizon_lyapunov if horizon_lyapunov is not None
            else len(errors) * dt_lyapunov)
    return float((int(exceed[0]) + 1) * dt_lyapunov)


def climate_histogram_error(
    truth_values: np.ndarray, pred_values: np.ndarray, bins: int = 50,
) -> float:
    """Relative L1 error between the truth/prediction value histograms.

    Scores whether a long rollout reproduces a chaotic system's invariant
    ("climate") statistics — e.g. the Lorenz-63 z-coordinate distribution —
    even at horizons where pointwise trajectory tracking has long since
    diverged (VPT alone would read as a flat failure there).

    Parameters
    ----------
    truth_values, pred_values:
        Arrays of any shape; flattened before histogramming (e.g. all steps
        of all rolled-out trajectories).
    bins:
        Number of histogram bins, shared by both distributions over the
        union of their value ranges.

    Returns
    -------
    error : ``sum(|pred_hist - truth_hist|) / sum(|truth_hist|)``, both
        histograms density-normalised over the same bin edges.
    """
    truth_values = np.asarray(truth_values, dtype=np.float64).ravel()
    pred_values = np.asarray(pred_values, dtype=np.float64).ravel()
    lo = min(truth_values.min(), pred_values.min())
    hi = max(truth_values.max(), pred_values.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    truth_hist, _ = np.histogram(truth_values, bins=edges, density=True)
    pred_hist, _ = np.histogram(pred_values, bins=edges, density=True)
    denom = float(np.sum(np.abs(truth_hist))) + 1e-12
    return float(np.sum(np.abs(pred_hist - truth_hist)) / denom)


def sample_rollout_starts(
    n_traj: int,
    trajectory_length: int,
    total_horizon: int,
    settle_steps: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_traj`` rollout start indices from the *settled* regime.

    Many stored simulations (e.g. TheWell's Gray-Scott) begin with a
    pattern-formation transient that the training pairs never sample
    (``WellDataset`` strides uniformly over the whole trajectory, which is
    almost all settled-regime). Starting a rollout at ``t=0`` therefore
    scores the model on an out-of-distribution regime it never saw. Each
    returned index is drawn uniformly from
    ``[settle_steps, trajectory_length - total_horizon - 1]`` so every
    rollout both starts settled and has room to run ``total_horizon`` steps.

    Raises
    ------
    ValueError
        If the trajectory is too short to leave a valid start window.
    """
    lo = settle_steps
    hi = trajectory_length - total_horizon - 1
    if hi < lo:
        raise ValueError(
            f"trajectory_length={trajectory_length} leaves no valid start in "
            f"[{lo}, {hi}] for total_horizon={total_horizon} past "
            f"settle_steps={settle_steps}")
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi + 1, size=n_traj)


def choose_dt(
    dt_table: "dict[int, tuple[float, float]]",
) -> tuple[int, bool]:
    """Pick the smallest Δt whose model error beats persistence.

    Parameters
    ----------
    dt_table:
        ``{dt: (model_rel_l2, persistence_rel_l2)}``.

    Returns
    -------
    chosen_dt, beats_persistence:
        The smallest ``dt`` with ``model_rel_l2 < persistence_rel_l2``, or
        (if no candidate beats persistence) the *largest* ``dt`` offered
        with ``beats_persistence=False`` — rolling out a model that loses
        to "predict no change" at every candidate step is still reported,
        never silently swapped for a better-looking gate.
    """
    winners = sorted(dt for dt, (model, pers) in dt_table.items() if model < pers)
    if winners:
        return winners[0], True
    return max(dt_table), False


def regime_masks(
    t0: np.ndarray, cutoff: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Split pair start indices ``t0`` into transient vs settled regimes.

    Gray-Scott (and similar reaction-diffusion systems stored by TheWell)
    spends its early stored steps in a pattern-*forming* transient before
    settling into the steady dynamics that dominate a trajectory. A single
    persistence/rel-L2 number averaged over pairs sampled uniformly across
    the whole trajectory blends these two regimes, which can make a
    surrogate's headline advantage largely an artifact of how many
    transient-regime pairs got sampled (see
    ``examples/thewell_grayscott_study.py --regime-split``).

    Parameters
    ----------
    t0:
        Pair start index (stored step) for each sampled pair, shape ``(n,)``.
    cutoff:
        Steps below this are "transient", at or above it "settled".

    Returns
    -------
    transient_mask, settled_mask : ``(n,)`` boolean ndarrays, ``t0 < cutoff``
        and ``t0 >= cutoff`` respectively (partition ``t0``: every index is
        in exactly one).
    fractions : ``{"transient": frac, "settled": frac}``
        Fraction of ``t0`` in each regime (0.0 for an empty ``t0``).
    """
    t0 = np.asarray(t0)
    transient_mask = t0 < cutoff
    settled_mask = ~transient_mask
    n = len(t0)
    fractions = {
        "transient": float(transient_mask.sum()) / n if n else 0.0,
        "settled": float(settled_mask.sum()) / n if n else 0.0,
    }
    return transient_mask, settled_mask, fractions


def layout_case_grid_frames(
    truths: np.ndarray, preds: np.ndarray, persistence: np.ndarray,
) -> np.ndarray:
    """Arrange a multi-case rollout into per-application grid-of-panels frames.

    A multi-case rollout GIF (one row per held-out trajectory, one column
    per panel) needs the same reshuffle every time it's built — this is
    that reshuffle, shared so the grid layout and its test coverage live in
    one place rather than being re-derived inline in a driver script.

    Parameters
    ----------
    truths, preds:
        ``(n_cases, n_apps, H, W)`` ground truth and rolled-out fields.
    persistence:
        ``(n_cases, H, W)`` frozen initial state per case, broadcast across
        every application (persistence never changes after t=0).

    Returns
    -------
    frames : ``(n_apps, n_cases, 4, H, W)`` ndarray
        Panel order per case is ``[truth, rollout, persistence,
        signed error]`` where ``error = preds - truths``; indexing
        ``frames[k]`` gives everything needed to draw application ``k``'s
        grid frame.
    """
    truths = np.asarray(truths)
    preds = np.asarray(preds)
    persistence = np.asarray(persistence)
    n_cases, n_apps, h, w = truths.shape
    if preds.shape != truths.shape:
        raise ValueError(f"preds shape {preds.shape} != truths shape {truths.shape}")
    if persistence.shape != (n_cases, h, w):
        raise ValueError(
            f"persistence shape {persistence.shape} != expected "
            f"{(n_cases, h, w)}")
    persistence_b = np.broadcast_to(
        persistence[:, None], (n_cases, n_apps, h, w))
    error = preds - truths
    return stack_case_panels(truths, preds, persistence_b, error)


def stack_case_panels(*panels: np.ndarray) -> np.ndarray:
    """Stack arbitrary same-shape ``(n_cases, n_apps, H, W)`` panel arrays
    into ``(n_apps, n_cases, n_panels, H, W)`` frames.

    Generalizes :func:`layout_case_grid_frames`'s fixed 4-panel
    ``[truth, rollout, persistence, error]`` layout to any number and order
    of panels — e.g. a short-step rollout GIF's ``[truth, rollout, error]``
    with no persistence column — so a new panel arrangement doesn't need its
    own reshuffle.
    """
    arrs = [np.asarray(p) for p in panels]
    stacked = np.stack(arrs, axis=2)  # (n_cases, n_apps, n_panels, H, W)
    return np.moveaxis(stacked, 0, 1)


def match_frame(
    target: np.ndarray, frames: np.ndarray, stride: int = 1,
    side: int | None = None,
) -> tuple[int, float]:
    """Brute-force nearest match of ``target`` against candidate ``frames``.

    ``WellDataset``'s pair sampler (used by ``load_grayscott``) discards which
    ``(file, trajectory, t0)`` each sampled pair came from. To recover it for
    a specific field of interest (e.g. a headline figure's test sample),
    downsample every candidate row of ``frames`` the same way the study does
    (``[::stride, ::stride][:side, :side]``) and return the index of the
    closest match plus its max-abs-diff — an exact provenance match should
    have a diff near zero (float32 round-trip), anything else is a
    coincidental near-match.

    Parameters
    ----------
    target:
        The field to identify, shape ``(side, side)`` (or already the
        downsampled shape expected after applying ``stride``/``side``).
    frames:
        Candidate frames, shape ``(N, H, W)`` at native (pre-downsample)
        resolution.
    stride, side:
        Same downsampling ``load_grayscott`` applies:
        ``frames[:, ::stride, ::stride][:, :side, :side]``. ``side`` defaults
        to ``target``'s trailing dimension.

    Returns
    -------
    index, max_abs_diff:
        Index into ``frames`` of the closest match and its max-abs-diff to
        ``target``.
    """
    target = np.asarray(target, dtype=np.float32)
    side = side if side is not None else target.shape[-1]
    frames = np.asarray(frames)
    ds = frames[:, ::stride, ::stride][:, :side, :side].astype(np.float32)
    diffs = np.max(np.abs(ds - target[None]), axis=(1, 2))
    idx = int(np.argmin(diffs))
    return idx, float(diffs[idx])
