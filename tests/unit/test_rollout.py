"""Tests for surge.inference.rollout — the autoregressive rollout helper."""
from __future__ import annotations

import numpy as np
import pytest

from surge.inference.rollout import (
    autoregressive_rollout,
    choose_dt,
    layout_case_grid_frames,
    match_frame,
    persistence_rel_l2,
    regime_masks,
    sample_rollout_starts,
    stack_case_panels,
)


def test_rollout_produces_n_fields_and_matching_rel_l2_length():
    rng = np.random.default_rng(0)
    state0 = rng.normal(size=(4, 4)).astype(np.float32)
    n_steps = 5

    # dummy linear model: multiply by a fixed decay factor each step
    def predict_fn(x):
        return 0.9 * x

    truths = np.stack([(0.9 ** (k + 1)) * state0 for k in range(n_steps)])
    preds, rel_l2 = autoregressive_rollout(predict_fn, state0, truths)

    assert preds.shape == (n_steps, *state0.shape)
    assert rel_l2.shape == (n_steps,)
    # the dummy model exactly matches the synthetic truth trajectory
    np.testing.assert_allclose(rel_l2, np.zeros(n_steps), atol=1e-5)
    np.testing.assert_allclose(preds, truths, atol=1e-5)


def test_rollout_uses_own_prediction_not_ground_truth():
    """Feeding back an intentionally wrong truth must not change the rollout."""
    state0 = np.ones((3, 3), dtype=np.float32)
    n_steps = 3

    def predict_fn(x):
        return x + 1.0

    correct_truths = np.stack(
        [state0 + (k + 1) for k in range(n_steps)])
    wrong_truths = correct_truths + 100.0  # only used for error scoring

    preds_a, _ = autoregressive_rollout(predict_fn, state0, correct_truths)
    preds_b, rel_l2_b = autoregressive_rollout(predict_fn, state0, wrong_truths)

    # rollout trajectory itself is independent of the truths passed in
    np.testing.assert_allclose(preds_a, preds_b)
    # but the reported error reflects comparison against the (wrong) truths
    assert np.all(rel_l2_b > 0)


def test_rollout_residual_mode_adds_increment_to_state():
    state0 = np.zeros((2, 2), dtype=np.float32)
    n_steps = 4

    def predict_fn(_x):
        return np.ones((2, 2), dtype=np.float32)  # constant increment

    truths = np.stack([np.full((2, 2), k + 1, dtype=np.float32)
                       for k in range(n_steps)])
    preds, rel_l2 = autoregressive_rollout(
        predict_fn, state0, truths, residual=True)

    np.testing.assert_allclose(preds, truths)
    np.testing.assert_allclose(rel_l2, np.zeros(n_steps), atol=1e-6)


def test_persistence_rel_l2_matches_manual_computation():
    state0 = np.array([1.0, 0.0])
    truths = np.stack([np.array([1.0, 1.0]), np.array([0.0, 1.0])])
    rel = persistence_rel_l2(state0, truths)
    expected = np.array([
        np.linalg.norm(state0 - truths[0]) / np.linalg.norm(truths[0]),
        np.linalg.norm(state0 - truths[1]) / np.linalg.norm(truths[1]),
    ])
    np.testing.assert_allclose(rel, expected)


def test_sample_rollout_starts_never_below_settle_steps_and_leaves_room():
    trajectory_length = 1001
    total_horizon = 160
    settle_steps = 200
    starts = sample_rollout_starts(
        n_traj=50, trajectory_length=trajectory_length,
        total_horizon=total_horizon, settle_steps=settle_steps, seed=0)
    assert len(starts) == 50
    assert np.all(starts >= settle_steps)
    assert np.all(starts + total_horizon < trajectory_length)


def test_sample_rollout_starts_raises_when_no_valid_window():
    with pytest.raises(ValueError):
        sample_rollout_starts(
            n_traj=1, trajectory_length=100, total_horizon=160,
            settle_steps=200, seed=0)


def test_choose_dt_picks_smallest_dt_that_beats_persistence():
    # dt=20 loses to persistence, dt=40 and dt=80 both beat it -> pick 40
    dt_table = {20: (0.15, 0.014), 40: (0.08, 0.174), 80: (0.05, 0.30)}
    chosen, beats = choose_dt(dt_table)
    assert chosen == 40
    assert beats is True


def test_choose_dt_falls_back_to_largest_when_none_beats_persistence():
    dt_table = {20: (0.15, 0.014), 40: (0.174, 0.028), 80: (0.20, 0.05)}
    chosen, beats = choose_dt(dt_table)
    assert chosen == 80
    assert beats is False


def test_regime_masks_partitions_on_cutoff():
    t0 = np.array([0, 50, 199, 200, 201, 999])
    transient, settled, fractions = regime_masks(t0, cutoff=200)
    np.testing.assert_array_equal(transient, [True, True, True, False, False, False])
    np.testing.assert_array_equal(settled, [False, False, False, True, True, True])
    # every index is in exactly one regime
    assert np.all(transient ^ settled)
    assert fractions["transient"] == pytest.approx(3 / 6)
    assert fractions["settled"] == pytest.approx(3 / 6)


def test_regime_masks_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    t0 = rng.integers(0, 1001, size=37)
    transient, settled, fractions = regime_masks(t0, cutoff=200)
    assert transient.sum() + settled.sum() == len(t0)
    assert fractions["transient"] + fractions["settled"] == pytest.approx(1.0)


def test_regime_masks_empty_input_returns_zero_fractions():
    transient, settled, fractions = regime_masks(np.array([]), cutoff=200)
    assert transient.size == 0
    assert settled.size == 0
    assert fractions == {"transient": 0.0, "settled": 0.0}


def test_layout_case_grid_frames_shape_and_panel_order():
    n_cases, n_apps, h, w = 4, 24, 5, 5
    rng = np.random.default_rng(0)
    truths = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)
    preds = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)
    persistence = rng.normal(size=(n_cases, h, w)).astype(np.float32)

    frames = layout_case_grid_frames(truths, preds, persistence)

    assert frames.shape == (n_apps, n_cases, 4, h, w)
    # panel order: [truth, rollout, persistence, signed error]
    np.testing.assert_allclose(frames[:, :, 0], np.moveaxis(truths, 0, 1))
    np.testing.assert_allclose(frames[:, :, 1], np.moveaxis(preds, 0, 1))
    # persistence is frozen: identical across every application
    for k in range(n_apps):
        np.testing.assert_allclose(frames[k, :, 2], persistence)
    np.testing.assert_allclose(frames[:, :, 3],
                               np.moveaxis(preds - truths, 0, 1))


def test_layout_case_grid_frames_rejects_mismatched_persistence_shape():
    n_cases, n_apps, h, w = 2, 3, 4, 4
    truths = np.zeros((n_cases, n_apps, h, w), dtype=np.float32)
    preds = np.zeros((n_cases, n_apps, h, w), dtype=np.float32)
    bad_persistence = np.zeros((n_cases, h, w + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        layout_case_grid_frames(truths, preds, bad_persistence)


def test_stack_case_panels_shape_and_order_three_panels():
    n_cases, n_apps, h, w = 3, 5, 4, 4
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)
    b = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)
    c = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)

    frames = stack_case_panels(a, b, c)

    assert frames.shape == (n_apps, n_cases, 3, h, w)
    np.testing.assert_allclose(frames[:, :, 0], np.moveaxis(a, 0, 1))
    np.testing.assert_allclose(frames[:, :, 1], np.moveaxis(b, 0, 1))
    np.testing.assert_allclose(frames[:, :, 2], np.moveaxis(c, 0, 1))


def test_stack_case_panels_agrees_with_layout_case_grid_frames():
    n_cases, n_apps, h, w = 2, 3, 4, 4
    rng = np.random.default_rng(1)
    truths = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)
    preds = rng.normal(size=(n_cases, n_apps, h, w)).astype(np.float32)
    persistence = rng.normal(size=(n_cases, h, w)).astype(np.float32)

    expected = layout_case_grid_frames(truths, preds, persistence)
    persistence_b = np.broadcast_to(persistence[:, None], (n_cases, n_apps, h, w))
    actual = stack_case_panels(truths, preds, persistence_b, preds - truths)

    np.testing.assert_allclose(actual, expected)


def test_match_frame_finds_exact_frame_with_downsampling():
    rng = np.random.default_rng(2)
    frames = rng.normal(size=(3, 8, 8)).astype(np.float32)
    stride, side = 2, 4
    target = frames[1, ::stride, ::stride][:side, :side]

    idx, diff = match_frame(target, frames, stride=stride, side=side)

    assert idx == 1
    assert diff < 1e-6


def test_match_frame_nonzero_diff_when_no_exact_match():
    frames = np.zeros((3, 4, 4), dtype=np.float32)
    frames[0] = 1.0
    frames[1] = 2.0
    frames[2] = 3.0
    target = np.full((4, 4), 2.4, dtype=np.float32)

    idx, diff = match_frame(target, frames)

    assert idx == 1
    assert diff == pytest.approx(0.4, abs=1e-6)
