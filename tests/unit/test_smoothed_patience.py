"""Tests for rolling-mean (smoothed) early stopping in the residual MLP."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from surge.model.backends.residual_mlp import ResidualMLPModel


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.05 * rng.standard_normal(n))
    return X, y.astype(np.float32)


def test_smoothed_patience_stops_before_max_epochs():
    X, y = _toy()
    Xv, yv = _toy(80, seed=1)
    model = ResidualMLPModel(
        hidden_layers=[16], n_epochs=400, patience=8, patience_window=5,
        random_state=0, verbose=0)
    model.fit(X, y, Xv, yv)
    hist = list(model.training_history)
    assert len(hist) < 400, "plateaued training must stop early"
    assert hist[-1].get("early_stop") is True
    assert "val_loss_smoothed" in hist[-1]


def test_window_one_matches_legacy_signal():
    """patience_window=1: smoothed signal equals raw val loss per epoch."""
    X, y = _toy(150)
    Xv, yv = _toy(50, seed=2)
    model = ResidualMLPModel(
        hidden_layers=[8], n_epochs=25, patience=0, patience_window=1,
        random_state=0, verbose=0)
    model.fit(X, y, Xv, yv)
    for rec in model.training_history:
        if "val_loss" in rec:
            assert rec["val_loss_smoothed"] == pytest.approx(rec["val_loss"])


def test_smoothed_signal_stops_promptly_at_saturation():
    """The rolling-mean signal keeps counting improvement while the trend
    decreases and terminates within patience+window epochs once the mean
    saturates — even though raw epochs still oscillate."""
    from collections import deque

    decreasing = [1.0 - 0.03 * i for i in range(20)]          # real progress
    plateau = [0.4 + (0.05 if i % 2 else -0.05) for i in range(60)]
    losses = decreasing + plateau
    patience, window_len = 8, 5

    window, best, no_improve = deque(maxlen=window_len), float("inf"), 0
    stop = None
    for i, loss in enumerate(losses):
        window.append(loss)
        sig = sum(window) / len(window)
        if sig < best:
            best, no_improve = sig, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stop = i
                break

    assert stop is not None, "must terminate on the plateau"
    # no premature stop during real progress; prompt stop after saturation
    assert 20 <= stop <= 20 + window_len + patience
