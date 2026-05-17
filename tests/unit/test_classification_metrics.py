"""Tests for surge.metrics classification helpers (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from surge.metrics import (
    accuracy_score,
    auroc,
    expected_calibration_error,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)


def test_accuracy_and_f1_binary():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.8)
    assert f1_score(y_true, y_pred, average="macro", zero_division=0) == pytest.approx(0.8)


def test_auroc_binary():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.55, 0.9])
    assert auroc(y_true, y_prob) == pytest.approx(1.0)


def test_log_loss_binary():
    y_true = np.array([0, 1])
    y_prob = np.array([[0.8, 0.2], [0.1, 0.9]])
    ll = log_loss(y_true, y_prob)
    assert ll < 0.5
    assert ll > 0.0


def test_top_k_accuracy_multiclass():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=40)
    y_prob = rng.random((40, 5))
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    k1 = top_k_accuracy_score(y_true, y_prob, k=1)
    acc = accuracy_score(y_true, y_prob.argmax(axis=1))
    assert k1 == pytest.approx(acc)


def test_expected_calibration_error_perfect():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.05] * 50 + [0.95] * 50)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05
