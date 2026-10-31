"""Benchmark -> model list wiring."""
import pytest

pytest.importorskip("torch")

from surge.benchmarks.leaderboard import _BENCHMARK_MODEL_OVERRIDES  # noqa: E402


def test_dino_probe_in_cifar10_sweep_only():
    assert "pytorch.dino_linear_probe" in _BENCHMARK_MODEL_OVERRIDES["vision.cifar10"]
    assert "pytorch.dino_linear_probe" not in _BENCHMARK_MODEL_OVERRIDES["vision.mnist"]
