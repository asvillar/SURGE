from __future__ import annotations

import numpy as np
import pytest

from surge.benchmarks import leaderboard
from surge.benchmarks.registry import benchmark_info, benchmark_metadata


def _write_npz(path, *, x_shape=(3, 2), y_shape=(3,)):
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.arange(np.prod(x_shape), dtype=float).reshape(x_shape)
    y = np.arange(np.prod(y_shape), dtype=float).reshape(y_shape)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def test_benchmark_data_root_resolves_inside_repo():
    root = leaderboard._bench_data_root()
    repo_root = root.parents[2]
    assert root == repo_root / "data" / "datasets" / "benchmarks"


@pytest.mark.parametrize(
    ("loader_name", "cache_rel"),
    [
        ("_load_burgers_1d", "pde/burgers_1d.npz"),
        ("_load_lorenz63", "sequence/lorenz63.npz"),
        ("_load_qlknn_transport", "plasma/qlknn_transport.npz"),
        ("_load_plasma_stability", "classification/plasma_stability.npz"),
    ],
)
def test_cached_npz_loaders_do_not_touch_generators_or_network(monkeypatch, tmp_path, loader_name, cache_rel):
    # Lorenz cache must be valid one-step 3->3 pairs (stale shapes are regenerated).
    shapes = {"x_shape": (4, 3), "y_shape": (4, 3)} if loader_name == "_load_lorenz63" else {}
    expected_X, expected_y = _write_npz(tmp_path / cache_rel, **shapes)
    monkeypatch.setattr(leaderboard, "_bench_data_root", lambda: tmp_path)

    def fail(*args, **kwargs):  # pragma: no cover - only called on regression
        raise AssertionError("cache hit should not call generator, network, or optional package")

    monkeypatch.setattr(leaderboard, "_fetch_openml", fail)
    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setitem(__import__("sys").modules, "fusion_surrogates", None)

    X, y = getattr(leaderboard, loader_name)()
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)


def test_cached_cmod_grouped_split_does_not_touch_network(monkeypatch, tmp_path):
    """The re-posed C-Mod loader is a fixed-split loader with its own cache layout."""
    cache = tmp_path / "plasma" / "cmod_density_limit_grouped.npz"
    cache.parent.mkdir(parents=True)
    arrays = {k: np.arange(i, i + 6, dtype=float).reshape(3, 2) if k.startswith("X") else np.arange(i, i + 3, dtype=float)
              for i, k in enumerate(("X_train", "y_train", "X_test", "y_test"))}
    np.savez_compressed(cache, **arrays)
    monkeypatch.setattr(leaderboard, "_bench_data_root", lambda: tmp_path)

    def fail(*args, **kwargs):  # pragma: no cover - only called on regression
        raise AssertionError("cache hit should not call generator, network, or optional package")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    X_train, y_train, X_test, y_test = leaderboard._load_cmod_density_limit()
    np.testing.assert_array_equal(X_train, arrays["X_train"])
    np.testing.assert_array_equal(y_test, arrays["y_test"])


def test_benchmark_info_exposes_structured_metadata():
    info = benchmark_info("pde.burgers_1d")
    meta = benchmark_metadata("pde.burgers_1d")

    assert info["cache_path"] == "data/datasets/benchmarks/pde/burgers_1d.npz"
    assert info["resource_expectation"]["memory_tier"] == "medium"
    assert meta["access"] == "generated_cached"
    assert meta["task_family"] == "field"


def test_skip_result_is_structured():
    result = leaderboard._skip_result(
        "tabular.iris",
        "missing.model",
        reason="not registered",
        stage="model_lookup",
    )

    assert not result.passed
    assert result.metrics == {}
    assert result.extra["status"] == "skipped"
    assert result.extra["skip_stage"] == "model_lookup"
    assert result.extra["benchmark_metadata"]["task_type"] == "classification"
