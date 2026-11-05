"""Tests for surge.benchmarks.paths (portable dataset-cache resolution)."""
from __future__ import annotations

from pathlib import Path

from surge.benchmarks import paths


def test_surge_data_dir_env_override_wins(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    assert paths.bench_data_root() == tmp_path / "benchmarks"


def test_checkout_resolves_under_repo_data_datasets(monkeypatch):
    monkeypatch.delenv("SURGE_DATA_DIR", raising=False)
    root = paths.repo_root()
    assert root is not None, "tests are expected to run from a SURGE checkout"
    assert paths.bench_data_root() == root / "data" / "datasets" / "benchmarks"


def test_fallback_to_xdg_cache_outside_a_checkout(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SURGE_DATA_DIR", raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda: None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert paths.bench_data_root() == tmp_path / "surge" / "datasets" / "benchmarks"
