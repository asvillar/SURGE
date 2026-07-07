"""Cross-platform path and portability tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from surge.utils import posix_str


class TestPosixStr:
    def test_forward_slashes_for_relative_paths(self):
        assert posix_str(Path("runs/tag/summary.json")) == "runs/tag/summary.json"

    def test_absolute_paths_use_forward_slashes(self, tmp_path: Path):
        p = tmp_path / "runs" / "tag"
        assert "/" in posix_str(p)
        assert "\\" not in posix_str(p)

    def test_never_emits_backslashes(self, tmp_path: Path):
        nested = tmp_path / "runs" / "tag" / "model.pt"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.touch()
        assert "\\" not in posix_str(nested)
        assert "/" in posix_str(nested)

    def test_roundtrip_readable_via_pathlib(self, tmp_path: Path):
        """Forward-slash paths stored in JSON open correctly on every OS."""
        target = tmp_path / "nested" / "artifact.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"ok": true}', encoding="utf-8")

        stored = posix_str(target)
        payload = {"model": stored}
        serialized = json.dumps(payload)
        restored = json.loads(serialized)["model"]

        assert Path(restored).read_text(encoding="utf-8") == '{"ok": true}'

    def test_dataset_summary_uses_posix_paths(self, tmp_path: Path):
        from surge.dataset import SurrogateDataset

        csv_path = tmp_path / "nested" / "data.csv"
        csv_path.parent.mkdir(parents=True)
        ds = SurrogateDataset()
        ds.file_path = csv_path
        summary = ds.summary()
        assert "\\" not in summary["file_path"]
        assert "/" in summary["file_path"]


class TestConstellarationCanonicalSplit:
    def test_deterministic_and_cached(self, tmp_path: Path, monkeypatch):
        from surge.benchmarks import leaderboard as lb

        cache_root = tmp_path / "benchmarks"
        monkeypatch.setattr(lb, "_bench_data_root", lambda: cache_root)

        train_a, test_a = lb.constellaration_canonical_split(100, seed=42)
        train_b, test_b = lb.constellaration_canonical_split(100, seed=42)

        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)
        assert len(train_a) + len(test_a) == 100
        assert len(test_a) == 20  # default 80/20
        assert len(set(train_a) & set(test_a)) == 0

        cache_file = (
            cache_root / "plasma" / "constellaration" / "split_n100_seed42_test0.2.npz"
        )
        assert cache_file.exists()

    def test_different_seeds_differ(self, tmp_path: Path, monkeypatch):
        from surge.benchmarks import leaderboard as lb

        monkeypatch.setattr(lb, "_bench_data_root", lambda: tmp_path / "benchmarks")

        train_a, test_a = lb.constellaration_canonical_split(50, seed=1)
        train_b, test_b = lb.constellaration_canonical_split(50, seed=2)
        assert not np.array_equal(test_a, test_b)


class TestBenchmarkTimestamp:
    def test_timestamp_has_no_colons(self):
        from surge.benchmarks.base import benchmark_timestamp

        ts = benchmark_timestamp()
        assert ":" not in ts
        assert "T" in ts
        assert ts.endswith("Z")

    def test_save_creates_windows_safe_directory(self, tmp_path: Path):
        from surge.benchmarks.base import BenchmarkResult

        result = BenchmarkResult(
            benchmark_key="synthetic.regression_1d",
            tier="0",
            task_type="regression",
            metrics={"test_r2": 0.9},
            passed=True,
            model_key="sklearn.ridge",
        )
        path = result.save(root=tmp_path)
        assert ":" not in str(path)
        assert path.exists()


class TestDatagenWindowsGuard:
    def test_windows_bash_script_guard(self, monkeypatch):
        from surge.datagen import DataGenerator

        gen = DataGenerator(dry_run=False, use_python_replacement=False, bin_dir="/nope")
        monkeypatch.setattr(os, "name", "nt")
        with pytest.raises(RuntimeError, match="not supported on Windows"):
            gen._call_replace("x", 1, "input.py")

    def test_default_bin_dir_uses_path_home(self, monkeypatch):
        from surge.datagen import DataGenerator

        fake_home = Path("/fake/home")
        monkeypatch.setattr(
            "surge.datagen.generator.Path.home", staticmethod(lambda: fake_home)
        )
        gen = DataGenerator()
        assert gen.bin_dir == str(fake_home / "tools" / "bin")
