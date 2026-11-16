"""
Tests for DataGenerator class.

Tests parameter sampling, file replacement, and batch generation.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from surge.datagen import DataGenerator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ranges():
    """Sample parameter ranges for testing."""
    return [[1.0, 10.0], [0.5, 2.0], [0.1, 1.0]]


@pytest.fixture
def sample_params():
    """Sample parameter names."""
    return ['ntor', 'pscale', 'batemanscale']


class TestDataGeneratorInit:
    """Test DataGenerator initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        gen = DataGenerator()
        assert gen is not None
        assert gen.dry_run is False
        # No script directory configured: built-in replacement is the default.
        assert gen.bin_dir is None
        assert gen.use_python_replacement is True

    def test_default_replaces_without_external_script(self, temp_dir, monkeypatch):
        """A plain DataGenerator() edits a variable with no external script."""
        monkeypatch.delenv("SURGE_DATAGEN_BIN", raising=False)
        inp = os.path.join(temp_dir, "input.dat")
        with open(inp, "w") as f:
            f.write("alpha = 1\nbeta = 2\n")
        DataGenerator()._call_replace("alpha", 7, inp)
        with open(inp) as f:
            text = f.read()
        assert "alpha = 7" in text and "beta = 2" in text

    def test_explicit_script_dir_disables_builtin(self):
        gen = DataGenerator(bin_dir="/some/dir")
        assert gen.bin_dir == "/some/dir"
        assert gen.use_python_replacement is False
    
    def test_init_dry_run(self):
        """Test initialization with dry_run."""
        gen = DataGenerator(dry_run=True)
        assert gen.dry_run is True
    
    def test_init_python_replacement(self):
        """Test initialization with use_python_replacement."""
        gen = DataGenerator(use_python_replacement=True)
        assert gen.use_python_replacement is True


class TestMakeSamples:
    """Test _make_samples method."""
    
    def test_lhs_sampling_basic(self, temp_dir):
        """Test basic LHS sampling."""
        gen = DataGenerator(dry_run=True)
        ranges = [[0.0, 1.0], [0.0, 1.0]]
        integer_mask = [False, False]
        
        samples = gen._make_samples(
            dim=2,
            n=10,
            ranges=ranges,
            integer_mask=integer_mask,
            method='lhs',
            seed=42
        )
        
        assert samples.shape == (10, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
    
    def test_lhs_sampling_integer(self, temp_dir):
        """Test LHS sampling with integer parameters."""
        gen = DataGenerator(dry_run=True)
        ranges = [[1, 10], [0.5, 2.0]]
        integer_mask = [True, False]
        
        samples = gen._make_samples(
            dim=2,
            n=10,
            ranges=ranges,
            integer_mask=integer_mask,
            method='lhs',
            seed=42
        )
        
        assert samples.shape == (10, 2)
        # First parameter should be integers
        assert np.all(samples[:, 0] == samples[:, 0].astype(int))
        assert np.all(samples[:, 0] >= 1)
        assert np.all(samples[:, 0] <= 10)
        # Second parameter should be floats
        assert np.all(samples[:, 1] >= 0.5)
        assert np.all(samples[:, 1] <= 2.0)
    
    def test_lhs_sampling_log_space(self, temp_dir):
        """Test LHS sampling in log space."""
        gen = DataGenerator(dry_run=True)
        ranges = [[1.0, 100.0], [0.5, 2.0]]
        integer_mask = [False, False]
        log_space = [True, False]
        
        samples = gen._make_samples(
            dim=2,
            n=10,
            ranges=ranges,
            integer_mask=integer_mask,
            method='lhs',
            seed=42,
            log_space=log_space
        )
        
        assert samples.shape == (10, 2)
        # First parameter should be in log space (concentrated at lower values)
        assert np.all(samples[:, 0] >= 1.0)
        assert np.all(samples[:, 0] <= 100.0)
        # Log-space samples should have lower median than linear
        # (most samples should be < 10 for range [1, 100])
        median_log = np.median(samples[:, 0])
        assert median_log < 50.0  # Should be biased toward lower values
        # Second parameter should be linear
        assert np.all(samples[:, 1] >= 0.5)
        assert np.all(samples[:, 1] <= 2.0)
    
    def test_random_sampling_basic(self, temp_dir):
        """Test basic random sampling."""
        gen = DataGenerator(dry_run=True)
        ranges = [[0.0, 1.0], [0.0, 1.0]]
        integer_mask = [False, False]
        
        samples = gen._make_samples(
            dim=2,
            n=10,
            ranges=ranges,
            integer_mask=integer_mask,
            method='random',
            seed=42
        )
        
        assert samples.shape == (10, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
    
    def test_random_sampling_reproducibility(self, temp_dir):
        """Test that random sampling is reproducible with same seed."""
        gen = DataGenerator(dry_run=True)
        ranges = [[0.0, 1.0], [0.0, 1.0]]
        integer_mask = [False, False]
        
        samples1 = gen._make_samples(
            dim=2, n=10, ranges=ranges, integer_mask=integer_mask,
            method='random', seed=42
        )
        samples2 = gen._make_samples(
            dim=2, n=10, ranges=ranges, integer_mask=integer_mask,
            method='random', seed=42
        )
        
        np.testing.assert_array_equal(samples1, samples2)


class TestGenerate:
    """Test generate method (simple case generation)."""
    
    def test_generate_basic(self, temp_dir, sample_params, sample_ranges):
        """Test basic batch generation."""
        gen = DataGenerator(dry_run=True, use_python_replacement=True)
        
        # Create a dummy input file
        input_file = Path(temp_dir) / "test_input.txt"
        input_file.write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        
        integer_mask = [True, False, False]
        
        results = gen.generate(
            inpnames=sample_params,
            inputfilename="test_input.txt",
            ranges=sample_ranges,
            integer_mask=integer_mask,
            n_samples=3,
            method='lhs',
            out_dir=temp_dir,
            base_case_dir=temp_dir,
            seed=42,
            confirm_dirs=False
        )
        
        assert len(results) == 3
        assert gen.last_batch_path is not None
    
    def test_generate_creates_batch_directory(self, temp_dir, sample_params, sample_ranges):
        """Test that generate creates batch directory."""
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        
        batch_root = Path(temp_dir) / "batches"
        batch_root.mkdir()
        
        input_file = batch_root / "test_input.txt"
        input_file.write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        
        integer_mask = [True, False, False]
        
        results = gen.generate(
            inpnames=sample_params,
            inputfilename="test_input.txt",
            ranges=sample_ranges,
            integer_mask=integer_mask,
            n_samples=2,
            method='random',
            out_dir=temp_dir,
            batch_root=str(batch_root),
            base_case_dir=str(batch_root),
            seed=42,
            confirm_dirs=False
        )
        
        # Should create batch1 directory
        batch_paths = [d for d in batch_root.iterdir() if d.is_dir() and d.name.startswith('batch')]
        assert len(batch_paths) == 1


class TestSamplesFileGeneration:
    """Test that samples.npz and metadata are saved correctly."""

    def test_samples_file_saved(self, temp_dir, sample_params, sample_ranges):
        """samples.npz is written into the batch folder."""
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        base = Path(temp_dir) / "base"
        base.mkdir()
        (base / "input.dat").write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        out_dir = Path(temp_dir) / "output"

        gen.generate(
            inpnames=sample_params,
            inputfilename="input.dat",
            ranges=sample_ranges,
            integer_mask=[True, False, False],
            n_samples=3,
            out_dir=str(out_dir),
            base_case_dir=str(base),
            seed=42,
        )

        samples_file = Path(gen.last_batch_path) / "samples.npz"
        assert samples_file.exists()
        data = np.load(samples_file)
        assert data["X"].shape == (3, 3)
        assert len(data["names"]) == 3


class TestGenerateDefaults:
    """out_dir, batch_root, seed and confirmation defaults."""

    def _kwargs(self, temp_dir, sample_params, sample_ranges):
        base = Path(temp_dir) / "base"
        base.mkdir(exist_ok=True)
        (base / "input.dat").write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        return dict(
            inpnames=sample_params,
            inputfilename="input.dat",
            ranges=sample_ranges,
            n_samples=3,
            base_case_dir=str(base),
        )

    def test_out_dir_required(self, temp_dir, sample_params, sample_ranges):
        gen = DataGenerator(use_python_replacement=True)
        with pytest.raises(ValueError, match="out_dir is required"):
            gen.generate(**self._kwargs(temp_dir, sample_params, sample_ranges))

    def test_batch_root_defaults_under_out_dir(self, temp_dir, sample_params, sample_ranges):
        gen = DataGenerator(use_python_replacement=True)
        out_dir = Path(temp_dir) / "out"
        gen.generate(out_dir=str(out_dir), seed=1, **self._kwargs(temp_dir, sample_params, sample_ranges))
        assert Path(gen.last_batch_path).parent == out_dir / "batches"

    def test_no_prompt_by_default(self, temp_dir, sample_params, sample_ranges, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("input() must not be called by default")

        monkeypatch.setattr("builtins.input", _boom)
        gen = DataGenerator(use_python_replacement=True)
        gen.generate(out_dir=str(Path(temp_dir) / "out"), seed=1,
                     **self._kwargs(temp_dir, sample_params, sample_ranges))

    @pytest.mark.parametrize("method", ["lhs", "random"])
    def test_seed_reproducible_in_generate(self, temp_dir, sample_params, sample_ranges, method):
        runs = []
        for k in range(2):
            gen = DataGenerator(use_python_replacement=True)
            gen.generate(out_dir=str(Path(temp_dir) / f"o{k}"), method=method, seed=7,
                         **self._kwargs(temp_dir, sample_params, sample_ranges))
            runs.append(np.load(Path(gen.last_batch_path) / "samples.npz")["X"])
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_make_samples_lhs_seeded(self, sample_ranges):
        gen = DataGenerator(use_python_replacement=True)
        args = (3, 5, sample_ranges, [False] * 3, "lhs")
        np.testing.assert_array_equal(gen._make_samples(*args, seed=3), gen._make_samples(*args, seed=3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
