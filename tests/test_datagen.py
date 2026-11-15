"""
Tests for DataGenerator class.

Tests parameter sampling, file replacement, batch generation, and equilibria modes.
"""

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
        assert gen.last_batch_dir is not None
    
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
            batch_root=str(batch_root),
            base_case_dir=str(batch_root),
            seed=42,
            confirm_dirs=False
        )
        
        # Should create batch1 directory
        batch_dirs = [d for d in batch_root.iterdir() if d.is_dir() and d.name.startswith('batch')]
        assert len(batch_dirs) == 1


class TestGenerateRunsFromEquilibria:
    """Test generate_runs_from_equilibria method."""
    
    def test_generate_runs_set_mode(self, temp_dir, sample_params, sample_ranges):
        """Test generating runs in 'set' mode."""
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        
        # Create source run directory with equilibria
        source_dir = Path(temp_dir) / "source_run"
        source_dir.mkdir()
        
        # Create two equilibrium directories
        for eq_name in ['case_1300', 'case_1301']:
            eq_dir = source_dir / eq_name
            eq_dir.mkdir()
            input_file = eq_dir / "input.dat"
            input_file.write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        
        integer_mask = [True, False, False]
        out_root = Path(temp_dir) / "output"
        out_root.mkdir()
        
        results = gen.generate_runs_from_equilibria(
            inpnames=sample_params,
            inputfilename="input.dat",
            source_run_dir=str(source_dir),
            ranges=sample_ranges,
            integer_mask=integer_mask,
            n_runs=2,
            method='lhs',
            out_root=str(out_root),
            equilibria_mode='set',
            seed=42
        )
        
        assert len(results) == 2  # Should create 2 runs
        # Check that run directories exist
        run1 = out_root / "run1"
        run2 = out_root / "run2"
        assert run1.exists()
        assert run2.exists()
        # Check that both runs have both equilibria
        assert (run1 / "case_1300").exists()
        assert (run1 / "case_1301").exists()
        assert (run2 / "case_1300").exists()
        assert (run2 / "case_1301").exists()
    
    def test_generate_runs_fixed_mode(self, temp_dir, sample_params, sample_ranges):
        """Test generating runs in 'fixed' mode (single equilibrium)."""
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        
        # Create source run directory with multiple equilibria
        source_dir = Path(temp_dir) / "source_run"
        source_dir.mkdir()
        
        # Create multiple equilibrium directories
        for eq_name in ['case_1300', 'case_1301', 'case_234321']:
            eq_dir = source_dir / eq_name
            eq_dir.mkdir()
            input_file = eq_dir / "input.dat"
            input_file.write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        
        integer_mask = [True, False, False]
        out_root = Path(temp_dir) / "output"
        out_root.mkdir()
        
        results = gen.generate_runs_from_equilibria(
            inpnames=sample_params,
            inputfilename="input.dat",
            source_run_dir=str(source_dir),
            ranges=sample_ranges,
            integer_mask=integer_mask,
            n_runs=3,
            method='lhs',
            out_root=str(out_root),
            equilibria_mode='fixed',
            seed=42
        )
        
        assert len(results) == 3  # Should create 3 runs
        # Check that run directories exist
        for run_idx in [1, 2, 3]:
            run_dir = out_root / f"run{run_idx}"
            assert run_dir.exists()
            # In fixed mode, should only have first equilibrium (case_1300)
            assert (run_dir / "case_1300").exists()
            assert not (run_dir / "case_1301").exists()
            # Check metadata includes equilibrium name
            assert results[run_idx - 1]['equilibrium'] == 'case_1300'
    
    def test_generate_runs_per_case_mode(self, temp_dir, sample_params, sample_ranges):
        """Test generating runs in 'per_case' mode."""
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        
        # Create source run directory with equilibria
        source_dir = Path(temp_dir) / "source_run"
        source_dir.mkdir()
        
        # Create two equilibrium directories
        for eq_name in ['case_1300', 'case_1301']:
            eq_dir = source_dir / eq_name
            eq_dir.mkdir()
            input_file = eq_dir / "input.dat"
            input_file.write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        
        integer_mask = [True, False, False]
        out_root = Path(temp_dir) / "output"
        out_root.mkdir()
        
        results = gen.generate_runs_from_equilibria(
            inpnames=sample_params,
            inputfilename="input.dat",
            source_run_dir=str(source_dir),
            ranges=sample_ranges,
            integer_mask=integer_mask,
            n_runs=2,
            method='lhs',
            out_root=str(out_root),
            equilibria_mode='per_case',
            seed=42
        )
        
        # Should create runs for each equilibrium
        assert len(results) == 4  # 2 equilibria × 2 runs
        # Check structure: case_1300/run1, case_1300/run2, etc.
        assert (out_root / "case_1300" / "run1").exists()
        assert (out_root / "case_1300" / "run2").exists()
        assert (out_root / "case_1301" / "run1").exists()
        assert (out_root / "case_1301" / "run2").exists()
    
    def test_generate_runs_invalid_mode(self, temp_dir, sample_params, sample_ranges):
        """Test that invalid equilibria_mode raises error."""
        gen = DataGenerator(dry_run=True)
        
        source_dir = Path(temp_dir) / "source_run"
        source_dir.mkdir()
        (source_dir / "case_1300").mkdir()
        
        with pytest.raises(ValueError, match="equilibria_mode must be"):
            gen.generate_runs_from_equilibria(
                inpnames=sample_params,
                inputfilename="input.dat",
                source_run_dir=str(source_dir),
                ranges=sample_ranges,
                integer_mask=[True, False, False],
                n_runs=1,
                equilibria_mode='invalid_mode',
                seed=42
            )


class TestCasePrefix:
    """case_prefix selects which sub-directories count as cases."""

    def test_case_prefix_filters_folders(self, temp_dir, sample_params, sample_ranges):
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        source_dir = Path(temp_dir) / "source_run"
        source_dir.mkdir()
        for name in ["case_1300", "other_1"]:
            d = source_dir / name
            d.mkdir()
            (d / "input.dat").write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        kwargs = dict(
            inpnames=sample_params,
            inputfilename="input.dat",
            source_run_dir=str(source_dir),
            ranges=sample_ranges,
            integer_mask=[True, False, False],
            n_runs=1,
            equilibria_mode="set",
            seed=1,
        )
        out_all = Path(temp_dir) / "out_all"
        gen.generate_runs_from_equilibria(out_root=str(out_all), **kwargs)
        assert (out_all / "run1" / "case_1300").exists()
        assert (out_all / "run1" / "other_1").exists()

        out_f = Path(temp_dir) / "out_filtered"
        gen.generate_runs_from_equilibria(out_root=str(out_f), case_prefix="case_", **kwargs)
        assert (out_f / "run1" / "case_1300").exists()
        assert not (out_f / "run1" / "other_1").exists()

        with pytest.raises(RuntimeError, match="No case directories"):
            gen.generate_runs_from_equilibria(
                out_root=str(Path(temp_dir) / "out_none"), case_prefix="zzz", **kwargs
            )


class TestSamplesFileGeneration:
    """Test that samples.npz and metadata are saved correctly."""
    
    def test_samples_file_saved(self, temp_dir, sample_params, sample_ranges):
        """Test that samples.npz is saved."""
        gen = DataGenerator(dry_run=False, use_python_replacement=True)
        
        source_dir = Path(temp_dir) / "source_run"
        source_dir.mkdir()
        eq_dir = source_dir / "case_1300"
        eq_dir.mkdir()
        (eq_dir / "input.dat").write_text("ntor = 5\npscale = 1.0\nbatemanscale = 0.5\n")
        
        out_root = Path(temp_dir) / "output"
        out_root.mkdir()
        
        gen.generate_runs_from_equilibria(
            inpnames=sample_params,
            inputfilename="input.dat",
            source_run_dir=str(source_dir),
            ranges=sample_ranges,
            integer_mask=[True, False, False],
            n_runs=3,
            out_root=str(out_root),
            equilibria_mode='set',
            seed=42
        )
        
        # Check that samples.npz exists
        samples_file = out_root / "samples.npz"
        assert samples_file.exists()
        
        # Load and verify contents
        data = np.load(samples_file)
        assert 'X' in data
        assert 'names' in data
        assert data['X'].shape[0] == 3  # 3 runs
        assert data['X'].shape[1] == 3  # 3 parameters
        assert len(data['names']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
