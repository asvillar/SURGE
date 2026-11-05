"""Tests for M3DC1 workflow execution in both modes: from file vs from batch dir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

# The M3D-C1 HDF5 readers are facility-specific and not part of the public
# tree; these tests only run where they are available (SURGE_M3DC1_SCRIPTS or
# an ignored scripts/m3dc1/ checkout).
def _m3dc1_readers_available() -> bool:
    try:
        from surge.datasets.m3dc1 import _get_m3dc1_loaders
        _get_m3dc1_loaders()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _m3dc1_readers_available(),
                                reason="facility-specific M3D-C1 readers not available")



# ---------------------------------------------------------------------------
# Fixtures: fake M3DC1 batch dir with minimal complex_v2 HDF5
# ---------------------------------------------------------------------------
COMPLEX_V2_FILENAME = "sdata_pertfields_grid_complex_v2.h5"
SDATA_COMPLEX_V2_FILENAME = "sdata_complex_v2.h5"


def _write_fake_complex_v2_h5(
    path: Path,
    *,
    n_m: int = 4,
    n_psi: int = 5,
    run_name: str = "sparc_1300",
) -> None:
    """Write a minimal sdata_pertfields_grid_complex_v2.h5 for testing."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py required for M3DC1 batch tests")

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    with h5py.File(path, "w") as f:
        rg = f.create_group(f"runs/{run_name}")
        # miller
        miller = rg.create_group("miller")
        miller.create_dataset("R0", data=1.8)
        miller.create_dataset("a", data=0.55)
        miller.create_dataset("kappa", data=2.0)
        miller.create_dataset("delta", data=0.35)
        # parset
        parset = rg.create_group("parset")
        parset.create_dataset("names", data=np.array([b"ntor", b"pscale", b"batemanscale"]))
        parset.create_dataset("values", data=np.array([3.0, 1.0, 1.0]))
        # flux_average
        fa = rg.create_group("flux_average")
        qg = fa.create_group("q")
        n_psin = 20
        qg.create_dataset("psin", data=np.linspace(0, 1, n_psin))
        qg.create_dataset("profile", data=np.linspace(1.2, 4.0, n_psin))
        pg = fa.create_group("p")
        pg.create_dataset("profile", data=np.linspace(1.0, 0.1, n_psin))
        # spectrum/p
        sp = rg.create_group("spectrum/p")
        m_modes = np.array([-2, -1, 0, 1][:n_m])
        spec = rng.uniform(0.01, 1.0, (n_m, n_psi)).astype(np.complex64)
        sp.create_dataset("m_modes", data=m_modes[:n_m])
        sp.create_dataset("spec", data=spec)


@pytest.fixture
def fake_m3dc1_batch_dir(tmp_path: Path) -> Path:
    """Create a minimal batch dir with one complex_v2 file."""
    run_dir = tmp_path / "run1" / "sparc_1300"
    h5_path = run_dir / COMPLEX_V2_FILENAME
    _write_fake_complex_v2_h5(h5_path, n_m=4, n_psi=5)
    return tmp_path


@pytest.fixture
def fake_m3dc1_batch_dir_sdata_complex(tmp_path: Path) -> Path:
    """Batch dir with sdata_complex_v2.h5 (same layout as pertfields complex_v2)."""
    run_dir = tmp_path / "run1" / "sparc_1300"
    h5_path = run_dir / SDATA_COMPLEX_V2_FILENAME
    _write_fake_complex_v2_h5(h5_path, n_m=4, n_psi=5)
    return tmp_path


@pytest.fixture
def fake_m3dc1_batch_dir_two_cases(tmp_path: Path) -> Path:
    """Create batch dir with 4 cases for train/val/test splits."""
    for i in range(4):
        run_name = f"sparc_130{i}"
        run_dir = tmp_path / f"run{i + 1}" / run_name
        h5_path = run_dir / COMPLEX_V2_FILENAME
        _write_fake_complex_v2_h5(h5_path, n_m=4, n_psi=5, run_name=run_name)
    return tmp_path


def _make_spec_from_file(data_path: Path, tmp_path: Path) -> SurrogateWorkflowSpec:
    """Build spec for mode A: dataset from pre-built file."""
    return SurrogateWorkflowSpec.from_dict({
        "dataset_path": str(data_path),
        "metadata_path": None,
        "test_fraction": 0.2,
        "val_fraction": 0.2,
        "standardize_inputs": True,
        "standardize_outputs": True,
        "predictions_scope": ["train", "val", "test"],
        "predictions_format": "csv",
        "output_dir": str(tmp_path),
        "run_tag": "test_from_file",
        "models": [
            {
                "key": "sklearn.random_forest",
                "name": "rf_test",
                "params": {"n_estimators": 10, "max_depth": 4},
                "hpo": {"enabled": False},
            }
        ],
    })


def _make_spec_from_batch(batch_dir: Path, tmp_path: Path) -> SurrogateWorkflowSpec:
    """Build spec for mode B: dataset from batch dir (full spectrum)."""
    return SurrogateWorkflowSpec.from_dict({
        "dataset_path": str(batch_dir),
        "dataset_source": "m3dc1_batch",
        "batch_dir_target_shape": [4, 5],
        "test_fraction": 0.2,
        "val_fraction": 0.2,
        "standardize_inputs": True,
        "standardize_outputs": True,
        "predictions_scope": ["train", "val", "test"],
        "predictions_format": "csv",
        "output_dir": str(tmp_path),
        "run_tag": "test_from_batch",
        "models": [
            {
                "key": "sklearn.random_forest",
                "name": "rf_test",
                "params": {"n_estimators": 10, "max_depth": 4},
                "hpo": {"enabled": False},
            }
        ],
    })


def _make_spec_from_batch_per_mode(
    batch_dir: Path,
    tmp_path: Path,
    *,
    batch_dir_filename: Optional[str] = None,
) -> SurrogateWorkflowSpec:
    """Build spec for mode C: dataset from batch dir (per-mode format)."""
    payload: dict = {
        "dataset_path": str(batch_dir),
        "dataset_source": "m3dc1_batch_per_mode",
        "test_fraction": 0.2,
        "val_fraction": 0.2,
        "standardize_inputs": True,
        "standardize_outputs": True,
        "predictions_scope": ["train", "val", "test"],
        "predictions_format": "csv",
        "output_dir": str(tmp_path),
        "run_tag": "test_from_batch_per_mode",
        "models": [
            {
                "key": "sklearn.random_forest",
                "name": "rf_test",
                "params": {"n_estimators": 10, "max_depth": 4},
                "hpo": {"enabled": False},
            }
        ],
    }
    if batch_dir_filename is not None:
        payload["batch_dir_filename"] = batch_dir_filename
    return SurrogateWorkflowSpec.from_dict(payload)


# ---------------------------------------------------------------------------
# Mode A: from pre-built file (pkl)
# ---------------------------------------------------------------------------
class TestWorkflowModeFromFile:
    """Workflow runs from pre-built dataset file."""

    def test_run_from_pkl_creates_artifacts(self, tmp_path: Path) -> None:
        """Mode A: workflow with dataset_path → .pkl runs and produces artifacts."""
        rng = np.random.default_rng(42)
        n_rows = 80
        df = pd.DataFrame({
            "eq_R0": np.full(n_rows, 1.8),
            "eq_a": np.full(n_rows, 0.55),
            "eq_kappa": np.full(n_rows, 2.0),
            "eq_delta": np.full(n_rows, 0.35),
            "q0": np.full(n_rows, 1.2),
            "q95": np.full(n_rows, 4.0),
            "qmin": np.full(n_rows, 1.1),
            "p0": np.full(n_rows, 1.0),
            "n": np.full(n_rows, 3),
            "m": np.tile([-2, -1, 0, 1], n_rows // 4)[:n_rows],
            "input_pscale": np.full(n_rows, 1.0),
            "input_batemanscale": np.full(n_rows, 1.0),
            **{f"output_p_{i}": rng.uniform(0.01, 1.0, n_rows) for i in range(5)},
        })
        data_path = tmp_path / "delta_p_per_mode.pkl"
        df.to_pickle(data_path)

        metadata_path = tmp_path / "metadata.yaml"
        metadata_path.write_text(
            "inputs:\n"
            "  - eq_R0\n  - eq_a\n  - eq_kappa\n  - eq_delta\n"
            "  - q0\n  - q95\n  - qmin\n  - p0\n  - n\n  - m\n"
            "  - input_pscale\n  - input_batemanscale\n"
            "outputs:\n"
            "  - output_p_0\n  - output_p_1\n  - output_p_2\n  - output_p_3\n  - output_p_4\n",
            encoding="utf-8",
        )

        spec_dict = {
            "dataset_path": str(data_path),
            "metadata_path": str(metadata_path),
            "test_fraction": 0.2,
            "val_fraction": 0.2,
            "standardize_inputs": True,
            "standardize_outputs": True,
            "predictions_scope": ["train", "val", "test"],
            "predictions_format": "csv",
            "output_dir": str(tmp_path),
            "run_tag": "test_from_file",
            "models": [
                {
                    "key": "sklearn.random_forest",
                    "name": "rf_test",
                    "params": {"n_estimators": 10, "max_depth": 4},
                    "hpo": {"enabled": False},
                }
            ],
        }
        spec = SurrogateWorkflowSpec.from_dict(spec_dict)

        summary = run_surrogate_workflow(spec)

        root = Path(summary["artifacts"]["root"])
        assert (root / "metrics.json").exists()
        assert (root / "workflow_summary.json").exists()
        assert summary["splits"]["train"] > 0
        assert len(summary["models"]) == 1
        assert summary["models"][0]["name"] == "rf_test"


# ---------------------------------------------------------------------------
# Mode B: from batch dir (full spectrum)
# ---------------------------------------------------------------------------
class TestWorkflowModeFromBatchDir:
    """Workflow runs from M3DC1 batch directory (full spectrum)."""

    def test_run_from_batch_dir_creates_artifacts(
        self, fake_m3dc1_batch_dir_two_cases: Path, tmp_path: Path
    ) -> None:
        """Mode B: workflow with dataset_source=m3dc1_batch runs and produces artifacts."""
        spec = _make_spec_from_batch(fake_m3dc1_batch_dir_two_cases, tmp_path)
        summary = run_surrogate_workflow(spec)

        root = Path(summary["artifacts"]["root"])
        assert (root / "metrics.json").exists()
        assert (root / "workflow_summary.json").exists()
        assert summary["splits"]["train"] > 0
        assert len(summary["models"]) == 1


# ---------------------------------------------------------------------------
# Mode C: from batch dir (per-mode)
# ---------------------------------------------------------------------------
class TestWorkflowModeFromBatchDirPerMode:
    """Workflow runs from M3DC1 batch directory in per-mode format."""

    def test_run_from_batch_dir_per_mode_creates_artifacts(
        self, fake_m3dc1_batch_dir: Path, tmp_path: Path
    ) -> None:
        """Mode C: workflow with dataset_source=m3dc1_batch_per_mode runs and produces artifacts."""
        spec = _make_spec_from_batch_per_mode(fake_m3dc1_batch_dir, tmp_path)
        summary = run_surrogate_workflow(spec)

        root = Path(summary["artifacts"]["root"])
        assert (root / "metrics.json").exists()
        assert (root / "workflow_summary.json").exists()
        assert summary["splits"]["train"] > 0
        assert len(summary["models"]) == 1
        # Per-mode: one row per (case, m) → 1 case × 4 m-modes = 4 rows
        assert summary["dataset"]["n_rows"] == 4

    def test_run_from_batch_dir_per_mode_multiple_cases(
        self, fake_m3dc1_batch_dir_two_cases: Path, tmp_path: Path
    ) -> None:
        """Per-mode with 4 cases yields 16 rows (4 cases × 4 m-modes)."""
        spec = _make_spec_from_batch_per_mode(fake_m3dc1_batch_dir_two_cases, tmp_path)
        summary = run_surrogate_workflow(spec)
        assert summary["dataset"]["n_rows"] == 16
        assert summary["splits"]["train"] > 0

    def test_run_from_batch_dir_per_mode_sdata_complex_v2_filename(
        self, fake_m3dc1_batch_dir_sdata_complex: Path, tmp_path: Path
    ) -> None:
        """CFS-style leaf name sdata_complex_v2.h5 via batch_dir_filename."""
        spec = _make_spec_from_batch_per_mode(
            fake_m3dc1_batch_dir_sdata_complex,
            tmp_path,
            batch_dir_filename=SDATA_COMPLEX_V2_FILENAME,
        )
        summary = run_surrogate_workflow(spec)
        assert summary["dataset"]["n_rows"] == 4


# ---------------------------------------------------------------------------
# Cross-check: both modes produce equivalent structure
# ---------------------------------------------------------------------------
class TestWorkflowModesProduceEquivalentStructure:
    """Both modes produce the same artifact structure."""

    def test_both_modes_produce_metrics_and_predictions(
        self, fake_m3dc1_batch_dir: Path, tmp_path: Path
    ) -> None:
        """From-file and from-batch-per-mode both produce metrics and predictions."""
        # Build pkl from batch dir first (simulate pre-build step)
        from surge.datasets import M3DC1Dataset

        dataset = M3DC1Dataset.from_batch_dir_per_mode(
            fake_m3dc1_batch_dir, verbose=False
        )
        # Save only numeric input/output columns (exclude run_id, eq_id)
        cols = dataset.input_columns + dataset.output_columns
        dataset.df[cols].to_pickle(tmp_path / "built.pkl")
        pkl_path = tmp_path / "built.pkl"

        # Mode A: from file (no metadata needed; columns are numeric only)
        spec_a = SurrogateWorkflowSpec.from_dict({
            "dataset_path": str(pkl_path),
            "test_fraction": 0.2,
            "val_fraction": 0.2,
            "standardize_inputs": True,
            "standardize_outputs": True,
            "predictions_format": "csv",
            "output_dir": str(tmp_path),
            "run_tag": "mode_a",
            "models": [
                {"key": "sklearn.random_forest", "name": "rf", "hpo": {"enabled": False}}
            ],
        })
        summary_a = run_surrogate_workflow(spec_a)

        # Mode C: from batch dir
        spec_c = _make_spec_from_batch_per_mode(fake_m3dc1_batch_dir, tmp_path)
        spec_c.run_tag = "mode_c"
        summary_c = run_surrogate_workflow(spec_c)

        for summary in (summary_a, summary_c):
            root = Path(summary["artifacts"]["root"])
            assert (root / "metrics.json").exists()
            assert (root / "workflow_summary.json").exists()
            pred_dir = root / "predictions"
            assert pred_dir.exists()
            assert any(pred_dir.glob("*_train.csv")), f"No train predictions in {pred_dir}"
