"""Tests that docs/CUSTOM_DATASET_TUTORIAL.md is practical and reproducible."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
CONFIGS = EXAMPLES / "configs"
DATA_DIR = EXAMPLES / "data"
TUTORIAL_SPEC = CONFIGS / "custom_dataset_tutorial.yaml"
TUTORIAL_META = CONFIGS / "custom_dataset_meta.yaml"
RUN_WORKFLOW = EXAMPLES / "run_workflow.py"
CUSTOM_TUTORIAL = EXAMPLES / "custom_dataset_tutorial.py"


def _make_dataframe(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    inputs = {
        "input_a": rng.normal(size=n_rows),
        "input_b": rng.normal(size=n_rows),
        "input_c": rng.normal(size=n_rows),
    }
    outputs = {
        "output_x": inputs["input_a"] * 0.5 + rng.normal(scale=0.05, size=n_rows),
        "output_y": inputs["input_b"] * -0.25 + rng.normal(scale=0.03, size=n_rows),
    }
    return pd.DataFrame({**inputs, **outputs})


def _write_tutorial_spec(tmp_path: Path, dataset_rel: str, fmt: str, run_tag: str) -> Path:
    payload = yaml.safe_load(TUTORIAL_SPEC.read_text(encoding="utf-8"))
    payload["dataset_path"] = dataset_rel
    payload["dataset_format"] = fmt if fmt != "csv" else "auto"
    payload["metadata_path"] = str(TUTORIAL_META.relative_to(REPO))
    payload["output_dir"] = str(tmp_path)
    payload["run_tag"] = run_tag
    payload["overwrite_existing_run"] = True
    spec_path = tmp_path / f"spec_{fmt}.yaml"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return spec_path


@pytest.fixture
def tutorial_dataset_csv(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "tutorial_sample.csv"
    _make_dataframe().to_csv(path, index=False)
    return path


class TestCustomDatasetTutorial:
    def test_bundled_metadata_matches_columns(self) -> None:
        meta = yaml.safe_load(TUTORIAL_META.read_text(encoding="utf-8"))
        df = _make_dataframe(n_rows=5)
        for col in meta["inputs"] + meta["outputs"]:
            assert col in df.columns

    def test_workflow_api_from_tutorial_spec(self, tutorial_dataset_csv: Path, tmp_path: Path) -> None:
        import surge
        from surge import SurrogateWorkflowSpec, run_surrogate_workflow

        spec_path = _write_tutorial_spec(
            tmp_path,
            str(tutorial_dataset_csv),
            "csv",
            "tutorial_api",
        )
        spec = SurrogateWorkflowSpec.from_dict(
            yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        )
        summary = run_surrogate_workflow(spec, invocation={"spec_path": str(spec_path)})

        root = Path(summary["artifacts"]["root"])
        assert (root / "metrics.json").is_file()
        assert (root / "workflow_summary.json").is_file()
        assert (root / "models" / "rf_tutorial.joblib").is_file()
        assert summary["models"][0]["metrics"]["test"]["r2"] > -1.0

    @pytest.mark.parametrize("fmt", ["csv", "pkl", "h5"])
    def test_run_workflow_cli_supports_documented_formats(
        self, tmp_path: Path, fmt: str
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = _make_dataframe()

        if fmt == "csv":
            dataset_path = data_dir / "sample.csv"
            df.to_csv(dataset_path, index=False)
            dataset_format = "auto"
        elif fmt == "pkl":
            dataset_path = data_dir / "sample.pkl"
            df.to_pickle(dataset_path)
            dataset_format = "pkl"
        else:
            h5py = pytest.importorskip("h5py")
            dataset_path = data_dir / "sample.h5"
            with h5py.File(dataset_path, "w") as handle:
                ds = handle.create_dataset("data", data=df.to_numpy())
                ds.attrs["column_names"] = np.array(
                    [str(c).encode("utf-8") for c in df.columns],
                    dtype="S",
                )
            dataset_format = "h5"

        spec_path = _write_tutorial_spec(
            tmp_path,
            str(dataset_path),
            dataset_format,
            f"tutorial_{fmt}",
        )

        completed = subprocess.run(
            [sys.executable, str(RUN_WORKFLOW), "--spec", str(spec_path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout

        run_dir = tmp_path / "runs" / f"tutorial_{fmt}"
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "models" / "rf_tutorial.joblib").is_file()

    def test_custom_dataset_tutorial_script(self, tmp_path: Path) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(CUSTOM_TUTORIAL),
                "--output-dir",
                str(tmp_path),
                "--run-tag",
                "tutorial_script",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout

        run_dir = tmp_path / "runs" / "tutorial_script"
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "models" / "rf_tutorial.joblib").is_file()

    def test_surrogate_dataset_load_documented_paths(self, tutorial_dataset_csv: Path) -> None:
        from surge import SurrogateDataset

        ds = SurrogateDataset.from_path(
            tutorial_dataset_csv,
            metadata_path=TUTORIAL_META,
        )
        assert ds.input_columns == ["input_a", "input_b", "input_c"]
        assert ds.output_columns == ["output_x", "output_y"]
        assert len(ds.df) == 120
