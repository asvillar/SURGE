"""Tests for the top-level ``surge`` CLI dispatcher."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

yaml = pytest.importorskip("yaml")

from surge.cli import main


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "surge run <spec.yaml>" in out


def test_version(capsys):
    assert main(["version"]) == 0
    assert "surge-ml" in capsys.readouterr().out


def test_models_lists_registry(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "sklearn.random_forest" in out


def test_models_verbose_shows_skip_reasons(capsys):
    assert main(["models", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Adapter registration:")
    assert "registered" in out


def test_run_workflow_from_yaml(tmp_path: Path, capfd):
    # capfd (fd-level) because the workflow engine re-binds sys.stdout
    # internally for its run.log tee.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = X @ [1.5, -2.0, 0.5] + 0.05 * rng.standard_normal(80)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y_target"] = y  # y_ prefix -> inferred as output column
    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)

    spec = {
        "dataset_path": str(csv),
        "models": [{"key": "sklearn.ridge"}],
        "run_tag": "cli_test",
        "output_dir": str(tmp_path / "runs"),
        "test_fraction": 0.25,
        "val_fraction": 0.25,
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    assert main(["run", str(spec_path)]) == 0
    capfd.readouterr()  # drain engine output (exact routing is engine-owned)

    # the engine nests runs/<tag>/ under output_dir
    run_dir = tmp_path / "runs" / "runs" / "cli_test"
    assert (run_dir / "spec.yaml").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["sklearn.ridge"]["test"]["r2"] > 0.9  # near-linear toy data
