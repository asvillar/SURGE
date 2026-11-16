"""request_uncertainty: true must complete and persist a usable val_uq JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from surge import cli, wizard


def _csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    df["y"] = df["a"] * 0.5 - df["b"] + rng.normal(scale=0.05, size=200)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _spec(tmp_path: Path, csv: Path, model: dict) -> Path:
    spec = {
        "dataset_path": str(csv),
        "metadata_overrides": {"inputs": ["a", "b", "c"], "outputs": ["y"]},
        "models": [model],
        "output_dir": str(tmp_path / "out"),
        "run_tag": "uq",
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _check(root: Path) -> None:
    assert list(root.rglob("metrics.json")), "metrics.json missing"
    uq_files = list(root.rglob("*_val_uq.json"))
    assert uq_files, "val_uq json missing"
    payload = json.loads(uq_files[0].read_text())
    assert isinstance(payload, dict)
    assert len(payload["mean"]) == len(payload["std"]) > 0


def test_gpr_uncertainty(tmp_path: Path) -> None:
    spec = _spec(tmp_path, _csv(tmp_path),
                 {"key": "sklearn.gpr", "name": "gpr", "request_uncertainty": True})
    assert cli.main(["run", str(spec)]) in (0, None)
    _check(tmp_path / "out")


def test_mlp_ensemble_uncertainty(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    spec = _spec(tmp_path, _csv(tmp_path),
                 {"key": "pytorch.mlp_ensemble", "name": "ens", "request_uncertainty": True,
                  "params": {"n_epochs": 2}})
    assert cli.main(["run", str(spec)]) in (0, None)
    _check(tmp_path / "out")


def test_wizard_uncertainty_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # wizard specs write runs/ under the cwd
    csv = _csv(tmp_path)
    spec = tmp_path / "wiz.yaml"
    assert wizard.main(["--data", str(csv), "--target", "y", "--goal", "uncertainty",
                        "--budget", "smoke", "--yes", "--out", str(spec)]) == 0
    assert cli.main(["run", str(spec)]) in (0, None)
    metrics_files = list(tmp_path.rglob("metrics.json"))
    assert metrics_files
    assert list(tmp_path.rglob("*_val_uq.json"))
    # the uncertainty model must be a usable model, not only a finished run
    import json
    metrics = json.loads(metrics_files[0].read_text())
    uq_models = [name for name in metrics if name != "baseline_rf"]
    assert uq_models and all(metrics[name]["test"]["r2"] > 0.5 for name in uq_models)
