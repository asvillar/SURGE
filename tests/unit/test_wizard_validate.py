"""Tests for surge init (wizard), surge validate, and the spec JSON Schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from surge.workflow.schema import (
    spec_json_schema,
    validate_file,
    validate_payload,
    write_schema,
)


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.standard_normal(50),
        "b": rng.standard_normal(50),
        "target": rng.standard_normal(50),
    })
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)
    return p


# ── validate ────────────────────────────────────────────────────────────


def test_validate_accepts_minimal_spec():
    assert validate_payload(
        {"dataset_path": "d.csv",
         "models": [{"key": "sklearn.ridge"}]}) == []


def test_validate_flags_unknown_key_with_suggestion():
    errors = validate_payload(
        {"dataset_pth": "d.csv", "models": [{"key": "sklearn.ridge"}]})
    assert any("dataset_pth" in e and "dataset_path" in e for e in errors)
    assert any("missing required key 'dataset_path'" in e for e in errors)


def test_validate_flags_bad_model_key_and_hpo_typo():
    errors = validate_payload(
        {"dataset_path": "d.csv",
         "models": [{"key": "sklearn.rigde", "hpo": {"n_trails": 5}}]})
    assert any("sklearn.ridge" in e for e in errors)      # did-you-mean
    assert any("n_trials" in e for e in errors)


def test_validate_flags_fraction_range_and_types():
    errors = validate_payload(
        {"dataset_path": "d.csv", "test_fraction": 1.4,
         "seed": "forty-two", "models": [{"key": "sklearn.ridge"}]})
    assert any("test_fraction" in e for e in errors)
    assert any("seed" in e for e in errors)


def test_validate_requires_models():
    errors = validate_payload({"dataset_path": "d.csv"})
    assert any("at least one model" in e for e in errors)


def test_validate_file_yaml_errors(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [unclosed")
    assert any("YAML" in e for e in validate_file(bad))
    assert any("not found" in e for e in validate_file(tmp_path / "nope.yaml"))


def test_cli_validate_exit_codes(tmp_path: Path, capsys):
    from surge.cli import main

    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(
        {"dataset_path": "d.csv", "models": [{"key": "sklearn.ridge"}]}))
    assert main(["validate", str(good)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"dataset_path": "d.csv", "models": []}))
    assert main(["validate", str(bad)]) == 2


# ── schema ──────────────────────────────────────────────────────────────


def test_schema_structure():
    s = spec_json_schema()
    assert s["properties"]["dataset_path"] == {"type": "string"}
    assert s["required"] == ["dataset_path"]
    assert set(s["$defs"]) == {"ModelConfig", "HPOConfig", "ResourceSpec"}
    assert s["$defs"]["ModelConfig"]["required"] == ["key"]


def test_shipped_schema_in_sync(tmp_path: Path):
    """surge/workflow/spec.schema.json must match the generator output."""
    shipped = Path("surge/workflow/spec.schema.json")
    assert shipped.exists(), "run: python -m surge.workflow.schema"
    regenerated = write_schema(tmp_path / "s.json").read_text()
    assert shipped.read_text() == regenerated


# ── wizard (non-interactive path) ───────────────────────────────────────


def test_init_writes_valid_spec(csv_file: Path, tmp_path: Path):
    from surge.wizard import main as wizard_main

    out = tmp_path / "spec.yaml"
    rc = wizard_main(["--data", str(csv_file), "--goal", "accuracy",
                      "--budget", "smoke", "--out", str(out), "--yes"])
    assert rc == 0
    assert validate_file(out) == []
    payload = yaml.safe_load(out.read_text())
    assert Path(payload["dataset_path"]).is_absolute()
    assert payload["models"], "wizard must propose at least one model"
    assert payload["sample_rows"] == 2000          # smoke budget subsamples
    text = out.read_text()
    assert "#" in text                              # commented output


def test_init_goal_changes_slate(csv_file: Path, tmp_path: Path):
    from surge.wizard import main as wizard_main

    out = tmp_path / "speed.yaml"
    assert wizard_main(["--data", str(csv_file), "--goal", "speed",
                        "--budget", "standard", "--out", str(out),
                        "--yes"]) == 0
    keys = [m["key"] for m in yaml.safe_load(out.read_text())["models"]]
    assert "sklearn.ridge" in keys


def test_init_missing_data_errors(tmp_path: Path):
    from surge.wizard import main as wizard_main

    assert wizard_main(["--data", str(tmp_path / "ghost.csv"),
                        "--yes"]) == 2
    assert wizard_main(["--yes"]) == 2              # --data required with --yes


def test_init_bad_target_errors(csv_file: Path, tmp_path: Path):
    from surge.wizard import main as wizard_main

    assert wizard_main(["--data", str(csv_file), "--target", "nope",
                        "--out", str(tmp_path / "s.yaml"), "--yes"]) == 2
