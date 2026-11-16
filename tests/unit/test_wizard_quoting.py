"""surge init must emit valid YAML for awkward column names and paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from surge.wizard import main as wizard_main
from surge.workflow.schema import validate_file

NASTY = ["a: b", "x #1", "- lead", "123", "it's"]
TARGET = "out [W]"


def test_init_quotes_awkward_names(tmp_path: Path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((60, len(NASTY))), columns=NASTY)
    df[TARGET] = df.sum(axis=1)
    data = tmp_path / "my data.csv"
    df.to_csv(data, index=False)
    out = tmp_path / "spec.yaml"
    assert wizard_main(["--data", str(data), "--target", TARGET, "--yes",
                        "--out", str(out)]) == 0
    assert validate_file(out) == []
    payload = yaml.safe_load(out.read_text())
    assert payload["metadata_overrides"]["inputs"] == NASTY
    assert payload["metadata_overrides"]["outputs"] == [TARGET]
    assert Path(payload["dataset_path"]) == data.resolve()
