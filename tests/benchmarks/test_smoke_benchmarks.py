"""Smoke tests for surge.benchmarks registry (Phase 4–5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surge.benchmarks.registry import list_benchmarks, run_benchmark


def test_list_benchmarks_nonempty():
    keys = list_benchmarks()
    assert "synthetic.regression_1d" in keys
    assert "tabular.iris" in keys


@pytest.mark.parametrize(
    "key",
    [
        "synthetic.regression_1d",
        "synthetic.classification_binary",
        "tabular.iris",
        "tabular.diabetes",
    ],
)
def test_benchmark_passes(key: str):
    res = run_benchmark(key, seed=42)
    assert res.benchmark_key == key
    assert res.passed, (key, res.metrics, res.message)


def test_cli_writes_json(tmp_path: Path):
    from surge.benchmarks.run import main

    out = tmp_path / "r.json"
    code = main(["--benchmark", "synthetic.regression_1d", "--output", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert "test_r2" in data["metrics"]
