"""Tests for the artifact-driven leaderboard report (surge.report.leaderboard)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("yaml")

from surge.report.leaderboard import (
    build_report,
    load_metadata,
    load_results,
    primary_metric_key,
    rank_models,
)


def _write_result(root: Path, bench: str, model: str, ts: str, **metrics):
    d = root / bench / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({
        "benchmark_key": bench,
        "model_key": model,
        "metrics": metrics,
        "passed": metrics.get("test_r2", 0) >= 0.5,
        "timestamp": ts,
        "surge_version": "0.1.0",
    }))


@pytest.fixture()
def reports(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark_reports"
    _write_result(root, "tabular.california_housing", "sklearn.random_forest",
                  "2026-01-01T00-00-00Z", test_r2=0.80, test_rmse=0.51, runtime_s=3.0)
    _write_result(root, "tabular.california_housing", "sklearn.random_forest",
                  "2026-01-02T00-00-00Z", test_r2=0.82, test_rmse=0.49, runtime_s=3.2)
    _write_result(root, "tabular.california_housing", "sklearn.mlp",
                  "2026-01-03T00-00-00Z", test_r2=0.40, test_rmse=0.89, runtime_s=60.0)
    return root


def test_metadata_loads_and_covers_known_benchmarks():
    meta = load_metadata()
    assert "tabular.california_housing" in meta
    entry = meta["tabular.california_housing"]
    assert entry["citation"].startswith("Pace & Barry")
    assert entry["threshold"]
    # extracted feature docs survive for the stellarator benchmark
    assert any("constellaration" in k for k in meta)


def test_load_results_aggregates_mean_std(reports: Path):
    res = load_results(reports)
    rf = res["tabular.california_housing"]["sklearn.random_forest"]
    assert rf["n_runs"] == 2
    assert rf["metrics"]["test_r2"]["mean"] == pytest.approx(0.81)
    assert rf["metrics"]["test_r2"]["std"] > 0
    assert rf["passed"] is True
    assert res["tabular.california_housing"]["sklearn.mlp"]["passed"] is False


def test_rank_models_orders_by_primary_metric(reports: Path):
    res = load_results(reports)["tabular.california_housing"]
    pm = primary_metric_key(res)
    assert pm == "test_r2"
    assert rank_models(res, pm)[0] == "sklearn.random_forest"


def test_build_report_is_self_contained_html(reports: Path, tmp_path: Path):
    out = build_report(reports, tmp_path / "lb.html")
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "sklearn.random_forest" in text
    assert "California Housing" in text          # metadata joined in
    assert "±" in text                            # std shown
    assert "http" not in text.split("</style>")[0]  # no external CSS/fonts
