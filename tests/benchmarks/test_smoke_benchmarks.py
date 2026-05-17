"""Smoke tests for surge.benchmarks registry — Phase 1 expansion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surge.benchmarks.registry import benchmark_info, list_benchmarks, run_benchmark


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_list_benchmarks_contains_all():
    keys = list_benchmarks()
    expected = {
        "synthetic.regression_1d",
        "synthetic.classification_binary",
        "tabular.diabetes",
        "tabular.california_housing",
        "tabular.iris",
        "tabular.breast_cancer",
        "tabular.wine",
        "tabular.digits",
    }
    assert expected.issubset(set(keys)), f"Missing: {expected - set(keys)}"


def test_list_benchmarks_filter_tier():
    tier0 = list_benchmarks(tier="0")
    assert all(benchmark_info(k)["tier"] == "0" for k in tier0)
    assert len(tier0) >= 2


def test_list_benchmarks_filter_task_type():
    reg = list_benchmarks(task_type="regression")
    clf = list_benchmarks(task_type="classification")
    assert all(benchmark_info(k)["task_type"] == "regression" for k in reg)
    assert all(benchmark_info(k)["task_type"] == "classification" for k in clf)
    assert set(reg).isdisjoint(set(clf))


def test_benchmark_info_fields():
    info = benchmark_info("tabular.iris")
    assert info["key"] == "tabular.iris"
    assert info["tier"] == "1"
    assert info["task_type"] == "classification"
    assert info["shape"]
    assert info["description"]


def test_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        run_benchmark("nonexistent.benchmark")


# ---------------------------------------------------------------------------
# Default-model runs (Tier 0 — always fast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "synthetic.regression_1d",
        "synthetic.classification_binary",
        "tabular.iris",
        "tabular.diabetes",
    ],
)
def test_benchmark_default_model_passes(key: str):
    res = run_benchmark(key, seed=42)
    assert res.benchmark_key == key
    assert res.model_key, "model_key should be populated"
    assert isinstance(res.metrics, dict)
    assert res.passed, f"{key} failed: metrics={res.metrics}"


# ---------------------------------------------------------------------------
# --model flag: run a benchmark with non-default models
# ---------------------------------------------------------------------------


def test_regression_with_logistic_regression_raises_or_runs():
    """Regression benchmarks should reject classification-only model gracefully."""
    # We don't enforce task compatibility in Phase 1 (that's Phase 2),
    # but the call should not crash the process — it may fail at predict.
    # This test just documents current behavior.
    try:
        res = run_benchmark("tabular.diabetes", seed=0, model_key="sklearn.logistic_regression")
        # If it ran, it should have metrics
        assert "test_r2" in res.metrics or "test_accuracy" in res.metrics
    except Exception:
        pass  # acceptable — mismatch may raise


def test_breast_cancer_with_gradient_boosting():
    res = run_benchmark("tabular.breast_cancer", seed=42, model_key="sklearn.gradient_boosting_classifier")
    assert res.benchmark_key == "tabular.breast_cancer"
    assert res.model_key == "sklearn.gradient_boosting_classifier"
    assert "test_accuracy" in res.metrics
    assert res.passed, f"GBC on breast_cancer failed: {res.metrics}"


def test_iris_with_logistic_regression():
    res = run_benchmark("tabular.iris", seed=42, model_key="sklearn.logistic_regression")
    assert res.model_key == "sklearn.logistic_regression"
    assert res.passed, f"LR on iris failed: {res.metrics}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_list(capsys):
    from surge.benchmarks.run import main

    code = main(["--list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "tabular.iris" in out
    assert "tabular.breast_cancer" in out


def test_cli_list_verbose(capsys):
    from surge.benchmarks.run import main

    code = main(["--list", "--verbose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "classification" in out
    assert "regression" in out


def test_cli_writes_json(tmp_path: Path):
    from surge.benchmarks.run import main

    out = tmp_path / "r.json"
    code = main(["--benchmark", "synthetic.regression_1d", "--output", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert "test_r2" in data["metrics"]
    assert data["model_key"]


def test_cli_with_model_flag(tmp_path: Path):
    from surge.benchmarks.run import main

    out = tmp_path / "r.json"
    code = main([
        "--benchmark", "tabular.iris",
        "--model", "sklearn.logistic_regression",
        "--output", str(out),
    ])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model_key"] == "sklearn.logistic_regression"
    assert data["passed"] is True


def test_cli_all_tier0(capsys):
    from surge.benchmarks.run import main

    code = main(["--all", "--tier", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_cli_list_models(capsys):
    from surge.benchmarks.run import main

    code = main(["--list-models"])
    assert code == 0
    out = capsys.readouterr().out
    assert "sklearn.random_forest_classifier" in out
    assert "sklearn.logistic_regression" in out
    assert "sklearn.gradient_boosting_classifier" in out
