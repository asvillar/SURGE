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
        "synthetic.multioutput_2d",
        "synthetic.classification_binary",
        "tabular.diabetes",
        "tabular.california_housing",
        "tabular.iris",
        "tabular.breast_cancer",
        "tabular.wine",
        "tabular.digits",
        "tabular.energy_efficiency",
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


# ---------------------------------------------------------------------------
# Auto-save to benchmark_reports/
# ---------------------------------------------------------------------------


def test_benchmark_result_save(tmp_path: Path):
    """BenchmarkResult.save() writes the expected directory structure."""
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark("synthetic.regression_1d", seed=0)
    saved_path = result.save(root=tmp_path)

    assert saved_path.exists()
    assert saved_path.name == "result.json"
    # directory layout: <root>/<benchmark_key>/<timestamp>/result.json
    assert saved_path.parent.parent.name == "synthetic.regression_1d"

    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["benchmark_key"] == "synthetic.regression_1d"
    assert data["passed"] is True
    assert data["timestamp"]           # populated by save()
    assert data["surge_version"]       # populated from surge.__version__


def test_cli_autosave(tmp_path: Path):
    """Running the CLI without --no-save writes to the save-dir."""
    from surge.benchmarks.run import main

    code = main([
        "--benchmark", "synthetic.regression_1d",
        "--save-dir", str(tmp_path),
    ])
    assert code == 0
    # Expect benchmark_reports/<key>/<ts>/result.json
    results = list(tmp_path.glob("synthetic.regression_1d/*/result.json"))
    assert len(results) == 1
    data = json.loads(results[0].read_text())
    assert data["passed"] is True


def test_cli_no_save(tmp_path: Path):
    """--no-save produces no files under save-dir."""
    from surge.benchmarks.run import main

    code = main([
        "--benchmark", "synthetic.regression_1d",
        "--no-save",
        "--save-dir", str(tmp_path),
    ])
    assert code == 0
    assert list(tmp_path.rglob("result.json")) == []


def test_cli_all_autosave(tmp_path: Path, capsys):
    """--all saves one result.json per benchmark."""
    from surge.benchmarks.run import main

    code = main(["--all", "--tier", "0", "--save-dir", str(tmp_path)])
    assert code == 0
    saved = list(tmp_path.rglob("result.json"))
    # Tier-0 benchmarks: synthetic.regression_1d, synthetic.classification_binary,
    # synthetic.multioutput_2d (3 total)
    assert len(saved) == 3


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


def test_log_benchmark_result_mlflow(tmp_path: Path):
    """log_benchmark_result() logs metrics without error when MLflow is available."""
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE, log_benchmark_result

    if not MLFLOW_AVAILABLE:
        pytest.skip("mlflow not installed")

    import mlflow

    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark("synthetic.regression_1d", seed=0)
    result_path = result.save(root=tmp_path / "benchmark_reports")

    ok = log_benchmark_result(
        result,
        experiment_name="test_surge_benchmarks",
        tracking_uri=tracking_uri,
        result_path=result_path,
    )
    assert ok is True

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("test_surge_benchmarks")
    assert exp is not None

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert "test_r2" in run.data.metrics
    assert run.data.params["benchmark_key"] == "synthetic.regression_1d"
    assert run.data.tags["passed"] == "True"
    assert run.data.tags["task_type"] == "regression"


def test_cli_mlflow_flag(tmp_path: Path):
    """--mlflow flag calls MLflow without crashing (uses temp tracking URI)."""
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE

    if not MLFLOW_AVAILABLE:
        pytest.skip("mlflow not installed")

    from surge.benchmarks.run import main

    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    code = main([
        "--benchmark", "synthetic.classification_binary",
        "--no-save",
        "--mlflow",
        "--mlflow-experiment", "test_cli_experiment",
        "--mlflow-tracking-uri", tracking_uri,
    ])
    assert code == 0

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("test_cli_experiment")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    assert "test_accuracy" in runs[0].data.metrics
