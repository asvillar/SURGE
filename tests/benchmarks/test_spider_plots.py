from __future__ import annotations

from surge.benchmarks.base import BenchmarkResult


def test_model_spider_chart_saves_png_and_pdf(tmp_path):
    from surge.viz.benchmark import plot_model_spider_chart

    results = [
        BenchmarkResult(
            benchmark_key="tabular.iris",
            tier="1",
            task_type="classification",
            model_key="sklearn.random_forest_classifier",
            metrics={
                "test_accuracy": 0.97,
                "test_f1_macro": 0.96,
                "runtime_s": 0.25,
                "peak_memory_mb": 12.0,
            },
            passed=True,
        ),
        BenchmarkResult(
            benchmark_key="tabular.iris",
            tier="1",
            task_type="classification",
            model_key="sklearn.logistic_regression",
            metrics={
                "test_accuracy": 0.93,
                "test_f1_macro": 0.92,
                "runtime_s": 0.05,
                "peak_memory_mb": 4.0,
            },
            passed=True,
        ),
    ]
    save_path = tmp_path / "spider.png"
    fig = plot_model_spider_chart(results, save_path=save_path)
    assert fig is not None
    assert save_path.exists()
    assert save_path.with_suffix(".pdf").exists()


def test_model_spider_chart_requires_three_metrics():
    import pytest
    from surge.viz.benchmark import plot_model_spider_chart

    result = BenchmarkResult(
        benchmark_key="tabular.diabetes",
        tier="1",
        task_type="regression",
        model_key="sklearn.ridge",
        metrics={"test_r2": 0.4, "runtime_s": 0.01},
        passed=True,
    )
    with pytest.raises(ValueError, match="at least three metrics"):
        plot_model_spider_chart([result])
