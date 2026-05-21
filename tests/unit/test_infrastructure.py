"""Infrastructure unit tests — non-model concerns.

Covers:
- ProgressList (append, JSONL log, close)
- plot_training (load from list/file, save figure, compare)
- BaseModelAdapter.training_history for sklearn models
- _uq_metrics from surge.benchmarks.tasks
- HPO cache (save/load/overwrite)
- Benchmark registration (plasma, CTR-23, constellaration_paper)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest


# ── ProgressList ──────────────────────────────────────────────────────────────


def test_progress_list_append_and_len():
    from surge.model.backends._progress import ProgressList

    pl = ProgressList(total_epochs=5, verbose=False)
    for i in range(5):
        pl.append({"epoch": i + 1, "train_loss": 1.0 - 0.1 * i})
    pl.close()

    assert len(pl) == 5
    assert pl[0]["epoch"] == 1
    assert pl[4]["train_loss"] == pytest.approx(0.6)


def test_progress_list_writes_jsonl(tmp_path):
    from surge.model.backends._progress import ProgressList

    log = tmp_path / "train.jsonl"
    pl = ProgressList(total_epochs=3, verbose=False, log_file=str(log))
    for i in range(3):
        pl.append({"epoch": i + 1, "train_loss": 0.1 * (i + 1)})
    pl.close()

    assert log.exists()
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    epoch_records = [r for r in records if not r.get("__run_start__")]
    assert len(epoch_records) == 3
    assert epoch_records[-1]["epoch"] == 3


def test_progress_list_close_idempotent():
    from surge.model.backends._progress import ProgressList

    pl = ProgressList(total_epochs=1, verbose=False)
    pl.append({"epoch": 1, "train_loss": 0.5})
    pl.close()
    pl.close()  # second close must not raise


def test_progress_list_verbose_no_crash():
    from surge.model.backends._progress import ProgressList

    pl = ProgressList(total_epochs=2, verbose=True, desc="TestModel")
    pl.append({"epoch": 1, "train_loss": 0.5})
    pl.append({"epoch": 2, "train_loss": 0.4})
    pl.close()
    assert len(pl) == 2


# ── plot_training ─────────────────────────────────────────────────────────────


def _make_history(n=5):
    return [
        {"epoch": i + 1, "train_loss": 1.0 / (i + 1), "val_loss": 1.2 / (i + 1)}
        for i in range(n)
    ]


def test_load_training_history_from_list():
    from surge.model.plot_training import load_training_history

    hist = _make_history(4)
    loaded = load_training_history(hist)
    assert loaded == hist


def test_load_training_history_from_jsonl(tmp_path):
    from surge.model.backends._progress import ProgressList
    from surge.model.plot_training import load_training_history

    log = tmp_path / "prog.jsonl"
    pl = ProgressList(total_epochs=4, verbose=False, log_file=str(log))
    for i in range(4):
        pl.append({"epoch": i + 1, "train_loss": 1.0 / (i + 1)})
    pl.close()

    loaded = load_training_history(log)
    assert len(loaded) == 4
    assert all("epoch" in r for r in loaded)


def test_plot_training_history_saves_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from surge.model.plot_training import plot_training_history

    save_path = tmp_path / "loss.png"
    fig = plot_training_history(_make_history(5), show=False, save_path=save_path)
    assert save_path.exists()
    assert fig is not None


def test_compare_training_histories(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from surge.model.plot_training import compare_training_histories

    save_path = tmp_path / "compare.png"
    fig = compare_training_histories(
        {
            "model_a": _make_history(4),
            "model_b": [{"epoch": i + 1, "train_loss": 2.0 / (i + 1)} for i in range(4)],
        },
        show=False,
        save_path=save_path,
    )
    assert save_path.exists()
    assert fig is not None


# ── BaseModelAdapter.training_history for sklearn ────────────────────────────


def test_sklearn_adapter_training_history_empty_before_fit():
    from surge.model.registry import MODEL_REGISTRY

    adapter = MODEL_REGISTRY.create("sklearn.ridge")
    assert adapter.training_history == []


def test_sklearn_adapter_training_history_empty_after_fit():
    from surge.model.registry import MODEL_REGISTRY

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    y = X[:, 0] + 0.1 * rng.standard_normal(60)

    adapter = MODEL_REGISTRY.create("sklearn.ridge")
    adapter.fit(X, y)
    assert adapter.training_history == []


def test_sklearn_adapter_plot_raises_when_no_history():
    """plot_training_history with empty history raises ValueError."""
    from surge.model.plot_training import plot_training_history

    with pytest.raises(ValueError):
        plot_training_history([])


# ── UQ metrics ────────────────────────────────────────────────────────────────


def _make_uq_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n)
    return y_true


def test_uq_metrics_keys():
    from surge.benchmarks.tasks import _uq_metrics

    y = _make_uq_data()
    mean_pred = y + 0.1 * np.random.default_rng(1).standard_normal(len(y))
    std_pred = np.full(len(y), 0.2)

    result = _uq_metrics(y, mean_pred, std_pred)
    assert set(result.keys()) == {"uq_picp", "uq_mpiw", "uq_crps", "uq_nll"}


def test_uq_metrics_picp_perfect_coverage():
    """Very wide std → PICP should be high (close to 1)."""
    from surge.benchmarks.tasks import _uq_metrics

    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)
    mean_pred = np.zeros(n)
    std_pred = np.full(n, 100.0)  # extremely wide intervals

    result = _uq_metrics(y, mean_pred, std_pred)
    assert result["uq_picp"] >= 0.90


def test_uq_metrics_picp_narrow_intervals():
    """Very narrow std → PICP should be below nominal coverage."""
    from surge.benchmarks.tasks import _uq_metrics

    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)
    mean_pred = np.zeros(n)
    std_pred = np.full(n, 1e-4)  # extremely narrow

    result = _uq_metrics(y, mean_pred, std_pred)
    assert result["uq_picp"] < 0.95


def test_uq_metrics_crps_improves_with_better_predictions():
    """Better predictions (closer to y_true) → lower CRPS."""
    from surge.benchmarks.tasks import _uq_metrics

    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)

    good_mean = y + 0.01 * rng.standard_normal(n)
    bad_mean = rng.standard_normal(n)  # unrelated noise
    std = np.full(n, 0.5)

    good = _uq_metrics(y, good_mean, std)["uq_crps"]
    bad = _uq_metrics(y, bad_mean, std)["uq_crps"]
    assert good < bad


def test_uq_metrics_nll_finite():
    from surge.benchmarks.tasks import _uq_metrics

    rng = np.random.default_rng(0)
    n = 100
    y = rng.standard_normal(n)
    mean_pred = y + 0.1 * rng.standard_normal(n)
    std_pred = np.full(n, 0.5)

    result = _uq_metrics(y, mean_pred, std_pred)
    assert np.isfinite(result["uq_nll"])


# ── HPO cache ─────────────────────────────────────────────────────────────────


def test_hpo_cache_round_trip(tmp_path):
    from surge.benchmarks.hpo import load_hpo_cache, save_hpo_cache

    params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}
    save_hpo_cache("tabular.iris", "sklearn.random_forest", params, root=tmp_path)
    loaded = load_hpo_cache("tabular.iris", "sklearn.random_forest", root=tmp_path)
    assert loaded == params


def test_hpo_cache_returns_none_when_missing(tmp_path):
    from surge.benchmarks.hpo import load_hpo_cache

    result = load_hpo_cache("nonexistent.bench", "no.model", root=tmp_path)
    assert result is None


def test_hpo_cache_overwrites(tmp_path):
    from surge.benchmarks.hpo import load_hpo_cache, save_hpo_cache

    save_hpo_cache("tabular.iris", "sklearn.ridge", {"alpha": 1.0}, root=tmp_path)
    save_hpo_cache("tabular.iris", "sklearn.ridge", {"alpha": 2.5}, root=tmp_path)
    loaded = load_hpo_cache("tabular.iris", "sklearn.ridge", root=tmp_path)
    assert loaded == {"alpha": 2.5}


# ── New benchmark registration ────────────────────────────────────────────────


def test_plasma_benchmarks_registered():
    from surge.benchmarks.registry import list_benchmarks

    keys = set(list_benchmarks())
    expected = {
        "plasma.cmod_density_limit",
        "plasma.qlknn_transport",
        "plasma.constellaration",
        "plasma.constellaration_paper",
    }
    assert expected.issubset(keys), f"Missing: {expected - keys}"


def test_ctr23_benchmarks_registered():
    from surge.benchmarks.registry import list_benchmarks

    keys = set(list_benchmarks())
    ctr23 = {
        "ctr23.abalone",
        "ctr23.bike_sharing",
        "ctr23.diamonds",
        "ctr23.house_sales",
        "ctr23.brazilian_houses",
    }
    assert ctr23.issubset(keys), f"Missing CTR-23: {ctr23 - keys}"


def test_constellaration_paper_benchmark_metadata():
    from surge.benchmarks.registry import benchmark_info

    info = benchmark_info("plasma.constellaration_paper")
    assert info["tier"] == "3"
    assert info["task_type"] == "regression"
    assert "90" in info["shape"]  # shape string mentions 90 input features
    assert info["description"]


def test_benchmark_runner_works_tabular():
    """Sanity: the generic benchmark runner can run tabular.california_housing."""
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "tabular.california_housing", model_key="sklearn.ridge", seed=0
    )
    assert result is not None
    assert "test_r2" in result.metrics
