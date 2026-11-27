"""Leaderboard metadata-key mapping and the matching registry alias (no training)."""
from surge.benchmarks import leaderboard
from surge.benchmarks.registry import resolve_benchmark_key


def test_metadata_key_mapping_contents():
    assert leaderboard._METADATA_KEY == {
        "tabular.covertype": "classification.covertype",
        "tabular.plasma_stability": "classification.plasma_stability",
    }


def test_mapped_keys_have_dataset_loaders():
    loaders = set(leaderboard.dataset_loader_keys())
    for key in leaderboard._METADATA_KEY.values():
        assert key in loaders, key


def test_classification_alias_resolves_through_registry():
    assert resolve_benchmark_key("classification.covertype") == "tabular.covertype"
    assert resolve_benchmark_key("classification.plasma_stability") == "tabular.plasma_stability"


def test_parallel_collect_reads_mapped_directory(tmp_path):
    import json

    from surge.benchmarks.run import _collect_parallel_results

    d = tmp_path / "classification.covertype" / "some.model"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "benchmark_key": "classification.covertype",
        "tier": "tier1",
        "task_type": "classification",
        "metrics": {"accuracy": 0.9},
        "passed": True,
        "model_key": "some.model",
    }))
    out = _collect_parallel_results(tmp_path, ["tabular.covertype"], 0.0)
    assert list(out) == ["classification.covertype"]
    assert out["classification.covertype"][0].model_key == "some.model"
