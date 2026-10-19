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


# ------------------------------------------------------------ status-aware v2

def _fake_tree(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark_reports"
    # ready: old better run (>3 h before) must be excluded from the headline
    _write_result(root, "tabular.california_housing", "old.model",
                  "2026-01-01T00-00-00Z", test_r2=0.99)
    _write_result(root, "tabular.california_housing", "new.model",
                  "2026-02-01T00-00-00Z", test_r2=0.80)
    # sanity
    _write_result(root, "pde.burgers_1d", "sanity.model",
                  "2026-02-01T00-00-00Z", test_r2=1.0)
    # synthetic fixture
    d = root / "fusion.m3dc1_sample" / "2026-02-01T00-00-00Z"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "benchmark_key": "fusion.m3dc1_sample", "model_key": "syn.model",
        "metrics": {"test_r2": 0.9}, "passed": True,
        "timestamp": "2026-02-01T00-00-00Z", "extra": {"synthetic": True}}))
    return root


def test_status_layout_synthetic_and_newest_sweep(tmp_path: Path):
    meta = load_metadata()
    assert meta["pde.burgers_1d"]["status"] == "sanity"
    root = _fake_tree(tmp_path)
    out = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                       demo_runs={})
    text = out.read_text()
    tail_at = text.index("<details class='tail'>")
    assert text.index('id="pde.burgers_1d"') > tail_at          # sanity only in tail
    assert 'id="pde.burgers_1d"' not in text[:tail_at]
    assert "synthetic fixture" in text
    # headline excludes the older, better run; history table keeps it
    head = text[text.index('id="tabular.california_housing"'):]
    first_table = head[:head.index("</table>")]
    assert "new.model" in first_table and "old.model" not in first_table
    assert "run history (all 2 results)" in head
    assert "newest sweep only" in text
    assert len(text.encode()) < 1_000_000


def test_synthetic_excluded_from_pass_count(tmp_path: Path):
    root = _fake_tree(tmp_path)
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}).read_text()
    # fusion.m3dc1_sample is not ready/real: tiles count only ready real runs
    assert "<b>1 / 1</b>" in text


def test_demo_section_embeds_video_when_present(tmp_path: Path):
    root = _fake_tree(tmp_path)
    mp4 = tmp_path / "demo.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
    with_v = build_report(root, tmp_path / "a.html", video=mp4, demo_runs={}).read_text()
    assert "data:video/mp4;base64," in with_v and "<video controls muted loop" in with_v
    no_v = build_report(root, tmp_path / "b.html", video=tmp_path / "missing.mp4",
                        demo_runs={}).read_text()
    assert "data:video/mp4" not in no_v


def test_budget_drops_mission_control(tmp_path: Path):
    root = _fake_tree(tmp_path)
    anim = tmp_path / "anim"
    anim.mkdir()
    (anim / "hpo_evolution_dark.webm").write_bytes(b"a" * 100)
    (anim / "mission_control_dark.webm").write_bytes(b"b" * 100_000)
    run = tmp_path / "run"
    run.mkdir()
    kw = dict(video=tmp_path / "x.mp4", demo_runs={"tabular.california_housing": str(run)},
              anim_dir=anim)
    full = build_report(root, tmp_path / "a.html", **kw).read_text()
    assert full.count("data:video/webm") == 2
    small = build_report(root, tmp_path / "b.html", budget_mb=0.05, **kw).read_text()
    assert small.count("data:video/webm") == 1 and "Mission-control clip dropped" in small


def test_hidden_omitted_and_focus_section(tmp_path: Path):
    from surge.benchmarks.status import apply_selection

    root = _fake_tree(tmp_path)
    src = Path(__file__).resolve().parents[2] / "surge/benchmarks/metadata.yaml"
    mp = tmp_path / "metadata.yaml"
    mp.write_text(src.read_text())
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"benchmarks": [
        {"key": "pde.burgers_1d", "show": False, "push": False, "note": ""},
        {"key": "tabular.california_housing", "show": True, "push": True,
         "note": "flagship tabular"}]}))
    apply_selection(sel, mp)
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}, metadata_path=mp).read_text()
    assert 'id="pde.burgers_1d"' not in text
    assert "1 benchmarks hidden by the maintainer" in text
    assert "<section class='focus'>" in text and "flagship tabular" in text
    assert "<span class='badge good'>focus</span>" in text
    plain = build_report(root, tmp_path / "p.html", video=tmp_path / "none.mp4",
                         demo_runs={}).read_text()
    assert "hidden by the maintainer" not in plain and "class='focus'" not in plain
