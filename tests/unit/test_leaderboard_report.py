"""Tests for the artifact-driven leaderboard report (surge.report.leaderboard)."""
from __future__ import annotations

import json
import re
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
    # a metadata copy whose california_housing definition changed between the
    # old (better) run and the new one: the old run must leave the headline
    src = Path(__file__).resolve().parents[2] / "surge/benchmarks/metadata.yaml"
    mp = tmp_path / "metadata.yaml"
    mp.write_text(src.read_text().replace("- key: tabular.california_housing\n",
                                          "- key: tabular.california_housing\n"
                                          "  definition_date: 2026-01-15\n", 1))
    out = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                       demo_runs={}, metadata_path=mp)
    text = out.read_text()
    tail_at = text.index("<details class='tail'>")
    assert text.index('id="pde.burgers_1d"') > tail_at          # sanity only in tail
    assert 'id="pde.burgers_1d"' not in text[:tail_at]
    assert "synthetic fixture" in text
    # headline excludes the run that predates the definition date; history keeps it
    head = text[text.index('id="tabular.california_housing"'):]
    first_table = head[:head.index("</table>")]
    assert "new.model" in first_table and "old.model" not in first_table
    assert "run history (all 2 results)" in head
    # without a definition date, each model's newest run is kept (old.model stays)
    plain = build_report(root, tmp_path / "lb2.html", video=tmp_path / "none.mp4",
                         demo_runs={}).read_text()
    head2 = plain[plain.index('id="tabular.california_housing"'):]
    assert "old.model" in head2[:head2.index("</table>")]
    assert len(text.encode()) < 1_000_000


def test_full_layout_expands_tail_benchmarks(tmp_path: Path):
    root = _fake_tree(tmp_path)
    text = build_report(root, tmp_path / "full.html", video=tmp_path / "none.mp4",
                        demo_runs={}, layout="full").read_text()
    assert "<details class='tail'>" not in text
    # sanity benchmark is rendered expanded, with its status badge still shown
    bench_at = text.index('id="pde.burgers_1d"')
    card = text[bench_at:text.index("</section>", bench_at)]
    assert "<span class='badge muted'>sanity</span>" in card
    # the spider chart for its category is still rendered (full layout keeps
    # the category blocks, just not collapsed)
    assert "<h2 class=\"cat\">" in text
    # the status layout still behaves as before (default, unchanged)
    status_text = build_report(root, tmp_path / "status.html", video=tmp_path / "none.mp4",
                               demo_runs={}).read_text()
    assert "<details class='tail'>" in status_text


def test_synthetic_excluded_from_pass_count(tmp_path: Path):
    root = _fake_tree(tmp_path)
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}).read_text()
    # fusion.m3dc1_sample is not ready/real: tiles count only ready real runs;
    # both california models (old + new, no definition date) count, both pass
    assert "<b>2 / 2</b>" in text


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
    import yaml
    doc = yaml.safe_load(mp.read_text())
    entries = doc.get("benchmarks", doc) if isinstance(doc, dict) else doc
    n_hidden = sum(1 for m in entries if m.get("visible") is False
                   and (root / m["key"]).exists())
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}, metadata_path=mp).read_text()
    assert 'id="pde.burgers_1d"' not in text
    assert f"{n_hidden} benchmarks hidden by the maintainer" in text
    assert "<section class='focus'>" in text
    assert "flagship tabular" not in text          # notes are internal by default
    assert "<span class='badge good'>focus</span>" in text
    internal = build_report(root, tmp_path / "lbm.html", video=tmp_path / "none.mp4",
                            demo_runs={}, metadata_path=mp, maintainer=True).read_text()
    assert "flagship tabular" in internal
    # a metadata file with no selection applied shows neither
    pristine = tmp_path / "pristine.yaml"
    pristine.write_text(re.sub(r"\n  (visible|focus|focus_note):[^\n]*", "", src.read_text()))
    plain = build_report(root, tmp_path / "p.html", video=tmp_path / "none.mp4",
                         demo_runs={}, metadata_path=pristine).read_text()
    assert "hidden by the maintainer" not in plain and "class='focus'" not in plain


SOTA_KEYS = {"value", "metric", "model", "source", "url", "protocol_match", "note"}


def test_metadata_schema_explainer_sota_io():
    from surge.report.leaderboard import load_metadata

    meta = load_metadata()
    assert len(meta) >= 28
    for key, m in meta.items():
        assert str(m.get("task_explainer") or "").strip(), key
        assert str(m.get("published_in") or "").strip(), key
        assert isinstance(m.get("sota"), dict) and set(m["sota"]) == SOTA_KEYS, key
        assert m["sota"]["protocol_match"] in {"same", "different", "unknown"}, key
        if m["sota"]["value"] is None:
            assert "no verified published number" in m["sota"]["note"], key
        assert m.get("inputs") and m.get("outputs"), key
        for f in m["inputs"] + m["outputs"]:
            assert f.get("name") and f.get("desc"), key


def _fake_meta(tmp_path: Path) -> Path:
    import yaml

    mp = tmp_path / "meta.yaml"
    mp.write_text(yaml.safe_dump({"benchmarks": [{
        "key": "tabular.california_housing", "name": "Fake", "status": "ready",
        "primary_metric": "R²", "shape": "2 → 1", "n": "10",
        "task_explainer": "Maps widgets to gizmos for testing.",
        "published_in": "Journal of Tests (2020)", "url": "https://example.org/p",
        "inputs": [{"name": "widget", "desc": "w"}], "outputs": [{"name": "gizmo", "desc": "g"}],
        "sota": {"value": 0.95, "metric": "R²", "model": "FakeNet", "source": "Doe 2020",
                 "url": "https://example.org/s", "protocol_match": "different", "note": "n"},
    }]}))
    return mp


def test_report_renders_about_and_sota(tmp_path: Path):
    root = _fake_tree(tmp_path)
    text = build_report(root, tmp_path / "s.html", video=tmp_path / "none.mp4",
                        demo_runs={}, metadata_path=_fake_meta(tmp_path)).read_text()
    assert "About this task" in text and "Maps widgets to gizmos" in text
    assert "Journal of Tests (2020)" in text
    assert "FakeNet 0.95 R² (Doe 2020, protocol: different)" in text
    assert "class='sota-mark'" in text  # dashed reference marker drawn in the bar


def test_ledger_has_sota_column(tmp_path: Path):
    from surge.benchmarks.status import render_status_md

    md = render_status_md(_fake_tree(tmp_path), _fake_meta(tmp_path))
    assert "| SOTA |" in md and "0.95 R² [diff]" in md


# --------------------------------------------------------------- index table

def test_index_table_has_one_row_per_visible_benchmark(tmp_path: Path):
    root = _fake_tree(tmp_path)
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}).read_text()
    assert 'id="bench-index"' in text
    # fusion.m3dc1_sample is hidden by metadata.yaml (visible: False), so only
    # 2 of the 3 benchmarks in the fake tree are visible/indexed
    assert text.count("<tr data-key='") == 2
    for key in ("tabular.california_housing", "pde.burgers_1d"):
        assert f"<tr data-key='{key}'" in text
        assert f"<a href='#{key}' class='idx-link'" in text
    assert "<tr data-key='fusion.m3dc1_sample'" not in text
    # best-model haystack carries the winning model's key for search
    assert "const BENCH_INDEX" in text
    assert "new.model" in text[text.index("const BENCH_INDEX"):text.index("</script>",
           text.index("const BENCH_INDEX"))]


def test_index_table_excludes_maintainer_hidden_benchmark(tmp_path: Path):
    from surge.benchmarks.status import apply_selection

    root = _fake_tree(tmp_path)
    src = Path(__file__).resolve().parents[2] / "surge/benchmarks/metadata.yaml"
    mp = tmp_path / "metadata.yaml"
    mp.write_text(src.read_text())
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"benchmarks": [
        {"key": "pde.burgers_1d", "show": False, "push": False, "note": ""}]}))
    apply_selection(sel, mp)
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}, metadata_path=mp).read_text()
    assert "<tr data-key='pde.burgers_1d'" not in text
    assert "<tr data-key='tabular.california_housing'" in text


def test_index_section_has_search_filter_and_count_controls(tmp_path: Path):
    root = _fake_tree(tmp_path)
    text = build_report(root, tmp_path / "lb.html", video=tmp_path / "none.mp4",
                        demo_runs={}).read_text()
    assert 'id="idx-search"' in text
    assert 'id="idx-follow"' in text
    assert 'id="idx-count"' in text
    assert "chip-toggle" in text
    assert "&uarr; index" in text  # card backlink
