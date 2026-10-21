"""Tests for the benchmark status ledger renderer (surge.benchmarks.status)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from surge.benchmarks.status import STATUS_ORDER, STATUS_TITLES, render_status_md
from surge.report.leaderboard import load_metadata


def _res(root: Path, bench: str, model: str, ts: str, r2: float, **extra):
    d = root / bench / ts
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "benchmark_key": bench, "model_key": model, "timestamp": ts,
        "metrics": {"test_r2": r2}, "passed": r2 > 0.5, "extra": extra,
    }))


def test_each_benchmark_once_under_its_status(tmp_path):
    meta = load_metadata()
    _res(tmp_path, "plasma.constellaration_multioutput", "old", "2026-06-15T06:24:26Z", 0.937)
    _res(tmp_path, "plasma.constellaration_multioutput", "new", "2026-09-25T18-05-41Z", 0.916)
    _res(tmp_path, "fusion.m3dc1_sample", "ridge", "2026-08-01T00:00:00Z", 0.9, synthetic=True)
    md = render_status_md(tmp_path)
    sections = {}
    cur = None
    for line in md.splitlines():
        if line.startswith("## "):
            cur = line[3:]
            sections[cur] = []
        elif cur and line.startswith("| `"):
            sections[cur].append(line.split("`")[1])
    assert set(sections) - {"Focus"} == {STATUS_TITLES[s] for s in STATUS_ORDER}
    for key, m in meta.items():
        homes = [s for s, ks in sections.items() if key in ks]
        assert homes == [STATUS_TITLES[m["status"]]], key
    assert "0.916" in md and "0.937" not in md
    row = next(ln for ln in md.splitlines() if ln.startswith("| `fusion.m3dc1_sample`"))
    assert "(synthetic fixture)" in row
    assert md == render_status_md(tmp_path)  # deterministic


def _sel(tmp_path, rows):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"benchmarks": rows}))
    return p


def test_apply_selection_sets_fields_and_leaves_rest(tmp_path, capsys):
    from surge.benchmarks.status import apply_selection

    src = Path(__file__).resolve().parents[2] / "surge/benchmarks/metadata.yaml"
    dst = tmp_path / "metadata.yaml"
    dst.write_text(src.read_text())
    before = load_metadata(dst)
    res = apply_selection(_sel(tmp_path, [
        {"key": "tabular.california_housing", "show": True, "push": True, "note": "headline"},
        {"key": "pde.burgers_1d", "show": False, "push": False, "note": ""},
        {"key": "nope.missing", "show": True, "push": True, "note": ""},
    ]), dst)
    after = load_metadata(dst)
    import yaml
    doc = yaml.safe_load(dst.read_text())
    entries = doc.get("benchmarks", doc) if isinstance(doc, dict) else doc
    assert res["unknown"] == ["nope.missing"]
    assert res["focus"] == sum(1 for m in entries if m.get("focus") is True)
    assert res["visible"] == sum(1 for m in entries if m.get("visible", True) is not False)
    assert "unknown benchmark key" in capsys.readouterr().err
    ch = after["tabular.california_housing"]
    assert ch["focus"] is True and ch["visible"] is True and ch["focus_note"] == "headline"
    assert after["pde.burgers_1d"]["visible"] is False
    for k, m in before.items():
        if k not in ("tabular.california_housing", "pde.burgers_1d"):
            assert after[k] == m
    strip = lambda m: {f: v for f, v in m.items() if f not in ("visible", "focus", "focus_note")}
    assert {k: strip(m) for k, m in after.items()} == {k: strip(m) for k, m in before.items()}
    assert dst.read_text().startswith("# SURGE benchmark metadata")  # comments survive
    apply_selection(_sel(tmp_path, [{"key": "tabular.california_housing", "show": True,
                                     "push": False, "note": ""}]), dst)
    assert "focus_note" not in load_metadata(dst)["tabular.california_housing"]


def test_status_ledger_focus_column_and_section(tmp_path):
    from surge.benchmarks.status import apply_selection

    src = Path(__file__).resolve().parents[2] / "surge/benchmarks/metadata.yaml"
    dst = tmp_path / "metadata.yaml"
    dst.write_text(src.read_text())
    md0 = render_status_md(tmp_path, dst)
    assert "| visible | focus |" in md0
    apply_selection(_sel(tmp_path, [{"key": "tabular.california_housing", "show": True,
                                     "push": True, "note": "key demo"}]), dst)
    md = render_status_md(tmp_path, dst)
    assert "## Focus" in md and "- `tabular.california_housing`: key demo" in md
    row = next(ln for ln in md.splitlines() if ln.startswith("| `tabular.california_housing`"))
    assert row.endswith("| ✓ | ✓ |")


DISCIPLINES = {
    "Fusion & Plasma Physics", "Nonlinear Dynamics & Chaos",
    "Fluid Dynamics & PDE Operators", "Materials Science", "Engineering",
    "Environmental Sciences", "Biology & Medicine", "Chemistry",
    "Operations Research", "Computer Vision",
}


def test_metadata_discipline_schema():
    meta = load_metadata()
    assert len(meta) >= 28
    for key, m in meta.items():
        assert m.get("discipline") in DISCIPLINES, key


def test_status_ledger_has_discipline_column(tmp_path):
    md = render_status_md(tmp_path)
    assert "| discipline |" in md
    row = next(ln for ln in md.splitlines()
               if ln.startswith("| `tabular.california_housing`"))
    assert "| Environmental Sciences |" in row
