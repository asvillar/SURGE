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
    assert set(sections) == {STATUS_TITLES[s] for s in STATUS_ORDER}
    for key, m in meta.items():
        homes = [s for s, ks in sections.items() if key in ks]
        assert homes == [STATUS_TITLES[m["status"]]], key
    assert "0.916" in md and "0.937" not in md
    row = next(ln for ln in md.splitlines() if ln.startswith("| `fusion.m3dc1_sample`"))
    assert "(synthetic fixture)" in row
    assert md == render_status_md(tmp_path)  # deterministic
