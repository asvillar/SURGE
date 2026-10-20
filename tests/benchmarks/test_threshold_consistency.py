"""metadata.yaml thresholds must match the code gates in ``_check_pass``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from surge.benchmarks.leaderboard import _check_pass

META = Path(__file__).resolve().parents[2] / "surge" / "benchmarks" / "metadata.yaml"
_PAT = re.compile(r"^(R²|Acc)\s*≥\s*([0-9.]+)$")
_METRIC = {"R²": "test_r2", "Acc": "test_accuracy"}
FOCUS_KEYS = [
    "plasma.qlknn10d",
    "plasma.constellaration",
    "classification.covertype",
    "classification.plasma_stability",
]
# Pre-existing drift outside this work-unit; shrink this set, never grow it.
KNOWN_DRIFT = {
    "tabular.airfoil_noise", "tabular.yacht_dynamics", "tabular.diabetes",
    "tabular.breast_cancer", "tabular.digits", "tabular.iris",
    "fusion.m3dc1_sample", "plasma.cmod_density_limit",
}


def _meta_thresholds() -> dict[str, tuple[str, float]]:
    rows = yaml.safe_load(META.read_text(encoding="utf-8"))["benchmarks"]
    out = {}
    for r in rows:
        m = _PAT.match(str(r.get("threshold", "")).strip())
        if m:
            out[r["key"]] = (_METRIC[m.group(1)], float(m.group(2)))
    return out


def _gate_matches(key: str, metric: str, thr: float) -> bool:
    """True when the code gate passes at ``thr`` and fails just below it."""
    return _check_pass(key, {metric: thr}) and not _check_pass(key, {metric: thr - 1e-3})


@pytest.mark.parametrize("key", FOCUS_KEYS)
def test_focus_benchmark_gate_matches_metadata(key):
    meta = _meta_thresholds()
    assert key in meta, f"{key}: metadata threshold missing or unparsable"
    metric, thr = meta[key]
    assert _gate_matches(key, metric, thr)


def test_all_simple_thresholds_match_metadata():
    bad = [k for k, (m, t) in _meta_thresholds().items() if not _gate_matches(k, m, t)]
    new = sorted(set(bad) - KNOWN_DRIFT)
    assert not new, f"code gate differs from metadata threshold: {new}"
