"""Tests for surge.benchmarks.dataset_io (FeatureSpec kind + output_kinds)."""
from __future__ import annotations

from surge.benchmarks.dataset_io import (
    CONSTELLARATION_MULTIOUTPUT_IO,
    QLKNN_TRANSPORT_IO,
    output_kinds,
)


def test_output_kinds_defaults_to_equilibrium_when_unset():
    # QLKNN has a single output with no "kind" set — must default, not error.
    kinds = output_kinds(QLKNN_TRANSPORT_IO)
    assert kinds == {"efeITG": "equilibrium"}


def test_constellaration_multioutput_trains_only_the_8_equilibrium_metrics():
    # The joint benchmark now trains on equilibrium-kind targets only — the
    # geometry (closed-form) and derived (algebraic) labels are dropped from
    # the spec entirely, not just reported as a sanity check.
    outputs = CONSTELLARATION_MULTIOUTPUT_IO["outputs"]
    assert len(outputs) == 8

    kinds = output_kinds(CONSTELLARATION_MULTIOUTPUT_IO)
    assert len(kinds) == 8
    assert set(kinds.values()) == {"equilibrium"}

    names = {o["name"] for o in outputs}
    assert "aspect_ratio" not in names
    assert "max_elongation" not in names
    assert "average_triangularity" not in names
    assert "aspect_ratio_over_edge_rotational_transform" not in names
    assert "log_10_qi" in names
    assert "vacuum_well" in names
