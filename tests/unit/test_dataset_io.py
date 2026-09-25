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


def test_constellaration_output_kinds_are_3_geometry_1_derived_8_equilibrium():
    kinds = output_kinds(CONSTELLARATION_MULTIOUTPUT_IO)
    assert len(kinds) == 12
    counts: dict[str, int] = {}
    for kind in kinds.values():
        counts[kind] = counts.get(kind, 0) + 1
    assert counts == {"geometry": 3, "derived": 1, "equilibrium": 8}

    assert kinds["aspect_ratio"] == "geometry"
    assert kinds["max_elongation"] == "geometry"
    assert kinds["average_triangularity"] == "geometry"
    assert kinds["aspect_ratio_over_edge_rotational_transform"] == "derived"
    assert kinds["log_10_qi"] == "equilibrium"
    assert kinds["vacuum_well"] == "equilibrium"
