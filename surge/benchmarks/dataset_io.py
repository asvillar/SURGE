"""Canonical input/output documentation for SURGE benchmark datasets."""

from __future__ import annotations

from typing import TypedDict


class _FeatureSpecRequired(TypedDict):
    name: str
    desc: str


class FeatureSpec(_FeatureSpecRequired, total=False):
    """One documented input or output feature of a benchmark dataset.

    ``kind`` (optional) classifies an *output* feature for datasets where
    different targets need fundamentally different amounts of computation
    to produce, so a single aggregate R² can hide that some targets are
    trivial and others are not. Recognised values:

    - ``"geometry"``   — a closed-form function of the raw inputs (e.g. a
      shape moment); a surrogate should recover it almost exactly.
    - ``"derived"``     — an algebraic combination of other labels (e.g. a
      ratio of two other outputs); not closed-form in the raw inputs, but
      not a full solve either.
    - ``"equilibrium"`` — requires the expensive numerical solve (e.g. a
      physics equilibrium solver); this is what a surrogate is actually for.

    Outputs without a ``kind`` are treated as ``"equilibrium"`` by
    :func:`output_kinds` so datasets that never set it are unaffected.
    """

    kind: str


class DatasetIOSpec(TypedDict, total=False):
    inputs: list[FeatureSpec]
    output: FeatureSpec
    outputs: list[FeatureSpec]
    note: str


QLKNN_TRANSPORT_IO: DatasetIOSpec = {
    "inputs": [
        {"name": "Ati", "desc": "Normalised ion temperature gradient"},
        {"name": "Ate", "desc": "Normalised electron temperature gradient"},
        {"name": "Ane", "desc": "Normalised electron density gradient"},
        {"name": "Ani", "desc": "Normalised ion density gradient"},
        {"name": "q", "desc": "Safety factor"},
        {"name": "smag", "desc": "Magnetic shear"},
        {"name": "x", "desc": "Normalised minor radius"},
        {"name": "Ti_Te", "desc": "Ion-to-electron temperature ratio"},
        {"name": "LogNuStar", "desc": "Log collisionality"},
        {"name": "normni", "desc": "Normalised ion density"},
    ],
    "output": {
        "name": "efeITG",
        "desc": "Electron heat flux from the ITG mode (gyroBohm-normalised units)",
    },
    "note": (
        "Ground truth is the QLKNN_7_11 surrogate (fusion_surrogates / DeepMind); "
        "SURGE models learn to emulate that mapping. Only samples with active ITG "
        "transport (efeITG > 0) are kept after cache generation."
    ),
}

QLKNN_FEATURE_NAMES: list[str] = [f["name"] for f in QLKNN_TRANSPORT_IO["inputs"]]
QLKNN_TARGET_NAME: str = QLKNN_TRANSPORT_IO["output"]["name"]

_CONSTELLARATION_OUTPUT_SPECS: list[FeatureSpec] = [
    {"name": "aspect_ratio", "desc": "Plasma aspect ratio", "kind": "geometry"},
    {
        "name": "aspect_ratio_over_edge_rotational_transform",
        "desc": "Aspect ratio / edge ι",
        "kind": "derived",
    },
    {"name": "max_elongation", "desc": "Maximum elongation", "kind": "geometry"},
    {
        "name": "axis_rotational_transform_over_n_field_periods",
        "desc": "Axis ι / n_fp",
        "kind": "equilibrium",
    },
    {
        "name": "edge_rotational_transform_over_n_field_periods",
        "desc": "Edge ι / n_fp",
        "kind": "equilibrium",
    },
    {
        "name": "axis_magnetic_mirror_ratio",
        "desc": "Axis magnetic mirror ratio",
        "kind": "equilibrium",
    },
    {
        "name": "edge_magnetic_mirror_ratio",
        "desc": "Edge magnetic mirror ratio",
        "kind": "equilibrium",
    },
    {"name": "average_triangularity", "desc": "Average triangularity", "kind": "geometry"},
    {"name": "vacuum_well", "desc": "Vacuum magnetic well", "kind": "equilibrium"},
    {
        "name": "minimum_normalized_magnetic_gradient_scale_length",
        "desc": "Min normalised |∇B| scale length",
        "kind": "equilibrium",
    },
    {
        "name": "flux_compression_in_regions_of_bad_curvature",
        "desc": "Flux compression in bad-curvature regions",
        "kind": "equilibrium",
    },
    {
        "name": "log_10_qi",
        "desc": "log₁₀ quasi-isodynamic quality (lower = better QI)",
        "kind": "equilibrium",
    },
]

CONSTELLARATION_INPUT_NAMES: list[str] = (
    [f"r_cos_{i}" for i in range(45)] + [f"z_sin_{i}" for i in range(45)]
)
CONSTELLARATION_OUTPUT_NAMES: list[str] = [f["name"] for f in _CONSTELLARATION_OUTPUT_SPECS]

CONSTELLARATION_MULTIOUTPUT_IO: DatasetIOSpec = {
    "inputs": [
        {"name": "r_cos[0:44]", "desc": "45 boundary R Fourier cos coefficients (5×9 VMEC modes)"},
        {"name": "z_sin[0:44]", "desc": "45 boundary Z Fourier sin coefficients (5×9 VMEC modes)"},
    ],
    "outputs": _CONSTELLARATION_OUTPUT_SPECS,
    "note": (
        "Single joint model: 90 boundary shape features → 12 stellarator metrics. "
        "Filtered cache: nfp=3, optimised DESC/VMEC, 0.05% outlier clip (26,897 rows). "
        "Contrast with plasma.constellaration_paper (12 independent 90→1 models)."
    ),
}

_BENCHMARK_IO: dict[str, DatasetIOSpec] = {
    "plasma.qlknn_transport": QLKNN_TRANSPORT_IO,
    "plasma.constellaration_multioutput": CONSTELLARATION_MULTIOUTPUT_IO,
}


def get_benchmark_io(benchmark_key: str) -> DatasetIOSpec | None:
    return _BENCHMARK_IO.get(benchmark_key)


def output_kinds(io: DatasetIOSpec) -> dict[str, str]:
    """Map each output name to its ``kind`` ("geometry" | "derived" | "equilibrium").

    Outputs without an explicit ``kind`` default to ``"equilibrium"`` so
    datasets that never set it (i.e. everything except ConStellaration) are
    unaffected.
    """
    outputs = io.get("outputs") or ([io["output"]] if io.get("output") else [])
    return {o["name"]: o.get("kind", "equilibrium") for o in outputs}


def format_inputs_short(io: DatasetIOSpec) -> str:
    return ", ".join(f["name"] for f in io["inputs"])


def format_io_summary(io: DatasetIOSpec) -> str:
    lines = [f"Inputs ({len(io['inputs'])}): " + format_inputs_short(io)]
    if io.get("outputs"):
        names = ", ".join(o["name"] for o in io["outputs"])
        lines.append(f"Outputs ({len(io['outputs'])}): {names}")
    elif io.get("output"):
        out = io["output"]
        lines.append(f"Output: {out['name']} — {out['desc']}")
    if io.get("note"):
        lines.append(io["note"])
    return "\n".join(lines)
