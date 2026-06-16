"""Canonical input/output documentation for SURGE benchmark datasets."""

from __future__ import annotations

from typing import TypedDict


class FeatureSpec(TypedDict):
    name: str
    desc: str


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
    {"name": "aspect_ratio", "desc": "Plasma aspect ratio"},
    {"name": "aspect_ratio_over_edge_rotational_transform", "desc": "Aspect ratio / edge ι"},
    {"name": "max_elongation", "desc": "Maximum elongation"},
    {"name": "axis_rotational_transform_over_n_field_periods", "desc": "Axis ι / n_fp"},
    {"name": "edge_rotational_transform_over_n_field_periods", "desc": "Edge ι / n_fp"},
    {"name": "axis_magnetic_mirror_ratio", "desc": "Axis magnetic mirror ratio"},
    {"name": "edge_magnetic_mirror_ratio", "desc": "Edge magnetic mirror ratio"},
    {"name": "average_triangularity", "desc": "Average triangularity"},
    {"name": "vacuum_well", "desc": "Vacuum magnetic well"},
    {"name": "minimum_normalized_magnetic_gradient_scale_length", "desc": "Min normalised |∇B| scale length"},
    {"name": "flux_compression_in_regions_of_bad_curvature", "desc": "Flux compression in bad-curvature regions"},
    {"name": "log_10_qi", "desc": "log₁₀ quasi-isodynamic quality (lower = better QI)"},
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
