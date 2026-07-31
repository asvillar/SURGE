"""Dispatch table for standard benchmarks.

Each benchmark has a **category** that describes its scientific purpose:

    smoke       — inline synthetic fixtures; used only for fast unit-level
                  CI smoke tests.  Never shown in --list by default.
    tabular     — real-world tabular regression / classification datasets.
                  Includes UCI legacy, modern CTR-23, multi-output regression
                  (SCM20d), and sequence-to-sequence tasks solvable by any
                  tabular regressor (Lorenz-63 windowed prediction).
    image       — image tensor (H×W×C) input / output, e.g. CIFAR-10, MNIST.
                  (Legacy alias: ``vision``.)
    field       — discretised function/field → function/field.  ``field`` covers
                  any benchmark whose input and output are discretised
                  functions/fields, including but not limited to PDEs.
                  (Legacy alias: ``pde``.)
    plasma      — fusion / plasma-physics surrogates (QLKNN, ConStellaration, …).

The ``surge run -b cifar10 -m all`` short-name syntax is handled via
``_SHORT_ALIASES`` below and resolved in ``run.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BenchmarkResult
from .tasks import (
    run_classification_covertype,
    run_classification_flow_regime,
    run_classification_plasma_stability,
    run_fusion_m3dc1_sample,
    run_multioutput_scm20d,
    run_pde_burgers_1d,
    run_pdebench_burgers_1d,
    run_pdebench_darcy_2d,
    run_pdebench_shallow_water_2d,
    run_sequence_lorenz63,
    run_tabular_superconductor,
    run_thewell_gray_scott,
    run_thewell_mhd,
    run_thewell_turbulence_2d,
    run_vision_cifar10,
    run_vision_mnist,
    run_synthetic_classification_binary,
    run_synthetic_multioutput_2d,
    run_synthetic_regression_1d,
    run_tabular_airfoil_noise,
    run_tabular_breast_cancer,
    run_tabular_california_housing,
    run_tabular_concrete_strength,
    run_tabular_diabetes,
    run_tabular_digits,
    run_tabular_energy_efficiency,
    run_tabular_iris,
    run_tabular_wine,
    run_tabular_yacht_dynamics,
)

# ─── Benchmark metadata ───────────────────────────────────────────────────────
# Each entry: (runner_fn, category, task_type, shape_desc, description)
#
# category choices: smoke | tabular | image | field | plasma
# Legacy aliases: vision→image, pde→field, sequence→tabular, multioutput→tabular
#
# 'smoke' benchmarks are hidden from --list by default (pass --smoke to show).
# All other categories are shown and are selectable with --category <name>.

_META: dict[str, tuple[Callable, str, str, str, str]] = {

    # ── Smoke tests (inline synthetic fixtures) ───────────────────────────────
    # These are NOT scientific benchmarks.  They exist solely so CI can check
    # that models run end-to-end without downloading any data.
    "synthetic.regression_1d": (
        run_synthetic_regression_1d, "smoke", "regression", "1→1 (n=400)",
        "Linear 1-D signal with Gaussian noise — inline smoke test only",
    ),
    "synthetic.multioutput_2d": (
        run_synthetic_multioutput_2d, "smoke", "regression", "8→2 (n=600)",
        "Multi-output 8→2 linear regression with Gaussian noise — inline smoke test only",
    ),
    "synthetic.classification_binary": (
        run_synthetic_classification_binary, "smoke", "classification", "20→2 (n=500)",
        "Binary labels from linear combo of features — inline smoke test only",
    ),
    "classification.flow_regime": (
        run_classification_flow_regime, "smoke", "classification", "3→4 (n=800)",
        "CFD flow regime 4-class labelling from Mach/Re/AoA — inline smoke test only",
    ),

    # ── Tabular: UCI legacy (sanity / baseline reference) ─────────────────────
    # Well-known datasets from the UCI ML Repository.  Useful as sanity checks
    # and for comparing against published MLP / tree baselines, but results
    # should not be treated as rigorous scientific benchmarks on their own.
    "tabular.diabetes": (
        run_tabular_diabetes, "tabular", "regression", "10→1 (n=442)",
        "UCI Diabetes / sklearn.datasets (Efron et al. 2004)",
    ),
    "tabular.california_housing": (
        run_tabular_california_housing, "tabular", "regression", "8→1 (n=20,640)",
        "California Housing / sklearn.datasets (Pace & Barry 1997)",
    ),
    "tabular.concrete_strength": (
        run_tabular_concrete_strength, "tabular", "regression", "8→1 (n=1,030)",
        "UCI Concrete Compressive Strength (Yeh 1998) [requires internet on first run]",
    ),
    "tabular.energy_efficiency": (
        run_tabular_energy_efficiency, "tabular", "regression", "8→1 (n=768)",
        "UCI Energy Efficiency — Heating Load (Tsanas & Xifara 2012) [requires internet]",
    ),
    "tabular.airfoil_noise": (
        run_tabular_airfoil_noise, "tabular", "regression", "5→1 (n=1,503)",
        "NASA Airfoil Self-Noise (Brooks et al. 1989) — UCI [requires internet]",
    ),
    "tabular.yacht_dynamics": (
        run_tabular_yacht_dynamics, "tabular", "regression", "6→1 (n=308)",
        "UCI Yacht Hydrodynamics (Gerritsma 1981) [requires internet]",
    ),
    "tabular.superconductor": (
        run_tabular_superconductor, "tabular", "regression", "81→1 (n=21,263)",
        "Superconductor Tc prediction (Hamidieh 2018) [requires internet]",
    ),
    "tabular.iris": (
        run_tabular_iris, "tabular", "classification", "4→3 (n=150)",
        "UCI Iris / sklearn.datasets (Fisher 1936)",
    ),
    "tabular.breast_cancer": (
        run_tabular_breast_cancer, "tabular", "classification", "30→2 (n=569)",
        "Wisconsin Breast Cancer / sklearn.datasets (UCI WDBC)",
    ),
    "tabular.wine": (
        run_tabular_wine, "tabular", "classification", "13→3 (n=178)",
        "UCI Wine / sklearn.datasets",
    ),
    "tabular.digits": (
        run_tabular_digits, "tabular", "classification", "64→10 (n=1,797)",
        "Optical digits / sklearn.datasets (Alpaydin 1998)",
    ),
    "tabular.covertype": (
        run_classification_covertype, "tabular", "classification", "54→7 (n=20k subsample)",
        "Forest Covertype 7-class classification (Blackard & Dean 1999) [requires internet]",
    ),
    "tabular.plasma_stability": (
        run_classification_plasma_stability, "tabular", "classification", "12→2 (n=10,000)",
        "UCI Electrical Grid Stability (Arzamasov 2018) [requires internet]",
    ),

    # ── Tabular: CTR-23 modern suite ──────────────────────────────────────────
    # Grinsztajn et al. (2022) "Why tree-based models still outperform deep
    # learning on tabular data" arXiv:2207.08815.
    # These are the recommended replacements for legacy UCI tabular benchmarks.
    "ctr23.abalone": (
        None, "tabular", "regression", "8→1 (n=4,177)",
        "Abalone age prediction — OpenML #183. Grinsztajn et al. 2022. "
        "[R² reference: RF≈0.57, XGB≈0.59, MLP≈0.53]",
    ),
    "ctr23.bike_sharing": (
        None, "tabular", "regression", "12→1 (n=17,389)",
        "Hourly bike rentals — OpenML #42712. Grinsztajn et al. 2022. "
        "[R² reference: RF≈0.97, XGB≈0.98, MLP≈0.93]",
    ),
    "ctr23.diamonds": (
        None, "tabular", "regression", "9→1 (n=15k subsample)",
        "Diamond price prediction — OpenML #42225. Grinsztajn et al. 2022. "
        "[R² reference: RF≈0.98, XGB≈0.98, MLP≈0.97]",
    ),
    "ctr23.house_sales": (
        None, "tabular", "regression", "19→1 (n=15k subsample)",
        "King County house sale prices — OpenML #42731. Grinsztajn et al. 2022. "
        "[R² reference: RF≈0.89, XGB≈0.91, MLP≈0.87]",
    ),
    "ctr23.brazilian_houses": (
        None, "tabular", "regression", "11→1 (n=10,692)",
        "Brazilian housing rental/sale price — OpenML #42688. Grinsztajn et al. 2022. "
        "[R² reference: RF≈0.96, XGB≈0.97, MLP≈0.95]",
    ),

    # ── Multi-output tabular ───────────────────────────────────────────────────
    "multioutput.scm20d": (
        run_multioutput_scm20d, "tabular", "regression", "61→16 (n=9,803)",
        "SCM20d supply-chain management multi-output regression [requires internet]",
    ),

    # ── Image (H×W×C tensor → label) ─────────────────────────────────────────
    "vision.mnist": (
        run_vision_mnist, "image", "classification", "784→10 (n=70k)",
        "MNIST digit recognition (LeCun et al. 1998) — top-1 accuracy "
        "[LeNet-5 ref: 99.2%, requires torchvision]",
    ),
    "vision.cifar10": (
        run_vision_cifar10, "image", "classification", "3072→10 (n=60k)",
        "CIFAR-10 image classification (Krizhevsky 2009) — top-1 accuracy "
        "[ResNet-20 ref: 91.3%, requires torchvision]",
    ),

    # ── Sequence / windowed prediction (tabular interface) ────────────────────
    "sequence.lorenz63": (
        run_sequence_lorenz63, "tabular", "regression", "60→60 (n=1,200)",
        "Lorenz-63 RK-4 short-horizon prediction — windowed 20-step input/output, "
        "inline ODE solver (no download)",
    ),

    # ── Field (discretised function/field → function/field) ───────────────────
    "pde.burgers_1d": (
        run_pde_burgers_1d, "field", "regression", "64→64 (n=1,024)",
        "Viscous Burgers 1D operator learning — inline FD solver (n_x=64, ν=0.01)",
    ),
    "pdebench.burgers_1d": (
        run_pdebench_burgers_1d, "field", "regression", "1024→1024 (n=9,000)",
        "PDEBench 1D Burgers ν=0.01 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.darcy_2d": (
        run_pdebench_darcy_2d, "field", "regression", "128×128→128×128 (n=10,000)",
        "PDEBench 2D Darcy Flow β=1.0 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.shallow_water_2d": (
        run_pdebench_shallow_water_2d, "field", "regression", "128×128→128×128 (n=1,000)",
        "PDEBench 2D Shallow Water Equations (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "thewell.gray_scott": (
        run_thewell_gray_scott, "field", "regression", "64×64×2→64×64×2",
        "TheWell Gray-Scott reaction-diffusion (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.turbulence_2d": (
        run_thewell_turbulence_2d, "field", "regression", "64×64×4→64×64×4",
        "TheWell 2D homogeneous turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.mhd": (
        run_thewell_mhd, "field", "regression", "64³×8→64³×8",
        "TheWell 3D MHD turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),

    # ── Plasma / fusion (DOE-relevant) ────────────────────────────────────────
    "fusion.m3dc1_sample": (
        run_fusion_m3dc1_sample, "plasma", "regression", "13→1 (n=2,000)",
        "M3DC1 equilibrium surrogate — 13 MHD params → stability metric",
    ),
    "plasma.cmod_density_limit": (
        None, "plasma", "classification", "6→2 (n=264,385)",
        "Alcator C-Mod density limit disruption — 6 plasma signals → binary label. "
        "MIT-PSFC open_density_limit_database. Greenwald limit physics. "
        "[MIT-PSFC/open_density_limit_database, GitHub]",
    ),
    "plasma.qlknn_transport": (
        None, "plasma", "regression", "10→1 (n≈7,475 post mask)",
        "QuaLiKiz/QLKNN turbulent electron heat flux surrogate. "
        "Inputs: Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni "
        "(normalised gyrokinetic parameters). "
        "Output: efeITG (ITG electron heat flux, gyroBohm units). "
        "van de Plassche et al. Nuclear Fusion 2020. "
        "[requires: pip install fusion_surrogates]",
    ),
    "plasma.qlknn10d": (
        None, "plasma", "regression", "9→1 (2.4M subsample of 290M)",
        "Train on the ACTUAL QuaLiKiz gyrokinetic fluxes behind QLKNN: "
        "predict the ITG leading flux (efiITG, GB units) from the 9-D "
        "input grid (Zeff, Ati, Ate, An, q, smag, x, Ti_Te, Nustar). "
        "van de Plassche et al. PoP 27, 022310 (2020); Zenodo "
        "10.5281/zenodo.3497066 (CC-BY 4.0). [13.3 GB open download]",
    ),
    "plasma.constellaration": (
        None, "plasma", "regression", "90→1 (10k subsample)",
        "ConStellaration boundary → log₁₀(qi). Subsamples 10k from paper-filtered cache (26,897 rows). "
        "[HF: proxima-fusion/constellaration, pip install datasets]",
    ),
    "plasma.constellaration_paper": (
        None, "plasma", "regression", "90→12 (n=26,897)",
        "Paper protocol (arXiv:2506.19583 §A.4): 12 metrics, one model each, mean test R². "
        "[HF: proxima-fusion/constellaration, pip install datasets]",
    ),
    "plasma.constellaration_multioutput": (
        None, "plasma", "regression", "90→12 (n=26,897)",
        "Single residual-MLP-style multi-output model: 90 boundary coeffs → 12 metrics. "
        "Same filtered cache as constellaration_paper. "
        "[HF: proxima-fusion/constellaration, pip install datasets]",
    ),
}

# ─── Short-name aliases ───────────────────────────────────────────────────────
# Lets users type ``surge run -b cifar10`` instead of ``vision.cifar10``.
# Values must be valid keys in _META above.
_SHORT_ALIASES: dict[str, str] = {
    # vision
    "cifar10":              "vision.cifar10",
    "mnist":                "vision.mnist",
    # tabular legacy
    "diabetes":             "tabular.diabetes",
    "california":           "tabular.california_housing",
    "california_housing":   "tabular.california_housing",
    "concrete":             "tabular.concrete_strength",
    "energy":               "tabular.energy_efficiency",
    "airfoil":              "tabular.airfoil_noise",
    "yacht":                "tabular.yacht_dynamics",
    "superconductor":       "tabular.superconductor",
    "iris":                 "tabular.iris",
    "breast_cancer":        "tabular.breast_cancer",
    "wine":                 "tabular.wine",
    "digits":               "tabular.digits",
    "covertype":            "tabular.covertype",
    "plasma_stability":     "tabular.plasma_stability",
    "classification.plasma_stability": "tabular.plasma_stability",
    # CTR-23
    "abalone":              "ctr23.abalone",
    "bike_sharing":         "ctr23.bike_sharing",
    "diamonds":             "ctr23.diamonds",
    "house_sales":          "ctr23.house_sales",
    "brazilian_houses":     "ctr23.brazilian_houses",
    # PDE
    "burgers":              "pde.burgers_1d",
    "burgers_1d":           "pde.burgers_1d",
    "darcy":                "pdebench.darcy_2d",
    "darcy_2d":             "pdebench.darcy_2d",
    "shallow_water":        "pdebench.shallow_water_2d",
    # sequence
    "lorenz":               "sequence.lorenz63",
    "lorenz63":             "sequence.lorenz63",
    # plasma
    "qlknn":                "plasma.qlknn_transport",
    "qlknn10d":             "plasma.qlknn10d",
    "cmod":                 "plasma.cmod_density_limit",
    "constellaration":      "plasma.constellaration",
    "constellaration_paper":"plasma.constellaration_paper",
    "constellaration_multi":"plasma.constellaration_multioutput",
    "constellaration_mimo":"plasma.constellaration_multioutput",
    # multi-output
    "scm20d":               "multioutput.scm20d",
}

_DATASET_METADATA_OVERRIDES: dict[str, dict[str, Any]] = {
    "tabular.diabetes": {
        "dataset_source": "sklearn.datasets",
        "license": "scikit-learn bundled dataset",
        "access": "bundled",
        "cache_path": None,
        "sample_count": 442,
        "feature_shape": "10",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu", "memory_tier": "small", "optional_dependencies": []},
    },
    "tabular.california_housing": {
        "dataset_source": "sklearn.datasets",
        "license": "sklearn/OpenML source dataset",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/tabular/sklearn_cache",
        "sample_count": 20640,
        "feature_shape": "8",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu", "memory_tier": "medium", "optional_dependencies": []},
    },
    "tabular.concrete_strength": {
        "dataset_source": "OpenML #4353",
        "license": "OpenML dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/tabular/sklearn_cache",
        "sample_count": 1030,
        "feature_shape": "8",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu", "memory_tier": "small", "optional_dependencies": []},
    },
    "classification.flow_regime": {
        "dataset_source": "SURGE inline generator",
        "license": "project generated fixture",
        "access": "generated",
        "cache_path": None,
        "sample_count": 800,
        "feature_shape": "3",
        "target_shape": "4 classes",
        "resource_expectation": {"device": "cpu", "memory_tier": "small", "optional_dependencies": []},
    },
    "sequence.lorenz63": {
        "dataset_source": "SURGE RK4 Lorenz-63 generator",
        "license": "project generated fixture",
        "access": "generated_cached",
        "cache_path": "data/datasets/benchmarks/sequence/lorenz63.npz",
        "sample_count": 1200,
        "feature_shape": "60",
        "target_shape": "60",
        "resource_expectation": {"device": "cpu", "memory_tier": "small", "optional_dependencies": []},
    },
    "pde.burgers_1d": {
        "dataset_source": "SURGE finite-difference Burgers generator",
        "license": "project generated fixture",
        "access": "generated_cached",
        "cache_path": "data/datasets/benchmarks/pde/burgers_1d.npz",
        "sample_count": 1024,
        "feature_shape": "64",
        "target_shape": "64",
        "resource_expectation": {"device": "cpu/gpu", "memory_tier": "medium", "optional_dependencies": ["torch"]},
    },
    "classification.plasma_stability": {
        "dataset_source": "UCI Electrical Grid Stability",
        "license": "UCI dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/classification/plasma_stability.npz",
        "sample_count": 10000,
        "feature_shape": "12",
        "target_shape": "2 classes",
        "resource_expectation": {"device": "cpu", "memory_tier": "small", "optional_dependencies": []},
    },
    "tabular.plasma_stability": {
        "dataset_source": "UCI Electrical Grid Stability",
        "license": "UCI dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/classification/plasma_stability.npz",
        "sample_count": 10000,
        "feature_shape": "12",
        "target_shape": "2 classes",
        "resource_expectation": {"device": "cpu", "memory_tier": "small", "optional_dependencies": []},
    },
    "plasma.cmod_density_limit": {
        "dataset_source": "MIT-PSFC/open_density_limit_database",
        "license": "upstream GitHub repository terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/plasma/cmod_density_limit.npz",
        "sample_count": 264385,
        "feature_shape": "6",
        "target_shape": "2 classes",
        "resource_expectation": {"device": "cpu", "memory_tier": "medium", "optional_dependencies": []},
    },
    "plasma.qlknn10d": {
        "dataset_source": "QLKNN10D Zenodo 10.5281/zenodo.3497066",
        "license": "CC-BY 4.0",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/plasma/qlknn10d_sub.npz",
        "sample_count": 2400000,
        "feature_shape": "9",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu/gpu", "memory_tier": "large", "optional_dependencies": ["h5py"]},
    },
    "plasma.qlknn_transport": {
        "dataset_source": "Google DeepMind fusion_surrogates QLKNN",
        "license": "upstream package/model terms",
        "access": "generated_cached",
        "cache_path": "data/datasets/benchmarks/plasma/qlknn_transport.npz",
        "sample_count": 7475,
        "feature_shape": "10",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu/gpu", "memory_tier": "medium", "optional_dependencies": ["fusion_surrogates", "jax"]},
    },
    "fusion.m3dc1_sample": {
        "dataset_source": "M3DC1 HDF5 supplied out-of-band",
        "license": "data-owner controlled",
        "access": "local_file",
        "cache_path": "data/datasets/benchmarks/fusion/m3dc1/m3dc1_sample.hdf5",
        "sample_count": None,
        "feature_shape": "13",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu", "memory_tier": "medium", "optional_dependencies": ["h5py"]},
    },
    "plasma.constellaration": {
        "dataset_source": "proxima-fusion/constellaration on HuggingFace",
        "license": "upstream dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/plasma/constellaration",
        "sample_count": 10000,
        "feature_shape": "90",
        "target_shape": "1",
        "resource_expectation": {"device": "cpu/gpu", "memory_tier": "medium", "optional_dependencies": ["datasets"]},
    },
    "plasma.constellaration_paper": {
        "dataset_source": "proxima-fusion/constellaration on HuggingFace",
        "license": "upstream dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/plasma/constellaration",
        "sample_count": 26897,
        "feature_shape": "90",
        "target_shape": "12",
        "resource_expectation": {"device": "cpu/gpu", "memory_tier": "medium", "optional_dependencies": ["datasets"]},
    },
    "plasma.constellaration_multioutput": {
        "dataset_source": "proxima-fusion/constellaration on HuggingFace",
        "license": "upstream dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/constellaration",
        "sample_count": 26897,
        "feature_shape": "90",
        "target_shape": "12 (joint model)",
        "resource_expectation": {"device": "cpu/gpu", "memory_tier": "medium", "optional_dependencies": ["datasets"]},
    },
    "vision.mnist": {
        "dataset_source": "torchvision MNIST",
        "license": "upstream dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/vision",
        "sample_count": 70000,
        "feature_shape": "784",
        "target_shape": "10 classes",
        "resource_expectation": {"device": "gpu recommended", "memory_tier": "medium", "optional_dependencies": ["torch", "torchvision"]},
    },
    "vision.cifar10": {
        "dataset_source": "torchvision CIFAR-10",
        "license": "upstream dataset terms",
        "access": "download_cached",
        "cache_path": "data/datasets/benchmarks/vision",
        "sample_count": 60000,
        "feature_shape": "3072",
        "target_shape": "10 classes",
        "resource_expectation": {"device": "gpu recommended", "memory_tier": "medium", "optional_dependencies": ["torch", "torchvision"]},
    },
}


def benchmark_metadata(key: str) -> dict[str, Any]:
    """Return structured dataset and resource metadata for a benchmark."""
    key = resolve_benchmark_key(key)
    _, category, task_type, shape, _ = _META[key]
    default = {
        "dataset_source": "OpenML/sklearn or SURGE benchmark registry",
        "license": "upstream dataset terms",
        "access": "download_cached" if category != "smoke" else "generated",
        "cache_path": "data/datasets/benchmarks/tabular/sklearn_cache" if category == "tabular" else None,
        "task_family": category,
        "task_type": task_type,
        "sample_count": None,
        "feature_shape": shape.split("(")[0].strip(),
        "target_shape": "",
        "resource_expectation": {
            "device": "cpu",
            "memory_tier": "small",
            "optional_dependencies": [],
        },
    }
    merged = {**default, **_DATASET_METADATA_OVERRIDES.get(key, {})}
    merged["task_family"] = category
    merged["task_type"] = task_type
    return merged


def _legacy_tier_for(category: str, key: str | None = None) -> str:
    """Return the historical numeric tier while category remains primary."""
    if key == "sequence.lorenz63":
        return "0"
    if category == "smoke":
        return "0"
    if category == "tabular":
        return "1"
    if category in {"image", "field", "plasma"}:
        return "2"
    return category


# Flat runner registry (key → callable) — None entries use the generic adapter path.
REGISTRY: dict[str, Callable[..., BenchmarkResult]] = {
    k: v[0] for k, v in _META.items() if v[0] is not None
}


def resolve_benchmark_key(key: str) -> str:
    """Resolve a short alias or full key to a canonical registry key.

    Examples
    --------
    >>> resolve_benchmark_key("cifar10")
    'vision.cifar10'
    >>> resolve_benchmark_key("vision.cifar10")
    'vision.cifar10'
    """
    if key in _META:
        return key
    if key in _SHORT_ALIASES:
        return _SHORT_ALIASES[key]
    raise KeyError(
        f"Unknown benchmark {key!r}. "
        f"Run 'surge run --list' to see all keys and short aliases."
    )


def list_benchmarks(
    *,
    category: str | None = None,
    task_type: str | None = None,
    include_smoke: bool = True,
    tier: str | None = None,
) -> list[str]:
    """Return sorted benchmark keys, optionally filtered.

    Parameters
    ----------
    category:
        One of ``tabular``, ``image``, ``field``, ``plasma``, or ``smoke``.
        Legacy aliases are accepted: ``vision`` → ``image``, ``pde`` → ``field``,
        ``sequence`` → ``tabular``, ``multioutput`` → ``tabular``.
    task_type:
        ``regression`` or ``classification``.
    include_smoke:
        If *False*, synthetic smoke-test benchmarks are excluded. The CLI
        passes this explicitly so ``surge run --list`` can stay quiet by
        default while the Python API remains exhaustive.
    """
    _CAT_ALIASES = {
        "vision": "image",       # legacy
        "pde": "field",          # legacy
        "sequence": "tabular",   # legacy
        "multioutput": "tabular", # legacy
    }
    effective_category = _CAT_ALIASES.get(category, category) if category else None

    keys = []
    for k, (_, cat, tt, _, _) in _META.items():
        if not include_smoke and cat == "smoke":
            continue
        if effective_category is not None and cat != effective_category:
            continue
        if tier is not None and _legacy_tier_for(cat, k) != str(tier):
            continue
        if task_type is not None and tt != task_type:
            continue
        keys.append(k)
    return sorted(keys)


def list_categories() -> list[str]:
    """Return the distinct categories present in the registry."""
    return sorted({cat for _, cat, *_ in _META.values() if cat != "smoke"})


def benchmark_info(key: str) -> dict[str, Any]:
    """Return metadata dict for a registered benchmark key (full or short)."""
    key = resolve_benchmark_key(key)
    _, category, task_type, shape, description = _META[key]
    metadata = benchmark_metadata(key)
    return {
        "key": key,
        "category": category,
        # keep numeric 'tier' for older code/tests while category is primary
        "tier": _legacy_tier_for(category, key),
        "task_type": task_type,
        "shape": shape,
        "description": description,
        **metadata,
    }


def run_benchmark(key: str, **kwargs: Any) -> BenchmarkResult:
    """
    Run a registered benchmark by key (full or short alias).

    Parameters
    ----------
    key:
        e.g. ``tabular.iris``, ``cifar10``, ``qlknn``.
    **kwargs:
        Passed to the underlying task (e.g. ``seed=``, ``model_key=``).
    """
    key = resolve_benchmark_key(key)

    # Fail fast on a regressor asked to classify (or vice versa): otherwise
    # the model trains to completion and dies inside the metric computation
    # with an opaque sklearn error.
    requested_model = kwargs.get("model_key")
    if requested_model:
        from .leaderboard import _task_mismatch

        mismatch = _task_mismatch(requested_model,
                                  benchmark_info(key)["task_type"])
        if mismatch is not None:
            raise ValueError(mismatch)

    runner = REGISTRY.get(key)
    if runner is None:
        # Generic adapter-based path (CTR-23, plasma, vision via leaderboard).
        from .leaderboard import _run_with_adapter
        from surge.model.registry import MODEL_REGISTRY

        model_key = kwargs.pop("model_key", "sklearn.ridge")
        seed = kwargs.pop("seed", 42)
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Model {model_key!r} not registered.")
        adapter = MODEL_REGISTRY.create(model_key)
        result = _run_with_adapter(key, adapter, seed=seed)
        if result is None:
            raise RuntimeError(f"Benchmark {key!r} with model {model_key!r} failed.")
        return result
    return runner(**kwargs)
