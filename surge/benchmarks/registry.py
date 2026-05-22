"""Dispatch table for standard benchmarks.

Each benchmark has a **category** that describes its scientific purpose:

    smoke       — inline synthetic fixtures; used only for fast unit-level
                  CI smoke tests.  Never shown in --list by default.
    tabular     — real-world tabular regression / classification datasets.
                  UCI legacy datasets live here alongside modern CTR-23 sets.
    vision      — image classification (CIFAR-10, MNIST).
    pde         — PDE / operator-learning problems (Burgers, Darcy, …).
    plasma      — fusion / plasma-physics surrogates (QLKNN, ConStellaration, …).
    sequence    — time-series / forecasting (Lorenz-63).
    multioutput — multi-target regression.

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
    run_fusion_mhd_sample,
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
# category choices: smoke | tabular | vision | pde | plasma | sequence | multioutput
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
        run_multioutput_scm20d, "multioutput", "regression", "61→16 (n=9,803)",
        "SCM20d supply-chain management multi-output regression [requires internet]",
    ),

    # ── Vision ────────────────────────────────────────────────────────────────
    "vision.mnist": (
        run_vision_mnist, "vision", "classification", "784→10 (n=70k)",
        "MNIST digit recognition (LeCun et al. 1998) — top-1 accuracy "
        "[LeNet-5 ref: 99.2%, requires torchvision]",
    ),
    "vision.cifar10": (
        run_vision_cifar10, "vision", "classification", "3072→10 (n=60k)",
        "CIFAR-10 image classification (Krizhevsky 2009) — top-1 accuracy "
        "[ResNet-20 ref: 91.3%, requires torchvision]",
    ),

    # ── Sequence / time-series ────────────────────────────────────────────────
    "sequence.lorenz63": (
        run_sequence_lorenz63, "sequence", "regression", "60→60 (n=1,200)",
        "Lorenz-63 RK-4 short-horizon prediction (inline, no download)",
    ),

    # ── PDE / operator learning ───────────────────────────────────────────────
    "pde.burgers_1d": (
        run_pde_burgers_1d, "pde", "regression", "64→64 (n=1,024)",
        "Viscous Burgers 1D operator learning — inline FD solver (n_x=64, ν=0.01)",
    ),
    "pdebench.burgers_1d": (
        run_pdebench_burgers_1d, "pde", "regression", "1024→1024 (n=9,000)",
        "PDEBench 1D Burgers ν=0.01 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.darcy_2d": (
        run_pdebench_darcy_2d, "pde", "regression", "128×128→128×128 (n=10,000)",
        "PDEBench 2D Darcy Flow β=1.0 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.shallow_water_2d": (
        run_pdebench_shallow_water_2d, "pde", "regression", "128×128→128×128 (n=1,000)",
        "PDEBench 2D Shallow Water Equations (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "thewell.gray_scott": (
        run_thewell_gray_scott, "pde", "regression", "64×64×2→64×64×2",
        "TheWell Gray-Scott reaction-diffusion (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.turbulence_2d": (
        run_thewell_turbulence_2d, "pde", "regression", "64×64×4→64×64×4",
        "TheWell 2D homogeneous turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.mhd": (
        run_thewell_mhd, "pde", "regression", "64³×8→64³×8",
        "TheWell 3D MHD turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),

    # ── Plasma / fusion (DOE-relevant) ────────────────────────────────────────
    "fusion.mhd_sample": (
        run_fusion_mhd_sample, "plasma", "regression", "13→1 (n=2,000)",
        "MHD-code equilibrium surrogate — 13 MHD params → stability metric",
    ),
    "plasma.cmod_density_limit": (
        None, "plasma", "classification", "6→2 (n=264,385)",
        "Alcator C-Mod density limit disruption — 6 plasma signals → binary label. "
        "MIT-PSFC open_density_limit_database. Greenwald limit physics. "
        "[MIT-PSFC/open_density_limit_database, GitHub]",
    ),
    "plasma.qlknn_transport": (
        None, "plasma", "regression", "10→1 (n=20,000)",
        "QuaLiKiz/QLKNN turbulent electron heat flux surrogate — 10 gyrokinetic params → efeITG. "
        "van de Plassche et al. Nuclear Fusion 2020. "
        "[requires: pip install fusion_surrogates]",
    ),
    "plasma.constellaration": (
        None, "plasma", "regression", "90→1 (n=10k subsample)",
        "ConStellaration: stellarator boundary shape → quasi-isodynamic quality (QI). "
        "182k QI-like VMEC equilibria, Proxima Fusion 2025. "
        "[proxima-fusion/constellaration on HuggingFace, requires: pip install datasets]",
    ),
    "plasma.constellaration_paper": (
        None, "plasma", "regression", "90→12 (n≈23k optimised, paper protocol)",
        "ConStellaration 12-metric per-metric evaluation (Goodman et al. 2025, arXiv:2506.19583 §A.4). "
        "One model per metric; paper baseline R²>0.97. "
        "[proxima-fusion/constellaration on HuggingFace, requires: pip install datasets]",
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
    "cmod":                 "plasma.cmod_density_limit",
    "constellaration":      "plasma.constellaration",
    "constellaration_paper":"plasma.constellaration_paper",
    # multi-output
    "scm20d":               "multioutput.scm20d",
}

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
    include_smoke: bool = False,
    # legacy arg kept for back-compat; ignored silently
    tier: str | None = None,
) -> list[str]:
    """Return sorted benchmark keys, optionally filtered.

    Parameters
    ----------
    category:
        One of ``tabular``, ``vision``, ``pde``, ``plasma``, ``sequence``,
        ``multioutput``, or ``smoke``.
    task_type:
        ``regression`` or ``classification``.
    include_smoke:
        If *False* (default), synthetic smoke-test benchmarks are excluded.
        Pass *True* or ``--smoke`` from the CLI to include them.
    """
    keys = []
    for k, (_, cat, tt, _, _) in _META.items():
        if not include_smoke and cat == "smoke":
            continue
        if category is not None and cat != category:
            continue
        if task_type is not None and tt != task_type:
            continue
        keys.append(k)
    return sorted(keys)


def list_categories() -> list[str]:
    """Return the distinct categories present in the registry."""
    return sorted({cat for _, cat, *_ in _META.values() if cat != "smoke"})


def benchmark_info(key: str) -> dict[str, str]:
    """Return metadata dict for a registered benchmark key (full or short)."""
    key = resolve_benchmark_key(key)
    _, category, task_type, shape, description = _META[key]
    return {
        "key": key,
        "category": category,
        # keep 'tier' for any code that still reads it
        "tier": category,
        "task_type": task_type,
        "shape": shape,
        "description": description,
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
