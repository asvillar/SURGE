"""Dispatch table for standard benchmarks."""

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
# Each entry: (runner_fn, tier, task_type, shape_desc, description)
_META: dict[str, tuple[Callable, str, str, str, str]] = {
    # ── Synthetic / inline (Tier 0) ──────────────────────────────────────────
    "synthetic.regression_1d": (
        run_synthetic_regression_1d, "0", "regression", "1→1 (n=400)",
        "Linear 1-D signal with Gaussian noise (inline fixture)",
    ),
    "synthetic.multioutput_2d": (
        run_synthetic_multioutput_2d, "0", "regression", "8→2 (n=600)",
        "Multi-output 8→2 linear regression with Gaussian noise (inline fixture)",
    ),
    "synthetic.classification_binary": (
        run_synthetic_classification_binary, "0", "classification", "20→2 (n=500)",
        "Binary labels from linear combo of features (inline fixture)",
    ),
    "sequence.lorenz63": (
        run_sequence_lorenz63, "0", "regression", "60→60 (n=1200)",
        "Lorenz-63 RK-4 short-horizon prediction (inline, no download)",
    ),
    "classification.flow_regime": (
        run_classification_flow_regime, "0", "classification", "3→4 (n=800)",
        "CFD flow regime 4-class labeling from Mach/Re/AoA (inline fixture, no download)",
    ),
    # ── UCI / sklearn tabular (Tier 1) ───────────────────────────────────────
    "tabular.diabetes": (
        run_tabular_diabetes, "1", "regression", "10→1 (n=442)",
        "UCI Diabetes / sklearn.datasets (Efron et al. 2004)",
    ),
    "tabular.california_housing": (
        run_tabular_california_housing, "1", "regression", "8→1 (n=20,640)",
        "California Housing / sklearn.datasets (Pace & Barry 1997)",
    ),
    "tabular.concrete_strength": (
        run_tabular_concrete_strength, "1", "regression", "8→1 (n=1,030)",
        "UCI Concrete Compressive Strength (Yeh 1998) [requires internet on first run]",
    ),
    "tabular.energy_efficiency": (
        run_tabular_energy_efficiency, "1", "regression", "8→1 (n=768)",
        "UCI Energy Efficiency — Heating Load (Tsanas & Xifara 2012) [requires internet on first run]",
    ),
    "tabular.airfoil_noise": (
        run_tabular_airfoil_noise, "1", "regression", "5→1 (n=1,503)",
        "NASA Airfoil Self-Noise (Brooks et al. 1989) — UCI [requires internet on first run]",
    ),
    "tabular.yacht_dynamics": (
        run_tabular_yacht_dynamics, "1", "regression", "6→1 (n=308)",
        "UCI Yacht Hydrodynamics (Gerritsma 1981) [requires internet on first run]",
    ),
    "tabular.superconductor": (
        run_tabular_superconductor, "1", "regression", "81→1 (n=21,263)",
        "Superconductor Tc prediction (Hamidieh 2018) [requires internet]",
    ),
    "tabular.iris": (
        run_tabular_iris, "1", "classification", "4→3 (n=150)",
        "UCI Iris / sklearn.datasets (Fisher 1936)",
    ),
    "tabular.breast_cancer": (
        run_tabular_breast_cancer, "1", "classification", "30→2 (n=569)",
        "Wisconsin Breast Cancer / sklearn.datasets (UCI WDBC)",
    ),
    "tabular.wine": (
        run_tabular_wine, "1", "classification", "13→3 (n=178)",
        "UCI Wine / sklearn.datasets",
    ),
    "tabular.digits": (
        run_tabular_digits, "1", "classification", "64→10 (n=1,797)",
        "Optical digits / sklearn.datasets (Alpaydin 1998)",
    ),
    "classification.covertype": (
        run_classification_covertype, "1", "classification", "54→7 (n=20k subsample)",
        "Forest Covertype 7-class classification (Blackard & Dean 1999) [requires internet]",
    ),
    "classification.plasma_stability": (
        run_classification_plasma_stability, "2", "classification", "12→2 (n=10,000)",
        "UCI Electrical Grid Stability (Arzamasov 2018) [requires internet on first run]",
    ),
    "multioutput.scm20d": (
        run_multioutput_scm20d, "1", "regression", "61→20 (n=9,803)",
        "SCM20d supply-chain management multi-output regression [requires internet]",
    ),
    # ── PDE / sequence (Tier 1–3) ────────────────────────────────────────────
    "pde.burgers_1d": (
        run_pde_burgers_1d, "1", "regression", "64→64 (n=1,024)",
        "Viscous Burgers 1D operator learning — inline FD solver (n_x=64, ν=0.01)",
    ),
    "pdebench.burgers_1d": (
        run_pdebench_burgers_1d, "3", "regression", "1024→1024 (n=9,000)",
        "PDEBench 1D Burgers ν=0.01 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.darcy_2d": (
        run_pdebench_darcy_2d, "3", "regression", "128×128→128×128 (n=10,000)",
        "PDEBench 2D Darcy Flow β=1.0 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.shallow_water_2d": (
        run_pdebench_shallow_water_2d, "3", "regression", "128×128→128×128 (n=1,000)",
        "PDEBench 2D Shallow Water Equations (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    # ── Vision (Tier 2) ──────────────────────────────────────────────────────
    "vision.mnist": (
        run_vision_mnist, "2", "classification", "784→10 (n=60k train)",
        "MNIST digit recognition (LeCun et al. 1998) — top-1 accuracy [requires torchvision]",
    ),
    "vision.cifar10": (
        run_vision_cifar10, "2", "classification", "3072→10 (n=50k train)",
        "CIFAR-10 image classification (Krizhevsky 2009) — top-1 accuracy [requires torchvision]",
    ),
    # ── Fusion / plasma (Tier 2–4) ───────────────────────────────────────────
    "fusion.mhd_sample": (
        run_fusion_mhd_sample, "2", "regression", "13→1 (n=2,000)",
        "MHD-code equilibrium surrogate (13 MHD params → stability metric, R²)",
    ),
    "thewell.gray_scott": (
        run_thewell_gray_scott, "4", "regression", "64×64×2→64×64×2 (15TB total)",
        "TheWell Gray-Scott reaction-diffusion (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.turbulence_2d": (
        run_thewell_turbulence_2d, "4", "regression", "64×64×4→64×64×4 (15TB total)",
        "TheWell 2D homogeneous turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.mhd": (
        run_thewell_mhd, "4", "regression", "64³×8→64³×8 (15TB total)",
        "TheWell 3D MHD turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    # ── CTR-23 (Grinsztajn et al. 2022, arXiv:2207.08815) ───────────────────
    "ctr23.abalone": (
        None, "2", "regression", "8→1 (n=4,177)",
        "Abalone age prediction — UCI/OpenML #183. Grinsztajn et al. 2022.",
    ),
    "ctr23.bike_sharing": (
        None, "2", "regression", "12→1 (n=17,389)",
        "Hourly bike rentals — OpenML #42712. Grinsztajn et al. 2022.",
    ),
    "ctr23.diamonds": (
        None, "2", "regression", "9→1 (n=15k subsample)",
        "Diamond price prediction — OpenML #42225. Grinsztajn et al. 2022.",
    ),
    "ctr23.house_sales": (
        None, "2", "regression", "19→1 (n=15k subsample)",
        "King County house sale prices — OpenML #42731. Grinsztajn et al. 2022.",
    ),
    "ctr23.brazilian_houses": (
        None, "2", "regression", "11→1 (n=10,692)",
        "Brazilian housing rental/sale price — OpenML #42688. Grinsztajn et al. 2022.",
    ),
    # ── DOE Fusion plasma datasets (Tier 2–3) ────────────────────────────────
    "plasma.cmod_density_limit": (
        None, "2", "classification", "6→2 (n=264,385)",
        "Alcator C-Mod density limit disruption — 6 plasma signals → binary label. "
        "MIT-PSFC open_density_limit_database. Greenwald limit physics. "
        "[MIT-PSFC/open_density_limit_database, GitHub]",
    ),
    "plasma.qlknn_transport": (
        None, "2", "regression", "10→1 (n=20,000)",
        "QuaLiKiz/QLKNN turbulent electron heat flux surrogate — 10 gyrokinetic params → efeITG. "
        "Reference: Google DeepMind fusion_surrogates (van de Plassche et al. NF 2020). "
        "[requires: pip install fusion_surrogates]",
    ),
    "plasma.constellaration": (
        None, "3", "regression", "90→1 (n=10k subsample)",
        "ConStellaration: stellarator boundary shape → quasi-isodynamic quality (QI). "
        "182k QI-like VMEC equilibria, Proxima Fusion 2025. "
        "[proxima-fusion/constellaration on HuggingFace, requires: pip install datasets]",
    ),
    "plasma.constellaration_paper": (
        None, "3", "regression", "90→12 (n≈23k optimised, paper protocol)",
        "ConStellaration 12-metric per-metric evaluation (arXiv:2506.19583 Appendix A.4). "
        "DESC/VMEC-optimised 3-field-period configs; one model per metric; paper baseline R²>0.97. "
        "[Goodman et al. 2025, proxima-fusion/constellaration on HuggingFace]",
    ),
}

# Flat runner registry (key → callable) — None entries use the generic adapter path.
REGISTRY: dict[str, Callable[..., BenchmarkResult]] = {
    k: v[0] for k, v in _META.items() if v[0] is not None
}


def list_benchmarks(*, tier: str | None = None, task_type: str | None = None) -> list[str]:
    """Return sorted benchmark keys, optionally filtered by tier or task_type."""
    keys = []
    for k, (_, t, tt, _, _) in _META.items():
        if tier is not None and t != tier:
            continue
        if task_type is not None and tt != task_type:
            continue
        keys.append(k)
    return sorted(keys)


def benchmark_info(key: str) -> dict[str, str]:
    """Return metadata dict for a registered benchmark key."""
    if key not in _META:
        raise KeyError(f"Unknown benchmark {key!r}. Use list_benchmarks().")
    _, tier, task_type, shape, description = _META[key]
    return {
        "key": key,
        "tier": tier,
        "task_type": task_type,
        "shape": shape,
        "description": description,
    }


def run_benchmark(key: str, **kwargs: Any) -> BenchmarkResult:
    """
    Run a registered benchmark by key.

    Parameters
    ----------
    key:
        e.g. ``synthetic.regression_1d``, ``tabular.iris``.
    **kwargs:
        Passed to the underlying task (e.g. ``seed=``, ``model_key=``).
    """
    if key not in _META:
        raise KeyError(
            f"Unknown benchmark {key!r}. Choose one of: {', '.join(list_benchmarks())}"
        )
    runner = REGISTRY.get(key)
    if runner is None:
        # Generic adapter-based path (used for CTR-23 and other loader-only benchmarks).
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
