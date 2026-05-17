"""Dispatch table for standard benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BenchmarkResult
from .tasks import (
    run_classification_flow_regime,
    run_classification_plasma_stability,
    run_fusion_mhd_sample,
    run_pde_burgers_1d,
    run_pdebench_burgers_1d,
    run_pdebench_darcy_2d,
    run_pdebench_shallow_water_2d,
    run_sequence_lorenz63,
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
    "synthetic.regression_1d": (
        run_synthetic_regression_1d, "0", "regression", "1→1",
        "Linear 1-D signal with Gaussian noise (inline fixture)",
    ),
    "synthetic.multioutput_2d": (
        run_synthetic_multioutput_2d, "0", "regression", "8→2",
        "Multi-output 8→2 linear regression with Gaussian noise (inline fixture)",
    ),
    "synthetic.classification_binary": (
        run_synthetic_classification_binary, "0", "classification", "20→2",
        "Binary labels from linear combo of features (inline fixture)",
    ),
    "tabular.diabetes": (
        run_tabular_diabetes, "1", "regression", "10→1",
        "UCI Diabetes / sklearn.datasets (Efron et al. 2004)",
    ),
    "tabular.california_housing": (
        run_tabular_california_housing, "1", "regression", "8→1",
        "California Housing / sklearn.datasets (Pace & Barry 1997)",
    ),
    "tabular.concrete_strength": (
        run_tabular_concrete_strength, "1", "regression", "8→1",
        "UCI Concrete Compressive Strength (Yeh 1998) [requires internet on first run]",
    ),
    "tabular.energy_efficiency": (
        run_tabular_energy_efficiency, "1", "regression", "8→1",
        "UCI Energy Efficiency — Heating Load (Tsanas & Xifara 2012) [requires internet on first run]",
    ),
    "tabular.iris": (
        run_tabular_iris, "1", "classification", "4→3",
        "UCI Iris / sklearn.datasets (Fisher 1936)",
    ),
    "tabular.breast_cancer": (
        run_tabular_breast_cancer, "1", "classification", "30→2",
        "Wisconsin Breast Cancer / sklearn.datasets (UCI WDBC)",
    ),
    "tabular.wine": (
        run_tabular_wine, "1", "classification", "13→3",
        "UCI Wine / sklearn.datasets",
    ),
    "tabular.digits": (
        run_tabular_digits, "1", "classification", "64→10",
        "Optical digits / sklearn.datasets (Alpaydin 1998)",
    ),
    "sequence.lorenz63": (
        run_sequence_lorenz63, "0", "regression", "60→60",
        "Lorenz-63 RK-4 short-horizon prediction (inline, no download)",
    ),
    "pde.burgers_1d": (
        run_pde_burgers_1d, "1", "regression", "64→64",
        "Viscous Burgers 1D operator learning — inline FD solver (n_x=64, ν=0.01)",
    ),
    "classification.flow_regime": (
        run_classification_flow_regime, "0", "classification", "3→4",
        "CFD flow regime 4-class labeling from Mach/Re/AoA (inline fixture, no download)",
    ),
    "tabular.airfoil_noise": (
        run_tabular_airfoil_noise, "1", "regression", "5→1",
        "NASA Airfoil Self-Noise (Brooks et al. 1989) — UCI [requires internet on first run]",
    ),
    "tabular.yacht_dynamics": (
        run_tabular_yacht_dynamics, "1", "regression", "6→1",
        "UCI Yacht Hydrodynamics (Gerritsma 1981) [requires internet on first run]",
    ),
    "classification.plasma_stability": (
        run_classification_plasma_stability, "2", "classification", "12→2",
        "UCI Electrical Grid Stability (Arzamasov 2018) [requires internet on first run]",
    ),
    "pdebench.burgers_1d": (
        run_pdebench_burgers_1d, "3", "regression", "1024→1024",
        "PDEBench 1D Burgers ν=0.01 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.darcy_2d": (
        run_pdebench_darcy_2d, "3", "regression", "16384→16384",
        "PDEBench 2D Darcy Flow β=1.0 (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "pdebench.shallow_water_2d": (
        run_pdebench_shallow_water_2d, "3", "regression", "16384→16384",
        "PDEBench 2D Shallow Water Equations (Takamoto et al. NeurIPS 2022) [requires HDF5 download]",
    ),
    "vision.mnist": (
        run_vision_mnist, "2", "classification", "784→10",
        "MNIST digit recognition (LeCun et al. 1998) — top-1 accuracy [requires torchvision]",
    ),
    "vision.cifar10": (
        run_vision_cifar10, "2", "classification", "3072→10",
        "CIFAR-10 image classification (Krizhevsky 2009) — top-1 accuracy [requires torchvision]",
    ),
    "fusion.mhd_sample": (
        run_fusion_mhd_sample, "2", "regression", "13→1",
        "MHD-code equilibrium surrogate (13 MHD params → stability metric, R²)",
    ),
    "thewell.gray_scott": (
        run_thewell_gray_scott, "4", "regression", "varies",
        "TheWell Gray-Scott reaction-diffusion (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.turbulence_2d": (
        run_thewell_turbulence_2d, "4", "regression", "varies",
        "TheWell 2D homogeneous turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
    "thewell.mhd": (
        run_thewell_mhd, "4", "regression", "varies",
        "TheWell 3D MHD turbulence (Ohana et al. NeurIPS 2024) [requires the-well pkg]",
    ),
}

# Flat runner registry (key → callable) used by run_benchmark().
REGISTRY: dict[str, Callable[..., BenchmarkResult]] = {k: v[0] for k, v in _META.items()}


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
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown benchmark {key!r}. Choose one of: {', '.join(list_benchmarks())}"
        )
    return REGISTRY[key](**kwargs)
