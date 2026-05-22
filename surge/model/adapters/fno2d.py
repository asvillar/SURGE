"""Adapter registering ``pytorch.fno2d``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_FNO2D_INFO = ModelInfo(
    architecture=(
        "2D Fourier Neural Operator (FNO-2D): learns mappings between function "
        "spaces on 2D grids by applying a learnable kernel in Fourier space. "
        "The forward pass alternates between (1) a pointwise linear layer and "
        "(2) a global spectral convolution that truncates to the lowest n_modes "
        "Fourier modes in each spatial dimension. This gives O(N log N) complexity "
        "and is mesh-invariant — the same model can be evaluated on different "
        "grid resolutions at inference time."
    ),
    use_cases=[
        "2D PDE operator learning: Darcy Flow, Navier-Stokes, shallow water equations",
        "Field-to-field regression where input/output are 2D spatial grids",
        "Scientific surrogates replacing expensive FEM/FVM solvers on regular meshes",
    ],
    not_for=[
        "Tabular or 1D regression — use pytorch.mlp or pytorch.fno1d",
        "Unstructured meshes — FNO requires a regular grid",
    ],
    strengths=[
        "State-of-the-art on PDEBench Darcy2D and Navier-Stokes benchmarks",
        "Mesh-invariant: train on 64×64, evaluate on 128×128",
        "Global receptive field via spectral convolution — no boundary artefacts",
    ],
    weaknesses=[
        "Requires regular grids — not applicable to finite-element meshes",
        "Memory scales quadratically with grid size (FFT dominates beyond ~256×256)",
        "n_modes hyperparameter sensitive: too few → underfitting, too many → overfitting",
    ],
    references=[
        "Li et al. (2021) 'Fourier Neural Operator for Parametric PDEs' "
        "ICLR 2021. https://arxiv.org/abs/2010.08895",
        "Takamoto et al. (2022) 'PDEBench: An Extensive Benchmark for Scientific "
        "Machine Learning' NeurIPS 2022. https://arxiv.org/abs/2210.07182",
    ],
)

_FNO2D_PROFILE = ResourceProfile(
    name="pytorch.fno2d",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="2D Fourier Neural Operator for 2D PDE operator learning. Li et al. ICLR 2021.",
)


class FNO2dAdapter(BaseModelAdapter):
    """FNO2d operator learning surrogate for 2D fields."""

    name = "pytorch.fno2d"
    backend = "pytorch"
    task_type = "regression"
    uses_internal_preprocessing = True
    resource_profile = _FNO2D_PROFILE
    _INFO = _FNO2D_INFO
    default_params: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": 32,
        "n_modes": 12,
        "n_layers": 4,
        "n_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.fno2d")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.FNO2dModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
