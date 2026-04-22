"""GPflow model adapters."""
from __future__ import annotations

from typing import Any

from ..hpc import ResourceProfile
from .base import BaseModelAdapter
from .gpflow_impl import GPFLOW_AVAILABLE, GPflowGPRModel, GPflowMultiKernelGPR


_GPFLOW_PROFILE = ResourceProfile(
    name="gpflow.gpr",
    supports_cpu=True,
    supports_gpu=True,  # TF handles device placement; single-device in v0.1.0
    worker_semantics="none",
    notes="GPflow follows TensorFlow's device placement; workers unused.",
)


class GPflowGPRAdapter(BaseModelAdapter):
    """Adapter around :class:`GPflowGPRModel`."""

    name = "gpflow.gpr"
    backend = "gpflow"
    resource_profile = _GPFLOW_PROFILE

    def __init__(self, **kwargs: Any) -> None:
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow not available. Install gpflow to enable GP adapters")
        super().__init__(**kwargs)

    def _build_model(self, **kwargs: Any) -> GPflowGPRModel:
        return GPflowGPRModel(**kwargs)

    def predict_with_uncertainty(self, X: Any) -> Any:
        return self._model.predict_with_uncertainty(X)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        return self._model.sample_posterior(X, num_samples=num_samples)


class GPflowMultiKernelAdapter(GPflowGPRAdapter):
    """Adapter for multi-kernel GPflow regression."""

    name = "gpflow.multi_kernel"

    def _build_model(self, **kwargs: Any) -> GPflowMultiKernelGPR:
        return GPflowMultiKernelGPR(**kwargs)
