"""GPflow model adapters."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..hpc import ResourceProfile
from .base import BaseModelAdapter

if TYPE_CHECKING:
    from .gpflow_impl import GPflowGPRModel, GPflowMultiKernelGPR


def gpflow_runtime_available() -> bool:
    """True only if GPflow/TensorFlow initialized in :mod:`surge.model.gpflow_impl`."""
    from . import gpflow_impl as impl

    return impl.GPFLOW_AVAILABLE


def __getattr__(name: str):  # PEP 562: lazy ``GPFLOW_AVAILABLE``
    if name == "GPFLOW_AVAILABLE":
        return gpflow_runtime_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        from . import gpflow_impl as impl

        if not impl.GPFLOW_AVAILABLE:
            raise ImportError("GPflow not available. Install gpflow to enable GP adapters")
        super().__init__(**kwargs)

    def _build_model(self, **kwargs: Any) -> Any:
        from .gpflow_impl import GPflowGPRModel

        return GPflowGPRModel(**kwargs)

    def predict_with_uncertainty(self, X: Any) -> Any:
        return self._model.predict_with_uncertainty(X)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        return self._model.sample_posterior(X, num_samples=num_samples)


class GPflowMultiKernelAdapter(GPflowGPRAdapter):
    """Adapter for multi-kernel GPflow regression."""

    name = "gpflow.multi_kernel"

    def _build_model(self, **kwargs: Any) -> Any:
        from .gpflow_impl import GPflowMultiKernelGPR

        return GPflowMultiKernelGPR(**kwargs)
