"""Adapter registering ``pytorch.fno1d`` in the SURGE model registry."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_FNO_PROFILE = ResourceProfile(
    name="pytorch.fno1d",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Fourier Neural Operator 1-D (Li et al. 2021).",
)


class FNO1dAdapter(BaseModelAdapter):
    """FNO1d operator learning adapter."""

    name = "pytorch.fno1d"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _FNO_PROFILE
    task_type = "regression"

    default_params: dict[str, Any] = {
        "hidden_channels": 64,
        "n_modes": 16,
        "n_layers": 4,
        "append_grid": True,
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.fno import FNO1dModel

        params = dict(self.default_params)
        params.update(kwargs)
        return FNO1dModel(**params)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
