"""Adapter registering ``pytorch.deeponet`` in the SURGE model registry."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_DNET_PROFILE = ResourceProfile(
    name="pytorch.deeponet",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Deep Operator Network (Lu et al. 2021).",
)


class DeepONetAdapter(BaseModelAdapter):
    """DeepONet operator learning adapter."""

    name = "pytorch.deeponet"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _DNET_PROFILE
    task_type = "regression"

    default_params: dict[str, Any] = {
        "n_basis": 64,
        "branch_width": 128,
        "trunk_width": 128,
        "n_hidden": 3,
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.deeponet import DeepONetModel

        params = dict(self.default_params)
        params.update(kwargs)
        return DeepONetModel(**params)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
