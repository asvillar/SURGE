"""Adapter registering ``pytorch.lenet5`` in the SURGE model registry."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_LENET_PROFILE = ResourceProfile(
    name="pytorch.lenet5",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="LeNet-5 for MNIST (LeCun 1998). Input: (N, 784) flat or (N, 1, 28, 28).",
)


class LeNet5Adapter(BaseModelAdapter):
    """LeNet-5 image classifier for MNIST-like tasks."""

    name = "pytorch.lenet5"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _LENET_PROFILE
    task_type = "classification"

    default_params: dict[str, Any] = {
        "n_classes": 10,
        "img_size": 28,
        "in_channels": 1,
        "dropout": 0.0,
        "n_epochs": 20,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "patience": 5,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.lenet import LeNet5Model

        params = dict(self.default_params)
        params.update(kwargs)
        return LeNet5Model(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        self._model.fit(X, y, X_val, y_val)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
