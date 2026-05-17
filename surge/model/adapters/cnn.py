"""Adapter registering ``pytorch.cnn1d`` in the SURGE model registry."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_CNN1D_PROFILE = ResourceProfile(
    name="pytorch.cnn1d",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Dilated 1-D CNN surrogate for spatial fields and sequences.",
)


class CNN1DAdapter(BaseModelAdapter):
    """Dilated 1-D CNN surrogate for spatial / sequence data."""

    name = "pytorch.cnn1d"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _CNN1D_PROFILE
    task_type = "regression"

    default_params: dict[str, Any] = {
        "hidden_channels": 64,
        "n_layers": 4,
        "kernel_size": 5,
        "dropout": 0.05,
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.cnn import CNN1DModel

        params = dict(self.default_params)
        params.update(kwargs)
        return CNN1DModel(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        self._model.fit(X, y, X_val, y_val)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
