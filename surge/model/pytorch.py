"""PyTorch-based adapters and ensemble helpers."""
from __future__ import annotations

from typing import Any

from .base import BaseModelAdapter
from .pytorch_impl import PyTorchMLPModel

try:
    import torch  # noqa: F401
    PYTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYTORCH_AVAILABLE = False


class PyTorchMLPAdapter(BaseModelAdapter):
    """PyTorch-based MLP model with enhanced capabilities."""

    name = "pytorch.mlp"
    backend = "pytorch"
    uses_internal_preprocessing = True
    handles_output_scaling = True

    def __init__(self, **kwargs: Any) -> None:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch models not available. Install torch: pip install torch")
        super().__init__(**kwargs)

    def _initialize(self) -> None:
        # Defer actual model construction until fit so input/output sizes are known
        self._model = None

    def _build_model(self, **kwargs: Any) -> PyTorchMLPModel:
        return PyTorchMLPModel(**kwargs)

    def fit(self, X: Any, y: Any) -> Any:
        if self._model is None:
            self._model = self._build_model(**self.params)
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any, n_samples: int = 20) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict_with_dropout(X, n_samples=n_samples)
