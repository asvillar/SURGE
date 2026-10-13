"""Adapter registering ``pytorch.convnext_unet`` (TheWell's strongest baseline)."""

from __future__ import annotations

from typing import Any

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE

_PROFILE = ResourceProfile(
    name="pytorch.convnext_unet",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="ConvNeXt-block U-Net (Liu 2022 blocks; Well baseline).",
)


class ConvNeXtUNetAdapter(BaseModelAdapter):
    """ConvNeXt U-Net 2D field-to-field adapter."""

    name = "pytorch.convnext_unet"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _PROFILE
    task_type = "regression"

    default_params: dict[str, Any] = {
        "base_channels": 48,
        "depth": 3,
        "blocks_per_stage": 2,
        "n_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 8,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.convnext_unet import ConvNeXtUNetModel

        params = dict(self.default_params)
        params.update(kwargs)
        return ConvNeXtUNetModel(**params)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
