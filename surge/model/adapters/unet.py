"""Adapter registering ``pytorch.unet``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_UNET_INFO = ModelInfo(
    architecture=(
        "U-Net encoder-decoder: a fully-convolutional architecture with a "
        "contracting path (encoder) that progressively halves spatial resolution "
        "via max-pooling while doubling channels, and an expansive path (decoder) "
        "that restores resolution via transposed convolutions. Skip connections "
        "concatenate encoder feature maps to decoder feature maps at each scale, "
        "preserving fine spatial detail. Originally designed for biomedical image "
        "segmentation; widely adopted for 2D field-to-field PDE surrogates."
    ),
    use_cases=[
        "2D field-to-field regression: input field → output field on same grid",
        "Image segmentation and semantic labelling",
        "PDE surrogates where local spatial correlations matter (e.g. Darcy flow, "
        "shallow water equations)",
    ],
    not_for=[
        "1D sequences or tabular data — use pytorch.fno1d or pytorch.mlp",
        "Operator learning across different resolutions — prefer FNO for that use case",
    ],
    strengths=[
        "Skip connections preserve fine-grained spatial detail at every scale",
        "Works well with limited training data — inductive biases suit scientific fields",
        "Flexible depth and channel width — easy to scale up/down",
    ],
    weaknesses=[
        "Limited global receptive field compared to FNO (local convolutions only)",
        "Memory-intensive for large grids — each scale stores full feature maps",
    ],
    references=[
        "Ronneberger, Fischer & Brox (2015) 'U-Net: Convolutional Networks for "
        "Biomedical Image Segmentation' MICCAI 2015. https://arxiv.org/abs/1505.04597",
        "Takamoto et al. (2022) 'PDEBench' NeurIPS 2022. https://arxiv.org/abs/2210.07182",
    ],
)

_UNET_PROFILE = ResourceProfile(
    name="pytorch.unet",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="U-Net encoder-decoder for 2D field-to-field prediction. Ronneberger et al. MICCAI 2015.",
)


class UNetAdapter(BaseModelAdapter):
    """U-Net 2D surrogate."""

    name = "pytorch.unet"
    backend = "pytorch"
    task_type = "regression"
    uses_internal_preprocessing = True
    resource_profile = _UNET_PROFILE
    _INFO = _UNET_INFO
    default_params: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 32,
        "depth": 3,
        "n_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 8,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.unet")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.UNetModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
