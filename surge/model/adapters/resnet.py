"""Adapters registering ``pytorch.resnet20`` and ``pytorch.resnet56``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_RESNET_INFO = ModelInfo(
    architecture=(
        "ResNet for CIFAR-10 (He et al. 2016): a stack of residual blocks where each "
        "block computes F(x) + x via a shortcut connection. For CIFAR-10 the network "
        "uses three stages of 2n blocks each (n=3 → ResNet-20 with ~0.27M params; "
        "n=9 → ResNet-56 with ~0.85M params), with 3×3 convolutions, batch "
        "normalisation, and ReLU. The skip connections solve the vanishing gradient "
        "problem, enabling training of networks far deeper than AlexNet."
    ),
    use_cases=[
        "Image classification on CIFAR-10/100 — the canonical small-image CNN benchmark",
        "Strong convolutional baseline to compare against attention-based models (ViT)",
        "Transfer learning feature extractor when fine-tuned on domain-specific images",
    ],
    not_for=[
        "Tabular or 1D data — use pytorch.mlp or pytorch.ft_transformer",
        "Very high-resolution images — use torchvision ResNet-50/101 with ImageNet weights",
    ],
    strengths=[
        "Residual connections allow training of very deep networks reliably",
        "State-of-the-art CIFAR-10 accuracy: ResNet-20 ≈ 91.3%, ResNet-56 ≈ 93.0% "
        "(He et al. 2016, Table 6)",
        "Computationally efficient: ResNet-20 trains in minutes on a single GPU",
        "Strong inductive bias for local spatial structure via convolutions",
    ],
    weaknesses=[
        "Fixed receptive field — cannot attend to global context without many layers",
        "Performance saturates with depth on CIFAR without wider channels (see WideResNet)",
    ],
    references=[
        "He et al. (2016) 'Deep Residual Learning for Image Recognition' "
        "CVPR 2016. https://arxiv.org/abs/1512.03385",
        "He et al. (2016) 'Identity Mappings in Deep Residual Networks' "
        "ECCV 2016. https://arxiv.org/abs/1603.05027",
    ],
)

_RESNET_PROFILE = ResourceProfile(
    name="pytorch.resnet_cifar",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="ResNet-20/56 for CIFAR-10 (He et al. 2016). Input: (N, 3072) flat or (N, 3, 32, 32).",
)


class _ResNetCIFARAdapter(BaseModelAdapter):
    resource_profile = _RESNET_PROFILE
    _INFO = _RESNET_INFO
    task_type = "classification"
    _n: int = 3  # overridden by subclasses
    default_params: dict[str, Any] = {
        "n_classes": 10,
        "img_size": 32,
        "in_channels": 3,
        "n_epochs": 100,
        "learning_rate": 0.1,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.resnet_cifar import ResNetCIFARModel

        params = dict(self.default_params)
        params.update(kwargs)
        params["n"] = self._n
        return ResNetCIFARModel(**params)

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


class ResNet20Adapter(_ResNetCIFARAdapter):
    """ResNet-20 for CIFAR-10 (n=3 blocks per stage; ~0.27M params)."""
    name = "pytorch.resnet20"
    backend = "pytorch"
    uses_internal_preprocessing = True
    _n = 3


class ResNet56Adapter(_ResNetCIFARAdapter):
    """ResNet-56 for CIFAR-10 (n=9 blocks per stage; ~0.85M params)."""
    name = "pytorch.resnet56"
    backend = "pytorch"
    uses_internal_preprocessing = True
    _n = 9
