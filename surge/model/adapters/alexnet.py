"""Adapter registering ``pytorch.alexnet``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_ALEXNET_INFO = ModelInfo(
    architecture=(
        "AlexNet adapted for small images (32×32 / 28×28): five convolutional "
        "layers with ReLU activations and max-pooling, followed by three fully "
        "connected layers with dropout (p=0.5). Batch normalisation replaces the "
        "original Local Response Normalisation to stabilise training on small datasets. "
        "The original architecture used 227×227 ImageNet inputs; this variant "
        "reduces the first conv stride and kernel size to suit CIFAR-10/MNIST."
    ),
    use_cases=[
        "Image classification benchmarks on 32×32 or 28×28 inputs (CIFAR-10, MNIST)",
        "Historical baseline for comparing CNN progress on small-image tasks",
        "Teaching / reference implementation of deep convolutional classifiers",
    ],
    not_for=[
        "High-resolution images without resizing — stride/kernel tuned for 32px",
        "Regression tasks or non-image (tabular) data",
    ],
    strengths=[
        "Lightweight and fast to train on CPU compared to deeper architectures",
        "Established reference point — CIFAR-10 accuracy well-characterised in literature",
    ],
    weaknesses=[
        "Outperformed by ResNet, ViT on virtually all benchmarks",
        "No skip connections — deep versions suffer vanishing gradients",
    ],
    references=[
        "Krizhevsky, Sutskever & Hinton (2012) 'ImageNet Classification with Deep "
        "Convolutional Neural Networks' NeurIPS 2012. https://papers.nips.cc/paper/4824",
    ],
)

_ALEXNET_PROFILE = ResourceProfile(
    name="pytorch.alexnet",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="AlexNet adapted for 32×32 / 28×28 inputs. Krizhevsky et al. NeurIPS 2012.",
)


class AlexNetAdapter(BaseModelAdapter):
    """AlexNet image classifier (small-image variant)."""

    name = "pytorch.alexnet"
    backend = "pytorch"
    task_type = "classification"
    uses_internal_preprocessing = True
    resource_profile = _ALEXNET_PROFILE
    _INFO = _ALEXNET_INFO
    default_params: dict[str, Any] = {
        "img_size": 32,
        "in_channels": 3,
        "n_classes": 10,
        "dropout_fc": 0.5,
        "n_epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "patience": 10,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.alexnet_cifar")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.AlexNetModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
