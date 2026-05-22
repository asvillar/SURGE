"""Adapters registering ``pytorch.vit``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_VIT_INFO = ModelInfo(
    architecture=(
        "Vision Transformer (ViT): divides the input image into fixed-size patches "
        "(default 4×4 for 32×32 inputs), projects each patch to a d_model-dimensional "
        "embedding via a linear layer, prepends a learnable [CLS] token, adds "
        "learnable positional embeddings, then processes the sequence through a "
        "standard Transformer encoder (multi-head self-attention + MLP blocks). "
        "The [CLS] token output feeds a linear classification head. "
        "This implementation uses pre-norm (LayerNorm before attention)."
    ),
    use_cases=[
        "Image classification on CIFAR-10/100 and similar small-image benchmarks",
        "Tasks where long-range spatial relationships matter (attention sees all patches)",
        "Research baseline for comparing attention-based vs. convolution-based vision models",
    ],
    not_for=[
        "Very small datasets (<10k images) without pre-training — ViT needs data to learn "
        "positional structure that convolutions get for free via inductive bias",
        "High-resolution images without hierarchical windowing (see Swin Transformer)",
        "Non-image (tabular) data — use pytorch.ft_transformer instead",
    ],
    strengths=[
        "Captures global context with O(n_patches²) attention — no receptive field limit",
        "Scales excellently with data and model size (ViT-L/16 surpasses CNNs at ImageNet scale)",
        "Unified architecture — same Transformer used for NLP, enabling multimodal transfer",
    ],
    weaknesses=[
        "Quadratic attention cost in n_patches — slow for large images without windowing",
        "Requires more data than CNNs to generalise without pre-training",
        "Patch size is a sensitive hyperparameter affecting resolution vs. sequence length",
    ],
    references=[
        "Dosovitskiy et al. (2021) 'An Image is Worth 16x16 Words: Transformers for "
        "Image Recognition at Scale' ICLR 2021. https://arxiv.org/abs/2010.11929",
        "Touvron et al. (2021) 'Training data-efficient image transformers & distillation "
        "through attention' ICML 2021. https://arxiv.org/abs/2012.12877",
    ],
)

_VIT_PROFILE = ResourceProfile(
    name="pytorch.vit",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Vision Transformer (patch-based). Dosovitskiy et al. ICLR 2021.",
)


class ViTAdapter(BaseModelAdapter):
    """ViT image classifier."""

    name = "pytorch.vit"
    backend = "pytorch"
    task_type = "classification"
    uses_internal_preprocessing = True
    resource_profile = _VIT_PROFILE
    _INFO = _VIT_INFO
    default_params: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "n_classes": 10,
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 4,
        "dropout": 0.1,
        "n_epochs": 30,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "patience": 10,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.vit")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.ViTModel(**params)

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
