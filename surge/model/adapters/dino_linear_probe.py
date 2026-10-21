"""Adapter registering ``pytorch.dino_linear_probe``.

**Transfer learning, not from-scratch.** This adapter wraps a frozen
DINOv2 ViT-S/14 self-supervised backbone (Oquab et al. 2023) used only as a
384-d feature extractor, plus a ``sklearn.LogisticRegression`` head fit on
those features (the published DINOv2 linear-probe protocol). It is NOT
comparable to from-scratch CIFAR-10 models such as ``pytorch.resnet20``
(trained with no ImageNet/LVD-142M pretraining) — see
``surge/benchmarks/metadata.yaml`` ``vision.cifar10.sota`` for both the
from-scratch ResNet-20 reference (~91.25%, He et al. 2016) and the
large-pretraining literature SOTA this adapter is positioned against.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaseModelAdapter, ModelInfo
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_DINO_INFO = ModelInfo(
    architecture=(
        "Frozen DINOv2 ViT-S/14 backbone (Oquab et al. 2023; self-supervised, "
        "pretrained on LVD-142M) used purely as a 384-d feature extractor — "
        "CIFAR-10 images are upsampled 32x32 -> 224x224 and normalised with "
        "ImageNet statistics, then passed through the frozen (no-grad, eval-mode) "
        "transformer once per split; features are cached to disk. A "
        "scikit-learn LogisticRegression linear probe is then fit on the "
        "384-d features. No backbone weights are updated (transfer learning, "
        "not from-scratch training)."
    ),
    use_cases=[
        "Strong, cheap-to-train transfer-learning baseline for CIFAR-10/100 "
        "and other small-image benchmarks",
        "Establishing an upper-bound reference against which from-scratch "
        "architectures (ResNet-20, ViT, AlexNet) can be judged",
        "Any task where a frozen self-supervised feature extractor + linear "
        "head is an appropriate literature-standard protocol",
    ],
    not_for=[
        "Measuring from-scratch architecture quality — use pytorch.resnet20 "
        "or pytorch.vit for that; this adapter's accuracy reflects DINOv2 "
        "pretraining on LVD-142M, not the linear head",
        "Domains far outside DINOv2's pretraining distribution (natural "
        "images) without re-validating transfer quality",
        "Settings where no internet/torch-hub cache access is available for "
        "the one-time backbone download",
    ],
    strengths=[
        "Near state-of-the-art accuracy from a linear head alone (DINOv2 "
        "linear-probe protocol; Oquab et al. 2023)",
        "Training cost is just a LogisticRegression fit once features are "
        "extracted; feature extraction itself is cached across reruns",
        "Frozen backbone — no risk of catastrophic forgetting or overfitting "
        "a small linear head to 45-50k images",
    ],
    weaknesses=[
        "Backbone download (~90MB) requires internet access and ~384-d "
        "feature extraction pass over the full dataset on first run",
        "Not a from-scratch result — not directly comparable to the "
        "Acc >= 0.91 from-scratch gate without noting the pretraining",
        "Linear probe accuracy is capped well below full fine-tuning of the "
        "backbone (which this adapter deliberately does not do)",
    ],
    references=[
        "Oquab et al. (2023) 'DINOv2: Learning Robust Visual Features without "
        "Supervision'. https://arxiv.org/abs/2304.07193",
        "Caron et al. (2021) 'Emerging Properties in Self-Supervised Vision "
        "Transformers' (DINO). ICCV 2021. https://arxiv.org/abs/2104.14294",
    ],
)

_DINO_PROFILE = ResourceProfile(
    name="pytorch.dino_linear_probe",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes=(
        "Frozen DINOv2 ViT-S/14 feature extraction (manual batching, no "
        "DataLoader workers) + sklearn LogisticRegression head. Device "
        "picked via surge.utils.resolve_device (SURGE_DEVICE env)."
    ),
)


class DINOLinearProbeAdapter(BaseModelAdapter):
    """Frozen DINOv2 ViT-S/14 features + LogisticRegression linear probe.

    Transfer learning: see module docstring and ``info()`` for details.
    ``prepare_for_fit`` tags the fit banner / ``result.extra["resources"]``
    with ``pretrained_backbone="dinov2_vits14"`` so leaderboard consumers can
    tell this result apart from from-scratch runs without re-deriving it.
    """

    name = "pytorch.dino_linear_probe"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _DINO_PROFILE
    _INFO = _DINO_INFO
    task_type = "classification"
    default_params: dict[str, Any] = {
        "n_classes": 10,
        "img_size": 32,
        "in_channels": 3,
        "batch_size": 256,
        "resize_to": 224,
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    }

    def prepare_for_fit(
        self,
        *,
        resources: Any = None,
        X_shape: Any = None,
        y_shape: Any = None,
        extra: Optional[dict] = None,
    ) -> dict:
        extra = dict(extra or {})
        extra.setdefault("pretrained_backbone", "dinov2_vits14")
        extra.setdefault("probe", "sklearn.LogisticRegression")
        extra.setdefault("transfer_learning", True)
        return super().prepare_for_fit(
            resources=resources, X_shape=X_shape, y_shape=y_shape, extra=extra,
        )

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.dino_linear_probe import DINOLinearProbeModel

        params = dict(self.default_params)
        params.update(kwargs)
        return DINOLinearProbeModel(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> Any:
        result = self._model.fit(X, y, X_val, y_val)
        self.mark_fitted()
        return result

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
