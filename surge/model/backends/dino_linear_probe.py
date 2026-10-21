"""Frozen DINOv2 ViT-S/14 backbone + LogisticRegression linear probe.

This backend is a **transfer-learning** baseline, not a from-scratch
classifier: DINOv2 (Oquab et al. 2023, "DINOv2: Learning Robust Visual
Features without Supervision", https://arxiv.org/abs/2304.07193) is a
self-supervised ViT pretrained on LVD-142M. The ``dinov2_vits14`` backbone
(21M params, 384-d [CLS] token) is downloaded once via ``torch.hub`` to the
user's torch hub cache, frozen (no gradient updates, eval mode), and used
purely as a feature extractor. A scikit-learn ``LogisticRegression`` head is
then fit on the extracted 384-d features — this mirrors the published
DINOv2 linear-probe protocol (see ``surge/benchmarks/metadata.yaml``
``vision.cifar10.sota`` for the reported numbers and source).

Because the backbone is frozen, features depend only on the input images
(not on any trainable state), so they are extracted once per split and
cached to an ``.npz`` file keyed by a cheap content fingerprint
(shape + strided value sample) — re-running the benchmark after the first
pass is seconds instead of minutes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ImageNet normalisation statistics used by the DINOv2 released weights.
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

HUB_REPO = "facebookresearch/dinov2"
HUB_MODEL = "dinov2_vits14"
EXPECTED_FEATURE_DIM = 384  # ViT-S/14 [CLS] token width (informational only)


def _default_cache_path() -> Path:
    """``<repo_root>/data/datasets/benchmarks/vision/dino_features_cifar10.npz``."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "datasets" / "benchmarks" / "vision" / "dino_features_cifar10.npz"


def _fingerprint(X: np.ndarray) -> str:
    """Cheap, order-sensitive content fingerprint for cache keys.

    Hashes the array shape plus a fixed-stride subsample of values, so cost
    is independent of ``N`` (no full-array hashing of tens of thousands of
    images) while still distinguishing train/val/test splits and different
    random seeds/shuffles.
    """
    flat = np.ascontiguousarray(X).reshape(-1)
    sample = flat[::997][:4096].astype(np.float32, copy=False)
    h = hashlib.sha1()
    h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    h.update(sample.tobytes())
    return h.hexdigest()[:16]


def load_hub_backbone() -> Any:
    """Download (once, cached by torch.hub) and return the raw DINOv2 ViT-S/14 module."""
    import torch

    return torch.hub.load(HUB_REPO, HUB_MODEL)


class DINOLinearProbeModel:
    """Frozen DINOv2 ViT-S/14 features + ``sklearn.LogisticRegression`` head.

    Parameters
    ----------
    hub_loader:
        Zero-arg callable returning an un-prepared ``nn.Module`` taking
        ``(N, 3, resize_to, resize_to)`` float tensors and returning
        ``(N, D)`` features. Defaults to :func:`load_hub_backbone` (real
        DINOv2 download). Tests inject a tiny random conv here to avoid
        any network access.
    cache_path:
        Where extracted features are cached (npz, one entry per content
        fingerprint). Defaults to
        ``data/datasets/benchmarks/vision/dino_features_cifar10.npz``.
    """

    def __init__(
        self,
        *,
        n_classes: int = 10,
        img_size: int = 32,
        in_channels: int = 3,
        batch_size: int = 256,
        resize_to: int = 224,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        hub_loader: Optional[Callable[[], Any]] = None,
        **_ignored: Any,
    ) -> None:
        self.n_classes = n_classes
        self.img_size = img_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.resize_to = resize_to
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._device_req = device
        self.cache_path = Path(cache_path) if cache_path else _default_cache_path()
        self._hub_loader = hub_loader or load_hub_backbone
        self._backbone: Any = None
        self._torch_device: Any = None
        self._clf: Any = None
        self.feature_dim_: Optional[int] = None
        self.training_history: list[dict] = []

    # -- backbone -------------------------------------------------------

    def _load_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        import torch

        from surge.utils import resolve_device

        self._torch_device = (
            torch.device(self._device_req) if self._device_req else resolve_device()
        )
        backbone = self._hub_loader()
        backbone.eval()
        backbone.to(self._torch_device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._backbone = backbone
        return backbone

    # -- feature extraction (cached) -------------------------------------

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        key = _fingerprint(np.asarray(X))
        n = len(X)
        cached = self._read_cache(key, n)
        if cached is not None:
            return cached

        import torch
        import torch.nn.functional as F

        backbone = self._load_backbone()
        device = self._torch_device

        imgs = np.asarray(X, dtype=np.float32)
        if imgs.ndim == 2:
            imgs = imgs.reshape(n, self.in_channels, self.img_size, self.img_size)
        mean = torch.tensor(DINOV2_MEAN, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std = torch.tensor(DINOV2_STD, dtype=torch.float32, device=device).view(1, -1, 1, 1)

        feats: Optional[np.ndarray] = None
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch = torch.from_numpy(imgs[start:end]).to(device)
                if batch.shape[1] != 3 and mean.shape[1] == 3:
                    batch = batch.repeat(1, 3, 1, 1) if batch.shape[1] == 1 else batch
                batch = F.interpolate(
                    batch, size=(self.resize_to, self.resize_to),
                    mode="bicubic", align_corners=False,
                )
                batch = (batch - mean) / std
                out = backbone(batch)
                if out.ndim > 2:
                    out = out.reshape(out.shape[0], -1)
                out_np = out.detach().cpu().numpy().astype(np.float32)
                if feats is None:
                    feats = np.empty((n, out_np.shape[1]), dtype=np.float32)
                feats[start:end] = out_np

        assert feats is not None
        self.feature_dim_ = feats.shape[1]
        self._write_cache(key, feats)
        return feats

    def _read_cache(self, key: str, n: int) -> Optional[np.ndarray]:
        if not self.cache_path.exists():
            return None
        try:
            with np.load(self.cache_path) as data:
                arr_key = f"feat_{key}"
                if arr_key in data.files:
                    feats = data[arr_key]
                    if feats.shape[0] == n:
                        self.feature_dim_ = feats.shape[1]
                        return feats
        except Exception:
            return None
        return None

    def _write_cache(self, key: str, feats: np.ndarray) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            try:
                with np.load(self.cache_path) as data:
                    existing = {k: data[k] for k in data.files}
            except Exception:
                existing = {}
        existing[f"feat_{key}"] = feats
        np.savez_compressed(self.cache_path, **existing)

    # -- sklearn-style contract -------------------------------------------

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> "DINOLinearProbeModel":
        from sklearn.linear_model import LogisticRegression

        feats = self._extract_features(X)
        y = np.asarray(y).ravel()
        self._clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state,
        )
        self._clf.fit(feats, y)
        self.training_history = [{
            "epoch": 0,
            "train_loss": float(-self._clf.score(feats, y)),
        }]
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise ValueError("Model must be fitted before predicting")
        return self._clf.predict(self._extract_features(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise ValueError("Model must be fitted before predicting")
        return self._clf.predict_proba(self._extract_features(X))

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({
            "clf": self._clf,
            "feature_dim": self.feature_dim_,
            "params": {
                "n_classes": self.n_classes,
                "img_size": self.img_size,
                "in_channels": self.in_channels,
                "batch_size": self.batch_size,
                "resize_to": self.resize_to,
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            },
        }, path)

    def load(self, path: str) -> None:
        import joblib

        data = joblib.load(path)
        self._clf = data["clf"]
        self.feature_dim_ = data.get("feature_dim")
