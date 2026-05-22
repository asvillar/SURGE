"""Vision Transformer (ViT) for image classification.

Patch-based ViT adapted for small images (MNIST 28×28, CIFAR-10 32×32).
Uses sinusoidal positional encoding + learnable [CLS] token.

Reference
---------
Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale" ICLR 2021. https://arxiv.org/abs/2010.11929
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

_LOG = logging.getLogger("surge.pytorch.vit")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _PatchEmbed(nn.Module if TORCH_AVAILABLE else object):
    """Split image into patches and project to d_model."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, n_patches, d_model)
        return self.proj(x).flatten(2).transpose(1, 2)


class _ViTNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        n_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.transformer(x)
        return self.head(self.norm(x[:, 0]))


class ViTModel:
    """ViT image classifier.

    Parameters
    ----------
    img_size : int
        Input image side length (assumes square).
    patch_size : int
        Patch side length.  ``img_size % patch_size == 0`` required.
    in_channels : int
        1 for grayscale, 3 for RGB.
    n_classes : int
        Number of output classes (inferred from ``y`` during fit if 0).
    d_model, n_heads, n_layers, dropout :
        Transformer hyper-parameters.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        n_classes: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_epochs: int = 30,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_heads = min(n_heads, d_model)
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self._net: Any = None
        self.is_fitted = False
        self.training_history: list[dict] = []

    def _to_tensor(self, X) -> "torch.Tensor":
        """Accept (B, C, H, W), (B, H, W), or (B, H*W*C) flat."""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            # flat: try to reshape
            n = arr.shape[0]
            total = arr.shape[1]
            c, h, w = self.in_channels, self.img_size, self.img_size
            arr = arr.reshape(n, c, h, w)
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]  # (B, 1, H, W)
        # normalise 0-255 to 0-1 if needed
        if arr.max() > 1.5:
            arr = arr / 255.0
        return torch.from_numpy(arr).to(self.device)

    def fit(self, X, y, **_: Any) -> "ViTModel":
        torch.manual_seed(self.random_state)
        y_arr = np.asarray(y, dtype=np.int64)
        if self.n_classes == 0:
            self.n_classes = int(y_arr.max()) + 1

        self._net = _ViTNet(
            img_size=self.img_size, patch_size=self.patch_size,
            in_channels=self.in_channels, n_classes=self.n_classes,
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, dropout=self.dropout,
        ).to(self.device)

        Xt = self._to_tensor(X).cpu()
        yt = torch.from_numpy(y_arr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)

        best_loss = float("inf")
        no_improve = 0
        from ._progress import ProgressList
        self.training_history = ProgressList(
            self.n_epochs, verbose=self.verbose,
            log_file=self.log_file, desc=type(self).__name__,
        )

        for epoch in range(self.n_epochs):
            self._net.train()
            eloss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                eloss += loss.item() * len(xb)
            epoch_loss = eloss / len(Xt)
            _LOG.info("ViT epoch %d/%d loss=%.4f", epoch + 1, self.n_epochs, epoch_loss)
            self.training_history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if self.patience > 0 and no_improve >= self.patience:
                    break

        self.training_history.close()
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError("Not fitted")
        self._net.eval()
        Xt = self._to_tensor(X)
        all_preds = []
        bs = self.batch_size
        for i in range(0, len(Xt), bs):
            with torch.no_grad():
                logits = self._net(Xt[i:i + bs])
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(all_preds)

    def predict_proba(self, X) -> np.ndarray:
        self._net.eval()
        Xt = self._to_tensor(X)
        probs = []
        bs = self.batch_size
        for i in range(0, len(Xt), bs):
            with torch.no_grad():
                logits = self._net(Xt[i:i + bs])
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "img_size": self.img_size, "patch_size": self.patch_size,
                "in_channels": self.in_channels, "n_classes": self.n_classes,
                "d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "dropout": self.dropout,
            },
            "net_state": self._net.state_dict() if self._net else None,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        cfg = d["config"]
        self.is_fitted = d["is_fitted"]
        self._net = _ViTNet(**cfg).to(self.device)
        if d["net_state"]:
            self._net.load_state_dict(d["net_state"])
