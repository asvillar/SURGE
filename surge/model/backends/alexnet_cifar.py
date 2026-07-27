"""AlexNet adapted for CIFAR-10 (32×32) and MNIST (28×28) inputs.

Simplified 5-conv-layer AlexNet with 3 fully-connected layers.
Dropout added to FC layers.

Reference
---------
Krizhevsky et al. (2012) "ImageNet Classification with Deep Convolutional
Neural Networks" NeurIPS 2012. https://papers.nips.cc/paper/4824-imagenet-classification
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.alexnet")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _AlexNetSmall(nn.Module if TORCH_AVAILABLE else object):
    """AlexNet adapted for small (32×32 or 28×28) images."""

    def __init__(self, in_channels: int, n_classes: int, dropout_fc: float = 0.5) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(256 * 2 * 2, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNetModel:
    """AlexNet image classifier (CIFAR/MNIST scale)."""

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        n_classes: int = 10,
        dropout_fc: float = 0.5,
        n_epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
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
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.dropout_fc = dropout_fc
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self._net: Any = None
        self.is_fitted = False
        self.training_history: Any = []

    def _to_tensor(self, X) -> "torch.Tensor":
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(len(arr), self.in_channels, self.img_size, self.img_size)
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]
        if arr.max() > 1.5:
            arr = arr / 255.0
        return torch.from_numpy(arr).to(self.device)

    def fit(self, X, y, **_: Any) -> "AlexNetModel":
        torch.manual_seed(self.random_state)
        y_arr = np.asarray(y, dtype=np.int64)
        if self.n_classes == 0:
            self.n_classes = int(y_arr.max()) + 1
        self._net = _AlexNetSmall(self.in_channels, self.n_classes, self.dropout_fc).to(self.device)
        Xt = self._to_tensor(X).cpu()
        yt = torch.from_numpy(y_arr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
        for i in range(0, len(Xt), self.batch_size):
            with torch.no_grad():
                all_preds.append(self._net(Xt[i:i + self.batch_size]).argmax(1).cpu().numpy())
        return np.concatenate(all_preds)

    def predict_proba(self, X) -> np.ndarray:
        self._net.eval()
        Xt = self._to_tensor(X)
        probs = []
        for i in range(0, len(Xt), self.batch_size):
            with torch.no_grad():
                probs.append(torch.softmax(self._net(Xt[i:i + self.batch_size]), dim=-1).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "img_size": self.img_size, "in_channels": self.in_channels,
                "n_classes": self.n_classes, "dropout_fc": self.dropout_fc,
            },
            "net_state": self._net.state_dict() if self._net else None,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        cfg = d["config"]
        self.is_fitted = d["is_fitted"]
        self._net = _AlexNetSmall(cfg["in_channels"], cfg["n_classes"], cfg["dropout_fc"]).to(self.device)
        if d["net_state"]:
            self._net.load_state_dict(d["net_state"])
