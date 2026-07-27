"""LeNet-5 backend for SURGE — LeCun et al. 1998.

Classic architecture: ``Conv2d(1,6,5)→AvgPool→Conv2d(6,16,5)→AvgPool→3×Linear``.
Input shape: ``(B, 1, 28, 28)`` (MNIST grayscale).

SURGE interface
---------------
This backend wraps training on raw pixel arrays.  The adapter handles
data loading via ``torchvision`` and passes flat ``(n_samples, 784)``
arrays from the benchmark.  Internally the model reshapes to
``(B, 1, 28, 28)`` and trains with a ``DataLoader``.

For classification tasks the model uses cross-entropy loss and exposes
both ``predict`` (class indices) and ``predict_proba`` (softmax scores).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.lenet5")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class LeNet5(nn.Module if TORCH_AVAILABLE else object):
    """LeNet-5 exactly as in LeCun 1998 (for MNIST, input 28×28)."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.0) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.shape[0], -1))


class LeNet5Model:
    """
    sklearn-compatible wrapper for :class:`LeNet5`.

    Parameters
    ----------
    n_classes:
        Number of output classes (10 for MNIST).
    img_size:
        Spatial dimension (height = width) for reshaping flat input.
        Default 28 for MNIST.
    n_epochs, learning_rate, batch_size, patience:
        Training knobs.
    """

    def __init__(
        self,
        n_classes: int = 10,
        img_size: int = 28,
        in_channels: int = 1,
        dropout: float = 0.0,
        n_epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        patience: int = 5,
        lr_decay_epochs: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.n_classes = n_classes
        self.img_size = img_size
        self.in_channels = in_channels
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.lr_decay_epochs = lr_decay_epochs
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self._net: Any = None
        self.is_fitted = False
        self.training_history: list[dict] = []

    def _to_4d(self, X: np.ndarray) -> "torch.Tensor":
        """Reshape (N, pixels) or (N, H, W) → (N, C, H, W)."""
        X = np.asarray(X, dtype="float32")
        if X.max() > 2.0:
            X = X / 255.0
        if X.ndim == 2:
            B = X.shape[0]
            X = X.reshape(B, self.in_channels, self.img_size, self.img_size)
        elif X.ndim == 3:
            X = X[:, np.newaxis, :, :]  # add channel dim
        return torch.from_numpy(X)

    def fit(self, X, y, X_val=None, y_val=None) -> "LeNet5Model":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xt = self._to_4d(X).to(self.device)
        yt = torch.from_numpy(np.asarray(y, dtype="int64")).to(self.device)

        self._net = LeNet5(n_classes=self.n_classes, dropout=self.dropout).to(self.device)
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_epochs, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xvt = self._to_4d(X_val).to(self.device)
            yvt = torch.from_numpy(np.asarray(y_val, dtype="int64")).to(self.device)

        best_acc = -1.0
        best_state = None
        no_improve = 0
        from ._progress import ProgressList
        self.training_history = ProgressList(
            self.n_epochs, verbose=self.verbose,
            log_file=self.log_file, desc=type(self).__name__,
        )

        for epoch in range(self.n_epochs):
            self._net.train()
            correct, total = 0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
            scheduler.step()
            record = {"epoch": epoch + 1, "train_acc": correct / total}
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    logits_v = self._net(Xvt)
                    val_acc = (logits_v.argmax(1) == yvt).float().mean().item()
                record["val_acc"] = val_acc
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.patience > 0 and no_improve >= self.patience:
                        record["early_stop"] = True
                        self.training_history.append(record)
                        break
            self.training_history.append(record)

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self.training_history.close()
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        self._net.eval()
        Xt = self._to_4d(X).to(self.device)
        loader = DataLoader(TensorDataset(Xt), batch_size=512)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                preds.append(self._net(xb).argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        self._net.eval()
        Xt = self._to_4d(X).to(self.device)
        loader = DataLoader(TensorDataset(Xt), batch_size=512)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                probs.append(torch.softmax(self._net(xb), dim=1).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "config": {
                "n_classes": self.n_classes, "img_size": self.img_size,
                "in_channels": self.in_channels, "dropout": self.dropout,
            },
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.is_fitted = data["is_fitted"]
        cfg = data["config"]
        self.n_classes = cfg["n_classes"]
        self.img_size = cfg["img_size"]
        if data["state_dict"] is not None:
            self._net = LeNet5(n_classes=cfg["n_classes"], dropout=cfg.get("dropout", 0.0)).to(self.device)
            self._net.load_state_dict(data["state_dict"])
