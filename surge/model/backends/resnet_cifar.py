"""ResNet for CIFAR-10 — He et al. 2016.

CIFAR-specific design: 16-filter stem, no max-pool, 3 residual stages
with ``(n, n, n)`` basic blocks.

    ``n=3``  → ResNet-20  (commonly cited: 91.25% on CIFAR-10)
    ``n=9``  → ResNet-56  (commonly cited: 93.03% on CIFAR-10)
    ``n=18`` → ResNet-110

Input: ``(B, 3, 32, 32)``.  Flat input accepted: ``(B, 3072)``.

SURGE interface
---------------
Same as :class:`~surge.model.backends.lenet.LeNet5Model` — accepts flat
arrays or shaped tensors; exposes ``predict`` and ``predict_proba``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

_LOG = logging.getLogger("surge.pytorch.resnet_cifar")

# CIFAR-10 channel mean/std — must match in training (aug path) and predict.
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _BasicBlock(nn.Module if TORCH_AVAILABLE else object):
    """2-layer residual block with BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module if TORCH_AVAILABLE else object):
    """He et al. 2016 CIFAR ResNet.  Set ``n=3`` for ResNet-20, ``n=9`` for ResNet-56."""

    def __init__(self, n: int = 3, n_classes: int = 10, in_channels: int = 3) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int):
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.shape[0], -1)
        return self.fc(x)


class ResNetCIFARModel:
    """
    sklearn-compatible wrapper for :class:`ResNetCIFAR`.

    Parameters
    ----------
    n:
        Depth parameter: 3 → ResNet-20, 9 → ResNet-56.
    n_classes:
        10 for CIFAR-10, 100 for CIFAR-100.
    img_size:
        32 for CIFAR-10/100.
    n_epochs, learning_rate, batch_size, weight_decay, patience:
        Training knobs.
    """

    def __init__(
        self,
        n: int = 3,
        n_classes: int = 10,
        img_size: int = 32,
        in_channels: int = 3,
        n_epochs: int = 100,
        learning_rate: float = 0.1,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.n = n
        self.n_classes = n_classes
        self.img_size = img_size
        self.in_channels = in_channels
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self._net: Any = None
        self.is_fitted = False
        self.training_history: list[dict] = []

    def _to_4d(self, X: np.ndarray) -> "torch.Tensor":
        X = np.asarray(X, dtype="float32")
        if X.max() > 2.0:
            X = X / 255.0
        B = X.shape[0]
        if X.ndim == 2:
            X = X.reshape(B, self.in_channels, self.img_size, self.img_size)
        elif X.ndim == 3:
            # (B, H, W) → add channel dim
            X = X[:, np.newaxis, :, :]
        # Normalise to CIFAR-10 statistics (must match training augmentation path).
        mean = np.array(_CIFAR_MEAN, dtype="float32").reshape(1, 3, 1, 1)
        std = np.array(_CIFAR_STD, dtype="float32").reshape(1, 3, 1, 1)
        if X.shape[1] == 3:
            X = (X - mean) / std
        return torch.from_numpy(X)

    def fit(self, X, y, X_val=None, y_val=None) -> "ResNetCIFARModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype="int64")

        # Apply standard CIFAR-10 augmentation only for RGB (3-channel) inputs.
        # He et al. 2016 Appendix: random horizontal flip + 4-pixel padding + random crop.
        # Grayscale inputs (MNIST) skip augmentation.
        use_augment = self.in_channels == 3
        if use_augment:
            import torchvision.transforms as T

            class _AugDataset(torch.utils.data.Dataset):
                def __init__(self, X_np, y_np, h, w, c):
                    # X_np: (N, C*H*W) float32 in [0,1]
                    self._X = X_np.reshape(-1, c, h, w)
                    self._y = y_np
                    self._aug = T.Compose([
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(h, padding=4),
                    ])
                    self._mean = torch.tensor(_CIFAR_MEAN).view(c, 1, 1)
                    self._std = torch.tensor(_CIFAR_STD).view(c, 1, 1)

                def __len__(self):
                    return len(self._y)

                def __getitem__(self, idx):
                    img = torch.from_numpy(self._X[idx].copy())  # (C, H, W)
                    img = self._aug(img)
                    img = (img - self._mean) / self._std
                    return img, int(self._y[idx])

            h = w = int(X_arr.shape[1] // self.in_channels) ** (1 / 2)
            # More robust: infer spatial dim from known channel count
            spatial = int((X_arr.shape[1] / self.in_channels) ** 0.5)
            train_dataset = _AugDataset(X_arr, y_arr, spatial, spatial, self.in_channels)
            loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            Xt = self._to_4d(X).to(self.device)
            yt = torch.from_numpy(y_arr).to(self.device)
            loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        self._net = ResNetCIFAR(n=self.n, n_classes=self.n_classes, in_channels=self.in_channels).to(self.device)
        optimizer = optim.SGD(
            self._net.parameters(), lr=self.learning_rate,
            momentum=0.9, weight_decay=self.weight_decay, nesterov=True,
        )
        milestones = [int(0.5 * self.n_epochs), int(0.75 * self.n_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

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
                xb = xb.to(self.device)
                yb = torch.as_tensor(yb, dtype=torch.int64).to(self.device)
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
        loader = DataLoader(TensorDataset(Xt), batch_size=256)
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
        loader = DataLoader(TensorDataset(Xt), batch_size=256)
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
                "n": self.n, "n_classes": self.n_classes,
                "img_size": self.img_size, "in_channels": self.in_channels,
            },
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.is_fitted = data["is_fitted"]
        cfg = data["config"]
        self.n = cfg["n"]
        self.n_classes = cfg["n_classes"]
        if data["state_dict"] is not None:
            self._net = ResNetCIFAR(n=cfg["n"], n_classes=cfg["n_classes"], in_channels=self.in_channels).to(self.device)
            self._net.load_state_dict(data["state_dict"])
