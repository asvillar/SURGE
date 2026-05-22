"""U-Net encoder-decoder for 2D field-to-field prediction.

Classic U-Net with skip connections for any spatial resolution.

Reference
---------
Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical
Image Segmentation" MICCAI 2015. https://arxiv.org/abs/1505.04597
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

_LOG = logging.getLogger("surge.pytorch.unet")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = F = optim = DataLoader = TensorDataset = None  # type: ignore


class _DoubleConv(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _UNetNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int, depth: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        ch = base_channels
        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        prev = in_channels
        enc_chs = []
        for _ in range(depth):
            self.enc.append(_DoubleConv(prev, ch))
            enc_chs.append(ch)
            prev = ch
            ch = min(ch * 2, 512)

        self.bottleneck = _DoubleConv(prev, ch)

        self.upconvs = nn.ModuleList()
        self.dec = nn.ModuleList()
        enc_chs_rev = list(reversed(enc_chs))
        up_in = ch
        for skip_ch in enc_chs_rev:
            self.upconvs.append(nn.ConvTranspose2d(up_in, skip_ch, 2, stride=2))
            self.dec.append(_DoubleConv(skip_ch * 2, skip_ch))
            up_in = skip_ch

        self.final = nn.Conv2d(enc_chs[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.dec, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))
        return self.final(x)


class UNetModel:
    """U-Net 2D surrogate. Input/output: ``(B, nx, ny)`` fields."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
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
        self.training_history: Any = []

    def _reshape(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            return arr[:, None, :, :]
        elif arr.ndim == 4:
            return arr
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")

    def fit(self, X, y, **_: Any) -> "UNetModel":
        torch.manual_seed(self.random_state)
        X_arr = self._reshape(X)
        y_arr = self._reshape(y)
        C_in = X_arr.shape[1]
        C_out = y_arr.shape[1]

        self._net = _UNetNet(C_in, C_out, self.base_channels, self.depth).to(self.device)
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        Xt = torch.from_numpy(X_arr)
        yt = torch.from_numpy(y_arr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

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
        X_arr = self._reshape(X)
        Xt = torch.from_numpy(X_arr)
        preds = []
        for i in range(0, len(Xt), self.batch_size):
            with torch.no_grad():
                preds.append(self._net(Xt[i:i + self.batch_size].to(self.device)).cpu().numpy())
        return np.concatenate(preds).squeeze(1)  # (B, nx, ny)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "in_channels": self.in_channels, "out_channels": self.out_channels,
                "base_channels": self.base_channels, "depth": self.depth,
            },
            "net_state": self._net.state_dict() if self._net else None,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        cfg = d["config"]
        self.is_fitted = d["is_fitted"]
        self._net = _UNetNet(**cfg).to(self.device)
        if d["net_state"]:
            self._net.load_state_dict(d["net_state"])
