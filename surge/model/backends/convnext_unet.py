"""ConvNeXt U-Net for 2D field-to-field prediction.

The strongest simple baseline in TheWell benchmark (Ohana et al. 2024):
a U-Net whose double-conv blocks are replaced by ConvNeXt blocks
(depthwise 7x7 convolution -> LayerNorm -> pointwise MLP with GELU,
residual with layer scale; Liu et al. 2022, "A ConvNet for the 2020s").
On turbulent_radiative_layer_2D the Well reports ConvNeXt-U-Net at
roughly half the one-step VRMSE of a vanilla U-Net.

Input/output contract matches ``pytorch.unet``: ``(B, C, H, W)``,
``(B, H, W)``, or flat ``(B, n*n)`` squares.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.convnext_unet")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = F = DataLoader = TensorDataset = None  # type: ignore


class _LayerNorm2d(nn.Module if TORCH_AVAILABLE else object):
    """LayerNorm over channels for (B, C, H, W)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ConvNeXtBlock(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, ch: int, mlp_ratio: int = 4,
                 layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = _LayerNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, mlp_ratio * ch, 1)
        self.pw2 = nn.Conv2d(mlp_ratio * ch, ch, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(ch, 1, 1))

    def forward(self, x):
        y = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma * y


class _CNextUNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_ch: int, out_ch: int, base: int, depth: int,
                 blocks_per_stage: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [min(base * 2**i, 512) for i in range(depth + 1)]

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth):
            self.enc.append(nn.Sequential(*[
                _ConvNeXtBlock(chs[i]) for _ in range(blocks_per_stage)]))
            self.down.append(nn.Sequential(
                _LayerNorm2d(chs[i]), nn.Conv2d(chs[i], chs[i + 1], 2, 2)))

        self.mid = nn.Sequential(*[
            _ConvNeXtBlock(chs[depth]) for _ in range(blocks_per_stage)])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, 2))
            self.dec.append(nn.Sequential(
                nn.Conv2d(2 * chs[i], chs[i], 1),
                *[_ConvNeXtBlock(chs[i]) for _ in range(blocks_per_stage)]))

        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc, self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:])
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


class ConvNeXtUNetModel:
    """ConvNeXt U-Net 2D surrogate (TheWell's strongest simple baseline)."""

    def __init__(
        self,
        base_channels: int = 48,
        depth: int = 3,
        blocks_per_stage: int = 2,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
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
        self.base_channels = base_channels
        self.depth = depth
        self.blocks_per_stage = blocks_per_stage
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

    def _reshape(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            n, total = arr.shape
            side = int(round(total ** 0.5))
            if side * side != total:
                raise ValueError(
                    f"Cannot infer square 2D grid from flat dim {total}")
            return arr.reshape(n, 1, side, side)
        if arr.ndim == 3:
            return arr[:, None, :, :]
        elif arr.ndim == 4:
            return arr
        raise ValueError(f"Expected 2D-flat, 3D or 4D array, got {arr.shape}")

    def fit(self, X, y, **_: Any) -> "ConvNeXtUNetModel":
        torch.manual_seed(self.random_state)
        X_arr = self._reshape(X)
        y_arr = self._reshape(y)
        self._net = _CNextUNet(X_arr.shape[1], y_arr.shape[1],
                               self.base_channels, self.depth,
                               self.blocks_per_stage).to(self.device)
        n_par = sum(p.numel() for p in self._net.parameters())
        _LOG.info("ConvNeXt-U-Net: %.1fM parameters", n_par / 1e6)
        opt = torch.optim.AdamW(self._net.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs)
        crit = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr)),
            batch_size=self.batch_size, shuffle=True)

        from ._progress import ProgressList
        self.training_history = ProgressList(
            self.n_epochs, verbose=self.verbose,
            log_file=self.log_file, desc=type(self).__name__)
        best, no_improve = float("inf"), 0
        for epoch in range(self.n_epochs):
            self._net.train()
            eloss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self._net(xb), yb)
                loss.backward()
                opt.step()
                eloss += loss.item() * len(xb)
            sched.step()
            epoch_loss = eloss / len(X_arr)
            self.training_history.append(
                {"epoch": epoch + 1, "train_loss": epoch_loss})
            if epoch_loss < best:
                best, no_improve = epoch_loss, 0
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
        Xt = torch.from_numpy(self._reshape(X))
        preds = []
        for i in range(0, len(Xt), self.batch_size):
            with torch.no_grad():
                preds.append(self._net(
                    Xt[i:i + self.batch_size].to(self.device)).cpu().numpy())
        out = np.concatenate(preds)
        return out[:, 0] if out.shape[1] == 1 else out

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {"base_channels": self.base_channels,
                       "depth": self.depth,
                       "blocks_per_stage": self.blocks_per_stage},
            "in_out": getattr(self, "_in_out", None),
            "net_state": (self._net.state_dict()
                          if self._net is not None else None),
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        self.is_fitted = d["is_fitted"]
        # network is rebuilt lazily on next fit(); state restoration for
        # inference-only use requires matching channel counts
        self._saved_state = d
