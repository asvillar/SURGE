"""Fourier Neural Operator 2D (FNO2d) — Li et al. 2021.

Learns operators between 2D function spaces on uniform grids.
Maps input field ``(B, nx, ny)`` → output field ``(B, nx, ny)``.

Each FNO layer:
  1. Spectral branch: 2D FFT → truncate to first n_modes×n_modes → complex weights → IFFT.
  2. Residual branch: pointwise Conv2d.
  3. Sum + GELU.

Reference
---------
Li et al. (2021) "Fourier Neural Operator for Parametric PDEs" ICLR 2021.
https://arxiv.org/abs/2010.08895
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.fno2d")

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


class SpectralConv2d(nn.Module if TORCH_AVAILABLE else object):
    """2D spectral convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        # 4 sets of weights for the 4 frequency quadrants
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes, n_modes, 2))
            for _ in range(4)
        ])

    def _comul(self, x, w):
        # x: (B, Cin, Mx, My) complex; w: (Cin, Cout, Mx, My) complex
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        # x: (B, Cin, nx, ny)
        B, C, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, Cin, nx, ny//2+1) complex
        # Clamp modes to actual frequency dimensions
        m_x = min(self.n_modes, nx // 2)
        m_y = min(self.n_modes, ny // 2 + 1)
        out_ft = torch.zeros(B, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)

        def w(i):
            wc = torch.view_as_complex(self.weights[i].contiguous())  # (Cin, Cout, n_modes, n_modes)
            return wc[:, :, :m_x, :m_y]

        out_ft[:, :, :m_x, :m_y]   = self._comul(x_ft[:, :, :m_x, :m_y],   w(0))
        out_ft[:, :, -m_x:, :m_y]  = self._comul(x_ft[:, :, -m_x:, :m_y],  w(1))
        out_ft[:, :, :m_x, -m_y:]  = self._comul(x_ft[:, :, :m_x, -m_y:],  w(2))
        out_ft[:, :, -m_x:, -m_y:] = self._comul(x_ft[:, :, -m_x:, -m_y:], w(3))

        return torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))  # (B, Cout, nx, ny)


class _FNO2dNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: int,
        n_layers: int,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.spec_layers = nn.ModuleList([
            SpectralConv2d(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_layers)
        ])
        self.res_layers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, 1) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1), nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, x):
        x = self.lift(x)
        for spec, res in zip(self.spec_layers, self.res_layers):
            x = F.gelu(spec(x) + res(x))
        return self.project(x)


class FNO2dModel:
    """FNO2d operator learning surrogate.

    Accepts input fields shaped ``(B, nx, ny)`` or ``(B, nx*ny)`` and
    returns output fields of the same spatial resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_modes: int = 12,
        n_layers: int = 4,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
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
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self._net: Any = None
        self._nx: int = 0
        self._ny: int = 0
        self.is_fitted = False
        self.training_history: list[dict] = []

    def _reshape_input(self, X):
        """Ensure shape (B, Cin, nx, ny)."""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            # flat: try square
            n = arr.shape[0]
            total = arr.shape[1]
            side = int(total ** 0.5)
            if side * side == total:
                arr = arr.reshape(n, 1, side, side)
            else:
                raise ValueError(f"Cannot infer 2D shape from flat dim {total}")
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]  # (B, 1, nx, ny)
        elif arr.ndim == 4:
            pass  # already (B, C, nx, ny)
        return arr

    def fit(self, X, y, **_: Any) -> "FNO2dModel":
        torch.manual_seed(self.random_state)

        X_arr = self._reshape_input(X)
        y_arr = self._reshape_input(y)

        self._nx, self._ny = X_arr.shape[2], X_arr.shape[3]
        C_in = X_arr.shape[1]
        C_out = y_arr.shape[1]

        self._net = _FNO2dNet(C_in, C_out, self.hidden_channels, self.n_modes, self.n_layers).to(self.device)
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
        X_arr = self._reshape_input(X)
        Xt = torch.from_numpy(X_arr)
        preds = []
        for i in range(0, len(Xt), self.batch_size):
            with torch.no_grad():
                preds.append(self._net(Xt[i:i + self.batch_size].to(self.device)).cpu().numpy())
        out = np.concatenate(preds)
        # Return flat if output was flat
        return out.squeeze(1)  # (B, nx, ny)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "in_channels": self.in_channels, "out_channels": self.out_channels,
                "hidden_channels": self.hidden_channels, "n_modes": self.n_modes,
                "n_layers": self.n_layers,
            },
            "net_state": self._net.state_dict() if self._net else None,
            "nx": self._nx, "ny": self._ny,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        cfg = d["config"]
        self._nx = d["nx"]; self._ny = d["ny"]
        self.is_fitted = d["is_fitted"]
        self._net = _FNO2dNet(**cfg).to(self.device)
        if d["net_state"]:
            self._net.load_state_dict(d["net_state"])
