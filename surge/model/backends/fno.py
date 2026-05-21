"""Fourier Neural Operator (FNO1d) — Li et al. 2021.

Implements a 1-D FNO for learning operators between function spaces on
uniform grids.  Typical use: mapping initial/boundary conditions to PDE
solutions.

Architecture
------------
Input:  ``(B, n_x, C_in)`` — function values sampled at ``n_x`` grid points
        plus optional positional/parameter channels.
Output: ``(B, n_x, C_out)``.

Each FNO layer performs:
    1. Spectral branch: FFT → truncate to first ``n_modes`` → learn complex
       weights ``W ∈ C^{C_in × C_out × n_modes}`` → IFFT.
    2. Residual branch: pointwise ``Conv1d`` of width ``C_in → C_out``.
    3. Sum + activation (GELU).

Final linear projection maps the last hidden channel to ``C_out``.

Interface
---------
Identical to :class:`surge.model.backends.cnn.CNN1DModel`: accepts flat
``(B, n_x)`` or ``(B, n_x, C_in)`` arrays, uses ``StandardScaler``, and
returns numpy arrays from ``predict``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

_LOG = logging.getLogger("surge.pytorch.fno1d")

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


class SpectralConv1d(nn.Module if TORCH_AVAILABLE else object):
    """Spectral convolution layer: FFT → weights → IFFT."""

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        # Complex weights, stored as real (2 = real + imag).
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes, 2)
        )

    def _complex_mul1d(self, x, w):
        # x: (B, C_in, n_modes) complex; w: (C_in, C_out, n_modes) complex
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x):
        # x: (B, C_in, n_x)
        B, C, n_x = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, n_x//2+1) complex
        modes = min(self.n_modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        w_complex = torch.view_as_complex(self.weights.contiguous())  # (C_in, C_out, n_modes)
        out_ft[:, :, :modes] = self._complex_mul1d(x_ft[:, :, :modes], w_complex[:, :, :modes])
        return torch.fft.irfft(out_ft, n=n_x, dim=-1)  # (B, C_out, n_x)


class FNO1dNet(nn.Module if TORCH_AVAILABLE else object):
    """Fourier Neural Operator network for 1-D domains."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden_channels, 1)
        self.spec_convs = nn.ModuleList([
            SpectralConv1d(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_layers)
        ])
        self.res_convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )

    def forward(self, x):
        # x: (B, C_in, n_x)
        x = self.lift(x)
        for spec, res in zip(self.spec_convs, self.res_convs):
            x = torch.nn.functional.gelu(spec(x) + res(x))
        return self.project(x)


class FNO1dModel:
    """
    sklearn-compatible wrapper for :class:`FNO1dNet`.

    Accepts ``X`` of shape ``(n_samples, n_x)`` (input field) and ``y``
    of shape ``(n_samples, n_x)`` or ``(n_samples, n_x, C_out)`` (output
    field).  An optional positional channel (normalised ``x ∈ [0, 1]``) is
    appended automatically if ``append_grid=True``.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        append_grid: bool = True,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.append_grid = append_grid
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file

        self._net: Any = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.training_history: list[dict] = []

    @staticmethod
    def _to_3d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
        if arr.ndim == 2:
            return arr[:, :, np.newaxis], True
        return arr, False

    @staticmethod
    def _from_3d(arr: np.ndarray, was_2d: bool) -> np.ndarray:
        if was_2d:
            return arr[:, :, 0]
        return arr

    def fit(self, X, y) -> "FNO1dModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim == 1:
            raise ValueError(
                "FNO1d requires the target y to be a spatial field with shape "
                "(n_samples, n_x) or (n_samples, n_x, c_out). "
                "Scalar targets (1-D y) are not supported — use pytorch.mlp or "
                "sklearn.random_forest for tabular regression instead."
            )

        X3, self._x_was_2d = self._to_3d(X_arr)
        y3, self._y_was_2d = self._to_3d(y_arr)
        B, n_x, c_in = X3.shape
        _, _, c_out = y3.shape

        # Optionally append normalised grid channel.
        if self.append_grid:
            grid = np.linspace(0, 1, n_x)[np.newaxis, :, np.newaxis].repeat(B, axis=0)
            X3 = np.concatenate([X3, grid], axis=2)  # (B, n_x, c_in + 1)
        c_in_total = X3.shape[2]

        Xs = self.scaler_X.fit_transform(X3.reshape(B, -1)).reshape(B, n_x, c_in_total)
        ys = self.scaler_y.fit_transform(y3.reshape(B, -1)).reshape(B, n_x, c_out)
        self._n_x = n_x
        self._c_in = c_in
        self._c_in_total = c_in_total
        self._c_out = c_out

        self._net = FNO1dNet(
            in_channels=c_in_total, out_channels=c_out,
            hidden_channels=self.hidden_channels, n_modes=self.n_modes, n_layers=self.n_layers,
        ).to(self.device)

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        Xt = torch.from_numpy(Xs.transpose(0, 2, 1).astype("float32"))
        yt = torch.from_numpy(ys.transpose(0, 2, 1).astype("float32"))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        best_val = float("inf")
        best_state = None
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
            record = {"epoch": epoch + 1, "train_loss": eloss / len(Xt)}
            if eloss / len(Xt) < best_val:
                best_val = eloss / len(Xt)
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
        if not self.is_fitted or self._net is None:
            raise ValueError("Model not fitted")
        self._net.eval()
        X3, _ = self._to_3d(np.asarray(X, dtype=float))
        B, n_x, _ = X3.shape
        if self.append_grid:
            grid = np.linspace(0, 1, n_x)[np.newaxis, :, np.newaxis].repeat(B, axis=0)
            X3 = np.concatenate([X3, grid], axis=2)
        Xs = self.scaler_X.transform(X3.reshape(B, -1)).reshape(B, n_x, self._c_in_total)
        Xt = torch.from_numpy(Xs.transpose(0, 2, 1).astype("float32")).to(self.device)
        with torch.no_grad():
            out = self._net(Xt).cpu().numpy()  # (B, C_out, n_x)
        out = out.transpose(0, 2, 1)  # (B, n_x, C_out)
        out_s = self.scaler_y.inverse_transform(out.reshape(B, -1)).reshape(B, n_x, self._c_out)
        return self._from_3d(out_s, self._y_was_2d)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
            "config": {
                "c_in_total": self._c_in_total, "c_out": self._c_out, "n_x": self._n_x,
                "hidden_channels": self.hidden_channels, "n_modes": self.n_modes,
                "n_layers": self.n_layers,
            },
            "is_fitted": self.is_fitted,
            "_x_was_2d": getattr(self, "_x_was_2d", True),
            "_y_was_2d": getattr(self, "_y_was_2d", True),
        }, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.is_fitted = data["is_fitted"]
        self._x_was_2d = data.get("_x_was_2d", True)
        self._y_was_2d = data.get("_y_was_2d", True)
        cfg = data["config"]
        self._c_in_total, self._c_out, self._n_x = cfg["c_in_total"], cfg["c_out"], cfg["n_x"]
        if data["state_dict"] is not None:
            self._net = FNO1dNet(
                in_channels=cfg["c_in_total"], out_channels=cfg["c_out"],
                hidden_channels=cfg["hidden_channels"], n_modes=cfg["n_modes"],
                n_layers=cfg["n_layers"],
            ).to(self.device)
            self._net.load_state_dict(data["state_dict"])
