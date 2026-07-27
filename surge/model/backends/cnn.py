"""1-D CNN surrogate backend for SURGE.

Designed for spatial fields and temporal sequences represented as 1-D
signals.  Input arrays follow the SURGE convention of shape
``(n_samples, n_x)`` for single-channel or ``(n_samples, n_x, C_in)``
for multi-channel data.  Internally transposed to ``(B, C_in, n_x)``
for ``torch.nn.Conv1d``.

Architecture
------------
- Configurable stack of dilated ``Conv1d`` layers with GELU activation.
- Optional residual projection when the channel width changes.
- Linear output head mapping to ``C_out`` channels.
- ``StandardScaler`` on both inputs and outputs.
- Early-stopping on validation loss if ``X_val`` is supplied.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.cnn1d")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _DilatedBlock(nn.Module if TORCH_AVAILABLE else object):
    """Conv1d → GELU → (optional residual projection)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x))) + self.proj(x)


class CNN1DNet(nn.Module if TORCH_AVAILABLE else object):
    """Stack of dilated residual blocks + linear head."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        blocks = []
        ch_in = in_channels
        for i in range(n_layers):
            dil = 2 ** (i % 4)
            blocks.append(_DilatedBlock(ch_in, hidden_channels, kernel_size, dil, dropout))
            ch_in = hidden_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        return self.head(self.blocks(x))


class CNN1DModel:
    """
    sklearn-compatible wrapper for :class:`CNN1DNet`.

    Accepts ``X`` of shape ``(n_samples, n_x)`` or
    ``(n_samples, n_x, C_in)`` and ``y`` of the same spatial size.

    Parameters
    ----------
    hidden_channels, n_layers, kernel_size, dropout:
        Architecture knobs.
    n_epochs, learning_rate, batch_size, patience:
        Training knobs.
    in_channels, out_channels:
        Set automatically from data if ``None``.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.05,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 20,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file

        self._net: Any = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.training_history: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_3d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (arr_3d, was_2d) where arr_3d has shape (B, n_x, C).

        CNN1D is a sequence-to-sequence model — both X and y must have the
        same spatial length n_x.  Scalar (1-D) targets are not supported;
        use ``pytorch.mlp`` or ``pytorch.residual_mlp`` for tabular data.
        """
        if arr.ndim == 1:
            raise ValueError(
                "CNN1D requires sequence targets (2-D or 3-D y). "
                "For scalar targets use pytorch.mlp or pytorch.residual_mlp."
            )
        if arr.ndim == 2:
            return arr[:, :, np.newaxis], True
        return arr, False

    @staticmethod
    def _from_3d(arr: np.ndarray, was_2d: bool) -> np.ndarray:
        if was_2d:
            return arr[:, :, 0]
        return arr

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None) -> "CNN1DModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X3, self._x_was_2d = self._to_3d(np.asarray(X, dtype=float))
        y3, self._y_was_2d = self._to_3d(np.asarray(y, dtype=float))
        B, n_x, c_in = X3.shape
        _, _, c_out = y3.shape

        # Fit scalers on flattened spatial dims.
        Xs = self.scaler_X.fit_transform(X3.reshape(B, -1)).reshape(B, n_x, c_in)
        ys = self.scaler_y.fit_transform(y3.reshape(B, -1)).reshape(B, n_x, c_out)

        in_ch = self._in_channels or c_in
        out_ch = self._out_channels or c_out
        self._n_x = n_x
        self._c_in = c_in
        self._c_out = c_out

        self._net = CNN1DNet(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_channels=self.hidden_channels,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Transpose to (B, C, L) for Conv1d.
        Xt = torch.from_numpy(Xs.transpose(0, 2, 1).astype("float32"))
        yt = torch.from_numpy(ys.transpose(0, 2, 1).astype("float32"))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv3, _ = self._to_3d(np.asarray(X_val, dtype=float))
            yv3, _ = self._to_3d(np.asarray(y_val, dtype=float))
            Bv = Xv3.shape[0]
            Xvs = self.scaler_X.transform(Xv3.reshape(Bv, -1)).reshape(Bv, n_x, c_in)
            yvs = self.scaler_y.transform(yv3.reshape(Bv, -1)).reshape(Bv, n_x, c_out)
            Xvt = torch.from_numpy(Xvs.transpose(0, 2, 1).astype("float32")).to(self.device)
            yvt = torch.from_numpy(yvs.transpose(0, 2, 1).astype("float32")).to(self.device)

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
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    vl = criterion(self._net(Xvt), yvt).item()
                record["val_loss"] = vl
                if vl < best_val:
                    best_val = vl
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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError("Model not fitted")
        self._net.eval()
        X3, _ = self._to_3d(np.asarray(X, dtype=float))
        B, n_x, c_in = X3.shape
        Xs = self.scaler_X.transform(X3.reshape(B, -1)).reshape(B, n_x, c_in)
        Xt = torch.from_numpy(Xs.transpose(0, 2, 1).astype("float32")).to(self.device)
        with torch.no_grad():
            out = self._net(Xt).cpu().numpy()  # (B, C_out, n_x)
        out = out.transpose(0, 2, 1)  # (B, n_x, C_out)
        B_out, n_x_out, c_out = out.shape
        out_s = self.scaler_y.inverse_transform(out.reshape(B_out, -1)).reshape(B_out, n_x_out, c_out)
        return self._from_3d(out_s, self._y_was_2d)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "config": {
                "in_ch": self._c_in, "out_ch": self._c_out, "n_x": self._n_x,
                "hidden_channels": self.hidden_channels, "n_layers": self.n_layers,
                "kernel_size": self.kernel_size, "dropout": self.dropout,
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
        self._c_in, self._c_out, self._n_x = cfg["in_ch"], cfg["out_ch"], cfg["n_x"]
        if data["state_dict"] is not None:
            self._net = CNN1DNet(
                in_channels=cfg["in_ch"], out_channels=cfg["out_ch"],
                hidden_channels=cfg["hidden_channels"], n_layers=cfg["n_layers"],
                kernel_size=cfg["kernel_size"], dropout=cfg["dropout"],
            ).to(self.device)
            self._net.load_state_dict(data["state_dict"])
