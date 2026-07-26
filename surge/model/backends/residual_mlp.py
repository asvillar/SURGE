"""Residual MLP backend for SURGE (He et al. 2015 skip-connection style).

Shares the same training loop contract as :mod:`surge.model.pytorch_impl`:
- ``fit(X_train, y_train, X_val=None, y_val=None)``
- ``predict(X)``
- ``training_history`` list[dict]

The architecture wraps every pair of hidden layers in a residual block:

    out = LayerNorm(Linear2(ReLU(Linear1(x))) + proj(x))

where ``proj`` is a 1×1 projection when dimensions differ.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from sklearn.preprocessing import StandardScaler

from ..layer_schedule import LayerSchedule, resolve_hidden_layers

_LOG = logging.getLogger("surge.pytorch.residual_mlp")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _ResidualBlock(nn.Module if TORCH_AVAILABLE else object):
    """Two-layer residual block with optional projection and LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ResidualBlock")
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        h = self.drop(self.act(self.linear1(x)))
        h = self.linear2(h)
        return self.norm(h + residual)


class ResidualMLPNet(nn.Module if TORCH_AVAILABLE else object):
    """Residual MLP network: stack of _ResidualBlocks followed by a linear output head."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ResidualMLPNet")
        super().__init__()
        sizes = [input_size] + list(hidden_layers)
        blocks = []
        for i in range(len(sizes) - 1):
            blocks.append(_ResidualBlock(sizes[i], sizes[i + 1], dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(sizes[-1], output_size)

    def forward(self, x):
        return self.head(self.blocks(x))


class ResidualMLPModel:
    """
    sklearn-compatible wrapper for :class:`ResidualMLPNet`.

    Parameters
    ----------
    hidden_layers:
        Explicit hidden widths when ``layer_schedule='explicit'``.
        Each width is clamped to ``[min_hidden_width, max_hidden_width]``.
        Example: ``[2, 139, 205, 125]`` or ``[302, 230, 510, 24, 125, 20]``.
    layer_schedule:
        ``'explicit'`` — use ``hidden_layers`` as given.
        ``'geometric'`` — compute widths from I/O dims and ``n_hidden_layers``.
    n_hidden_layers:
        Number of hidden layers for the geometric schedule (default 2).
    max_hidden_width:
        Upper cap on any hidden layer width (default 1024).
    min_hidden_width:
        Lower cap on any hidden layer width (default 1).
    n_epochs:
        Training epochs.  Default 200.
    learning_rate:
        Adam learning rate.  Default 1e-3.
    batch_size:
        Mini-batch size.  Default 64.
    dropout_rate:
        Dropout applied inside each residual block.  Default 0.1.
    patience:
        Early-stopping patience (epochs without improvement of the
        early-stopping signal).
    patience_window:
        Rolling-mean window (epochs) for the early-stopping signal.
        1 (default) compares raw per-epoch validation loss — the
        historical behavior; larger windows compare the smoothed loss,
        which is robust to noisy validation curves and stops on true
        saturation rather than on a lucky epoch.
    min_delta:
        Minimum decrease of the (smoothed) validation loss that counts
        as an improvement for the patience counter.
        Default 20.  Set to 0 to disable.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    random_state:
        Seed for PyTorch and numpy RNG.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        layer_schedule: LayerSchedule = "explicit",
        n_hidden_layers: int | None = None,
        max_hidden_width: int = 1024,
        min_hidden_width: int = 1,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        dropout_rate: float = 0.1,
        patience: int = 20,
        patience_window: int = 1,
        min_delta: float = 0.0,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        checkpoint_every_n_epochs: int = 0,
        **_kwargs: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.hidden_layers = hidden_layers or [128, 128]
        self.layer_schedule: LayerSchedule = layer_schedule
        self.n_hidden_layers = n_hidden_layers
        self.max_hidden_width = max(1, int(max_hidden_width))
        self.min_hidden_width = max(1, int(min_hidden_width))
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.patience_window = max(1, int(patience_window))
        self.min_delta = float(min_delta)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file or os.environ.get("SURGE_TRAINING_PROGRESS_JSONL")
        self.checkpoint_every_n_epochs = max(0, int(checkpoint_every_n_epochs))
        self._checkpoint_dir = os.environ.get("SURGE_CHECKPOINT_DIR")

        self._net: Any = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.training_history: list[dict] = []

    def _maybe_save_epoch_checkpoint(self, epoch_num: int) -> None:
        if not self._checkpoint_dir or self.checkpoint_every_n_epochs <= 0:
            return
        if epoch_num % self.checkpoint_every_n_epochs != 0:
            return
        if self._net is None:
            return
        try:
            d = Path(self._checkpoint_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"epoch_{epoch_num:04d}.pt"
            torch.save(
                {
                    "epoch": epoch_num,
                    "state_dict": self._net.state_dict(),
                    "scaler_X": self.scaler_X,
                    "scaler_y": self.scaler_y,
                    "config": {
                        "hidden_layers": self.hidden_layers,
                        "dropout_rate": self.dropout_rate,
                    },
                },
                path,
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ) -> "ResidualMLPModel":
        import numpy as np

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_train = np.asarray(y_train)
        is_1d = y_train.ndim == 1
        if is_1d:
            y_train = y_train.reshape(-1, 1)

        X_s = self.scaler_X.fit_transform(X_train)
        y_s = self.scaler_y.fit_transform(y_train)

        n_in = X_s.shape[1]
        n_out = y_s.shape[1]

        resolved = resolve_hidden_layers(
            n_in=n_in,
            n_out=n_out,
            schedule=self.layer_schedule,
            hidden_layers=self.hidden_layers,
            n_hidden_layers=self.n_hidden_layers,
            min_width=self.min_hidden_width,
            max_width=self.max_hidden_width,
        )
        self._resolved_hidden_layers = resolved

        self._net = ResidualMLPNet(
            input_size=n_in,
            hidden_layers=resolved,
            output_size=n_out,
            dropout=self.dropout_rate,
        ).to(self.device)

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        Xt = torch.from_numpy(X_s.astype("float32"))
        yt = torch.from_numpy(y_s.astype("float32"))
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size,
            shuffle=True,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            y_val_arr = np.asarray(y_val)
            if y_val_arr.ndim == 1:
                y_val_arr = y_val_arr.reshape(-1, 1)
            Xv = torch.from_numpy(
                self.scaler_X.transform(X_val).astype("float32")
            ).to(self.device)
            yv = torch.from_numpy(
                self.scaler_y.transform(y_val_arr).astype("float32")
            ).to(self.device)

        best_val_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0
        from collections import deque
        recent_val: deque = deque(maxlen=self.patience_window)
        best_smoothed = float("inf")
        from ._progress import ProgressList
        self.training_history = ProgressList(
            self.n_epochs, verbose=self.verbose,
            log_file=self.log_file, desc=type(self).__name__,
        )

        for epoch in range(self.n_epochs):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(Xt)

            record: dict = {"epoch": epoch + 1, "train_loss": train_loss}
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_loss = criterion(self._net(Xv), yv).item()
                record["val_loss"] = val_loss
                # best-weight restoration always tracks the raw minimum
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                # early-stopping signal: rolling mean over patience_window
                recent_val.append(val_loss)
                smoothed = sum(recent_val) / len(recent_val)
                record["val_loss_smoothed"] = smoothed
                if smoothed < best_smoothed - self.min_delta:
                    best_smoothed = smoothed
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.patience > 0 and no_improve >= self.patience:
                        record["early_stop"] = True
                        self.training_history.append(record)
                        break
            self.training_history.append(record)
            self._maybe_save_epoch_checkpoint(epoch + 1)

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.training_history.close()
        self.is_fitted = True
        self._is_1d = is_1d
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X) -> Any:
        import numpy as np

        if not self.is_fitted or self._net is None:
            raise ValueError("Model not fitted yet")
        self._net.eval()
        Xs = self.scaler_X.transform(np.asarray(X))
        Xt = torch.from_numpy(Xs.astype("float32")).to(self.device)
        with torch.no_grad():
            out = self._net(Xt).cpu().numpy()
        out = self.scaler_y.inverse_transform(out)
        return out.ravel() if self._is_1d else out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({
            "state_dict": self._net.state_dict() if self._net is not None else None,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "config": {
                "hidden_layers": getattr(self, "_resolved_hidden_layers", self.hidden_layers),
                "layer_schedule": self.layer_schedule,
                "n_hidden_layers": self.n_hidden_layers,
                "max_hidden_width": self.max_hidden_width,
                "min_hidden_width": self.min_hidden_width,
                "dropout_rate": self.dropout_rate,
                "n_in": self._net.blocks[0].linear1.in_features if self._net else None,
                "n_out": self._net.head.out_features if self._net else None,
            },
            "is_fitted": self.is_fitted,
            "_is_1d": getattr(self, "_is_1d", True),
        }, path)

    def load(self, path: str) -> None:
        import joblib

        data = joblib.load(path)
        cfg = data["config"]
        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.is_fitted = data["is_fitted"]
        self._is_1d = data.get("_is_1d", True)
        self.layer_schedule = cfg.get("layer_schedule", "explicit")
        self.n_hidden_layers = cfg.get("n_hidden_layers")
        self.max_hidden_width = cfg.get("max_hidden_width", 1024)
        self.min_hidden_width = cfg.get("min_hidden_width", 1)
        self.hidden_layers = cfg["hidden_layers"]
        self._resolved_hidden_layers = cfg["hidden_layers"]
        if data["state_dict"] is not None and cfg["n_in"] is not None:
            self._net = ResidualMLPNet(
                input_size=cfg["n_in"],
                hidden_layers=cfg["hidden_layers"],
                output_size=cfg["n_out"],
                dropout=cfg["dropout_rate"],
            ).to(self.device)
            self._net.load_state_dict(data["state_dict"])
