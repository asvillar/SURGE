"""LSTM / GRU surrogate backends for temporal / sequence data.

Both models share the same sklearn-compatible interface.  Input arrays
are expected in **flat** format: ``X.shape = (n_samples, T_in * n_state)``
and ``y.shape = (n_samples, T_out * n_state)``.  The model internally
reshapes to ``(batch, T, n_state)`` for the recurrent layers.

If the data is passed already shaped ``(n_samples, T, n_state)`` the
adapter reshapes it to flat before fitting so that tabular models in the
same leaderboard can consume the same arrays.

Architecture
------------
- Encoder: stacked LSTM/GRU processes the input window.
- Decoder (direct): single linear layer maps the last hidden state to
  the flattened output ``T_out * n_state``.
- ``StandardScaler`` on the flattened X and y.
- Early-stopping on validation loss.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.rnn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _RNNNet(nn.Module if TORCH_AVAILABLE else object):
    """Encoder-decoder RNN: input window → flat output prediction."""

    def __init__(
        self,
        n_state: int,
        T_in: int,
        T_out: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        cell: str,  # "lstm" | "gru"
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.n_state = n_state
        self.T_in = T_in
        self.T_out = T_out
        self.hidden_size = hidden_size
        cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = cls(
            input_size=n_state,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, T_out * n_state)

    def forward(self, x):
        # x: (B, T_in, n_state)
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # (B, hidden)
        return self.head(last)  # (B, T_out * n_state)


class _RNNModel:
    """
    Base sklearn-compatible wrapper for :class:`_RNNNet`.

    Parameters
    ----------
    n_state:
        Number of state variables per time step (e.g. 3 for Lorenz-63).
        Inferred from data if ``None``.
    T_in:
        Input context length (time steps). Inferred from ``X.shape[1] // n_state``.
    T_out:
        Prediction horizon. Inferred from ``y.shape[1] // n_state``.
    hidden_size, n_layers:
        RNN architecture.
    n_epochs, learning_rate, batch_size, patience:
        Training knobs.
    """

    _cell: str = "lstm"  # overridden by subclasses

    def __init__(
        self,
        n_state: Optional[int] = None,
        T_in: Optional[int] = None,
        T_out: Optional[int] = None,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
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
        self.n_state = n_state
        self.T_in = T_in
        self.T_out = T_out
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
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
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None) -> "_RNNModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        # Ensure at least 2D for shape[1] access throughout.
        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]
        B, feat_x = X_arr.shape[0], X_arr.shape[1] if X_arr.ndim == 2 else X_arr.shape[1] * X_arr.shape[2]
        if X_arr.ndim == 3:
            X_arr = X_arr.reshape(B, -1)
        if y_arr.ndim == 3:
            y_arr = y_arr.reshape(y_arr.shape[0], -1)

        n_state = self.n_state or _infer_n_state(X_arr, y_arr)
        T_in = self.T_in or (X_arr.shape[1] // n_state)
        T_out = self.T_out or (y_arr.shape[1] // n_state)
        self._n_state = n_state
        self._T_in = T_in
        self._T_out = T_out

        Xs = self.scaler_X.fit_transform(X_arr)
        ys = self.scaler_y.fit_transform(y_arr)

        self._net = _RNNNet(
            n_state=n_state, T_in=T_in, T_out=T_out,
            hidden_size=self.hidden_size, n_layers=self.n_layers,
            dropout=self.dropout, cell=self._cell,
        ).to(self.device)

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        Xt = torch.from_numpy(Xs.reshape(-1, T_in, n_state).astype("float32"))
        yt = torch.from_numpy(ys.astype("float32"))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = np.asarray(X_val, dtype=float)
            yv = np.asarray(y_val, dtype=float)
            if Xv.ndim == 3:
                Xv = Xv.reshape(Xv.shape[0], -1)
            if yv.ndim == 3:
                yv = yv.reshape(yv.shape[0], -1)
            Xvs = self.scaler_X.transform(Xv)
            yvs = self.scaler_y.transform(yv)
            Xvt = torch.from_numpy(Xvs.reshape(-1, T_in, n_state).astype("float32")).to(self.device)
            yvt = torch.from_numpy(yvs.astype("float32")).to(self.device)

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
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 3:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        Xs = self.scaler_X.transform(X_arr)
        Xt = torch.from_numpy(
            Xs.reshape(-1, self._T_in, self._n_state).astype("float32")
        ).to(self.device)
        with torch.no_grad():
            out = self._net(Xt).cpu().numpy()
        result = self.scaler_y.inverse_transform(out)
        # If original y was 1D (scalar), ravel the output.
        if self._T_out == 1 and self._n_state == 1:
            return result.ravel()
        return result

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
            "config": {
                "n_state": self._n_state, "T_in": self._T_in, "T_out": self._T_out,
                "hidden_size": self.hidden_size, "n_layers": self.n_layers,
                "dropout": self.dropout, "cell": self._cell,
            },
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.is_fitted = data["is_fitted"]
        cfg = data["config"]
        self._n_state, self._T_in, self._T_out = cfg["n_state"], cfg["T_in"], cfg["T_out"]
        self.hidden_size = cfg["hidden_size"]
        self.n_layers = cfg["n_layers"]
        if data["state_dict"] is not None:
            self._net = _RNNNet(**cfg).to(self.device)
            self._net.load_state_dict(data["state_dict"])


def _infer_n_state(X: np.ndarray, y: np.ndarray) -> int:
    """Best-effort n_state inference from GCD of feature counts."""
    from math import gcd
    g = gcd(X.shape[1], y.shape[1])
    # Return g if it's a plausible state dimension (<= 20).
    return g if 1 <= g <= 20 else 1


# ------------------------------------------------------------------
# Concrete subclasses
# ------------------------------------------------------------------


class LSTMModel(_RNNModel):
    """LSTM surrogate. See :class:`_RNNModel` for parameters."""
    _cell = "lstm"


class GRUModel(_RNNModel):
    """GRU surrogate. See :class:`_RNNModel` for parameters."""
    _cell = "gru"
