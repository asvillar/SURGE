"""Kolmogorov-Arnold Network (KAN) backend.

Uses ``efficient-kan`` — a pure-PyTorch implementation of B-spline KAN layers:
    pip install git+https://github.com/Blealtan/efficient-kan

Architecture
------------
Stacked ``KANLinear`` layers with configurable:
- ``hidden_dims`` — list of hidden widths (e.g. [64, 64])
- ``grid_size`` — number of B-spline grid points per activation (default 5)
- ``spline_order`` — B-spline degree (default 3)

Training loop identical to other SURGE PyTorch backends: Adam optimiser,
``StandardScaler`` on inputs, early stopping on validation MSE.

Reference
---------
Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks" arXiv:2404.19756.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.kan")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore

try:
    from efficient_kan import KAN as _EfficientKAN  # type: ignore
    EFFICIENT_KAN_AVAILABLE = True
except ImportError:
    EFFICIENT_KAN_AVAILABLE = False
    _EfficientKAN = None  # type: ignore


def _build_kan_net(layer_dims: list[int], grid_size: int, spline_order: int):
    """Return an efficient-KAN network for the given layer dimensions."""
    if not EFFICIENT_KAN_AVAILABLE:
        raise ImportError("efficient-kan required: pip install git+https://github.com/Blealtan/efficient-kan")
    return _EfficientKAN(layer_dims, grid_size=grid_size, spline_order=spline_order)


class KANModel:
    """KAN regression / classification surrogate.

    Parameters
    ----------
    hidden_dims : list[int]
        Hidden layer widths. Input and output dims are inferred from data.
    grid_size : int
        Number of B-spline grid points (controls spline resolution).
    spline_order : int
        B-spline degree (3 = cubic).
    task : str
        ``"regression"`` or ``"classification"``.
    n_epochs, learning_rate, batch_size, patience :
        Training knobs.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        grid_size: int = 5,
        spline_order: int = 3,
        task: str = "regression",
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE or not EFFICIENT_KAN_AVAILABLE:
            raise ImportError(
                "PyTorch + efficient-kan required. "
                "pip install torch git+https://github.com/Blealtan/efficient-kan"
            )
        self.hidden_dims = hidden_dims or [64, 64]
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.task = task
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._net: Any = None
        self.is_fitted = False
        self._n_outputs = 1
        self.training_history: list[dict] = []

    def fit(self, X, y, **_: Any) -> "KANModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        n_in = X_arr.shape[1]

        Xs = self.scaler_X.fit_transform(X_arr)
        if self.task == "regression":
            self._n_outputs = y_arr.shape[1]
            ys = self.scaler_y.fit_transform(y_arr)
        else:
            # For classification, n_outputs = number of classes
            y_flat = y_arr.ravel().astype(np.int64)
            self._n_outputs = int(y_flat.max()) + 1
            ys = y_arr  # class indices — no scaling
        n_out = self._n_outputs

        layer_dims = [n_in] + list(self.hidden_dims) + [n_out]
        self._net = _build_kan_net(layer_dims, self.grid_size, self.spline_order).to(self.device)

        if self.task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)

        Xt = torch.from_numpy(Xs.astype(np.float32))
        if self.task == "regression":
            yt = torch.from_numpy(ys.astype(np.float32))
        else:
            yt = torch.from_numpy(ys.ravel().astype(np.int64))

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
                pred = self._net(xb)
                if self.task == "regression" and pred.shape[1] == 1:
                    pred = pred  # (B, 1)
                loss = criterion(pred, yb)
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
        Xs = self.scaler_X.transform(np.asarray(X, dtype=np.float64))
        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self._net(Xt).cpu().numpy()
        if self.task == "regression":
            out_inv = self.scaler_y.inverse_transform(out)
            return out_inv.ravel() if self._n_outputs == 1 else out_inv
        # classification: argmax
        return out.argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        self._net.eval()
        Xs = self.scaler_X.transform(np.asarray(X, dtype=np.float64))
        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._net(Xt)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "hidden_dims": self.hidden_dims, "grid_size": self.grid_size,
                "spline_order": self.spline_order, "task": self.task,
                "n_outputs": self._n_outputs,
            },
            "net_state": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        self.scaler_X = d["scaler_X"]
        self.scaler_y = d["scaler_y"]
        self.is_fitted = d["is_fitted"]
        cfg = d["config"]
        self._n_outputs = cfg["n_outputs"]
        self.hidden_dims = cfg["hidden_dims"]
        self.grid_size = cfg["grid_size"]
        self.spline_order = cfg["spline_order"]
        self.task = cfg["task"]
        if d["net_state"] is not None:
            n_in = self.scaler_X.n_features_in_
            layer_dims = [n_in] + list(self.hidden_dims) + [self._n_outputs]
            self._net = _build_kan_net(layer_dims, self.grid_size, self.spline_order).to(self.device)
            self._net.load_state_dict(d["net_state"])
