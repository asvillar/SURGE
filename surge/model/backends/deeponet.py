"""Deep Operator Network (DeepONet) — Lu et al. 2021.

Implements the vanilla DeepONet for 1-D → 1-D operator learning.

Architecture
------------
- **Branch network**: encodes the input function ``u`` sampled at ``p``
  sensor points → latent vector of size ``n_basis``.
- **Trunk network**: encodes the query/output location ``y ∈ R^{d_out}``
  → latent vector of size ``n_basis``.
- **Output**: dot product of branch and trunk outputs, summed over the
  basis dimension, plus a bias term.

SURGE interface
---------------
Flat input arrays:
  ``X.shape = (n_samples, n_sensors)``    — input function at sensor pts
  ``y.shape = (n_samples, n_query_pts)``  — output function at query pts

Query points are assumed to be the same for every sample and are
constructed as a uniform grid on ``[0, 1]`` if not supplied.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

_LOG = logging.getLogger("surge.pytorch.deeponet")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


def _mlp(dims: list[int], act=None) -> "nn.Sequential":
    """Build a simple fully-connected MLP."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act() if act else nn.Tanh())
    return nn.Sequential(*layers)


class DeepONet(nn.Module if TORCH_AVAILABLE else object):
    """Vanilla DeepONet."""

    def __init__(
        self,
        n_sensors: int,
        n_query: int,
        n_basis: int = 64,
        branch_width: int = 128,
        trunk_width: int = 128,
        n_hidden: int = 3,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        branch_dims = [n_sensors] + [branch_width] * n_hidden + [n_basis]
        trunk_dims = [1] + [trunk_width] * n_hidden + [n_basis]
        self.branch = _mlp(branch_dims, act=nn.Tanh)
        self.trunk = _mlp(trunk_dims, act=nn.Tanh)
        # Trunk includes activation on output so last activation is applied.
        self.trunk_act = nn.Tanh()
        self.bias = nn.Parameter(torch.zeros(1))
        self.n_query = n_query

    def forward(self, u: "torch.Tensor", y_pts: "torch.Tensor") -> "torch.Tensor":
        """
        Parameters
        ----------
        u:      (B, n_sensors) — input function values
        y_pts:  (n_query, 1) — query point locations

        Returns
        -------
        (B, n_query) — predicted output function
        """
        bk = self.branch(u)  # (B, n_basis)
        tk = self.trunk_act(self.trunk(y_pts))  # (n_query, n_basis)
        return torch.einsum("bn,qn->bq", bk, tk) + self.bias


class DeepONetModel:
    """
    sklearn-compatible wrapper for :class:`DeepONet`.

    Parameters
    ----------
    n_sensors:
        Number of sensor points (= ``X.shape[1]``).  Inferred from data.
    n_query:
        Number of query points (= ``y.shape[1]``).  Inferred from data.
    n_basis, branch_width, trunk_width, n_hidden:
        Architecture knobs.
    """

    def __init__(
        self,
        n_basis: int = 64,
        branch_width: int = 128,
        trunk_width: int = 128,
        n_hidden: int = 3,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.n_basis = n_basis
        self.branch_width = branch_width
        self.trunk_width = trunk_width
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state

        self._net: Any = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.training_history: list[dict] = []

    def fit(self, X, y) -> "DeepONetModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        B, n_sensors = X_arr.shape
        n_query = y_arr.shape[1]
        self._n_sensors = n_sensors
        self._n_query = n_query

        Xs = self.scaler_X.fit_transform(X_arr)
        ys = self.scaler_y.fit_transform(y_arr)

        self._net = DeepONet(
            n_sensors=n_sensors, n_query=n_query, n_basis=self.n_basis,
            branch_width=self.branch_width, trunk_width=self.trunk_width,
            n_hidden=self.n_hidden,
        ).to(self.device)

        # Build query point grid once: uniform on [0, 1].
        y_pts = torch.linspace(0, 1, n_query).unsqueeze(1).to(self.device)  # (n_query, 1)

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        Xt = torch.from_numpy(Xs.astype("float32"))
        yt = torch.from_numpy(ys.astype("float32"))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        self.training_history = []

        for epoch in range(self.n_epochs):
            self._net.train()
            eloss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb, y_pts), yb)
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
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError("Model not fitted")
        self._net.eval()
        X_arr = np.asarray(X, dtype=float)
        Xs = self.scaler_X.transform(X_arr)
        Xt = torch.from_numpy(Xs.astype("float32")).to(self.device)
        y_pts = torch.linspace(0, 1, self._n_query).unsqueeze(1).to(self.device)
        with torch.no_grad():
            out = self._net(Xt, y_pts).cpu().numpy()
        return self.scaler_y.inverse_transform(out)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
            "config": {
                "n_sensors": self._n_sensors, "n_query": self._n_query,
                "n_basis": self.n_basis, "branch_width": self.branch_width,
                "trunk_width": self.trunk_width, "n_hidden": self.n_hidden,
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
        self._n_sensors, self._n_query = cfg["n_sensors"], cfg["n_query"]
        if data["state_dict"] is not None:
            self._net = DeepONet(**cfg).to(self.device)
            self._net.load_state_dict(data["state_dict"])
