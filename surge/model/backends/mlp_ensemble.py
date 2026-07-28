"""Deep MLP ensemble backend — matches the ConStellaration paper architecture.

Reference:
    Cadena et al. (2025) arXiv:2506.19583 — Appendix A.4
    "ensemble model of ten MLPs with three layers, 256 hidden units, tanh activations"
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.mlp_ensemble")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


class _MLP(nn.Module if TORCH_AVAILABLE else object):
    """Simple fully-connected MLP with configurable depth, width, and activation."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        output_size: int,
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation.lower()]
        layers: list[nn.Module] = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class MLPEnsembleModel:
    """Ensemble of *n_ensembles* independent MLPs for regression.

    Matches the surrogate baseline in the ConStellaration paper (Cadena et al. 2025):

    * 10 ensemble members
    * 3 hidden layers of 256 units each
    * tanh activation
    * MSE loss, Adam optimiser
    * early stopping (patience=30 by default)

    Each member is trained from a different random seed, giving a mean prediction
    and an ensemble standard-deviation estimate (epistemic uncertainty).

    Parameters
    ----------
    n_ensembles : int
        Number of independent MLP members (default 10).
    hidden_dim : int
        Hidden-layer width (default 256).
    n_layers : int
        Number of hidden layers (default 3).
    activation : str
        Activation: ``"tanh"`` | ``"relu"`` | ``"gelu"`` (default ``"tanh"``).
    n_epochs : int
        Max training epochs per member (default 200).
    learning_rate : float
        Adam learning rate (default 1e-3).
    batch_size : int
        Mini-batch size (default 256).
    patience : int
        Early-stopping patience (0 = disabled, default 30).
    dropout : float
        Dropout rate (default 0.0, matching paper).
    device : str | None
        ``"cpu"``, ``"cuda"``, or None (auto-detect).
    random_state : int
        Base seed; member *i* uses ``random_state + i``.
    """

    def __init__(
        self,
        n_ensembles: int = 10,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "tanh",
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        patience: int = 30,
        dropout: float = 0.0,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_kwargs: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        self.n_ensembles = n_ensembles
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file

        self._members: list[_MLP] = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self._is_1d = True
        self.training_history: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_member(
        self,
        net: "_MLP",
        X_t: "torch.Tensor",
        y_t: "torch.Tensor",
        X_v: "torch.Tensor | None",
        y_v: "torch.Tensor | None",
        member_idx: int,
    ) -> list[dict]:
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        n = X_t.shape[0]
        last_record: dict | None = None
        best_val_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for epoch in range(self.n_epochs):
            net.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            batches = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            epoch_loss /= max(batches, 1)

            record: dict = {"epoch": epoch, "train_loss": epoch_loss, "member": member_idx}

            if X_v is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = criterion(net(X_v), y_v).item()
                record["val_loss"] = val_loss
                if val_loss < best_val_loss - 1e-7:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if self.patience > 0 and no_improve >= self.patience:
                    _LOG.debug("Member %d early-stopped at epoch %d", member_idx, epoch)
                    record["early_stop"] = True
                    self.training_history.append(record)
                    last_record = record
                    break

            self.training_history.append(record)
            last_record = record

        if best_state is not None:
            net.load_state_dict(best_state)
        return last_record

    # ------------------------------------------------------------------
    # Public sklearn-like interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ) -> "MLPEnsembleModel":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        self._is_1d = y_train.ndim == 1
        if self._is_1d:
            y_train = y_train.reshape(-1, 1)

        X_s = self.scaler_X.fit_transform(X_train).astype("float32")
        y_s = self.scaler_y.fit_transform(y_train).astype("float32")

        n_in = X_s.shape[1]
        n_out = y_s.shape[1]

        Xt = torch.from_numpy(X_s).to(self.device)
        yt = torch.from_numpy(y_s).to(self.device)

        Xv = yv = None
        if X_val is not None and y_val is not None:
            y_val = np.asarray(y_val, dtype=float)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            Xv = torch.from_numpy(
                self.scaler_X.transform(np.asarray(X_val, dtype=float)).astype("float32")
            ).to(self.device)
            yv = torch.from_numpy(
                self.scaler_y.transform(y_val).astype("float32")
            ).to(self.device)

        self._members = []
        from ._progress import ProgressList

        # Stream per-epoch records to a tail-friendly JSONL log (and/or tqdm) so
        # training can be monitored live via surge.viz.training.load_training_history.
        self.training_history = ProgressList(
            total_epochs=self.n_epochs * self.n_ensembles,
            verbose=self.verbose,
            log_file=self.log_file,
            desc=f"{type(self).__name__}",
        )

        for m in range(self.n_ensembles):
            torch.manual_seed(self.random_state + m)
            net = _MLP(
                n_in, self.hidden_dim, self.n_layers, n_out, self.activation, self.dropout
            ).to(self.device)
            last = self._train_member(net, Xt, yt, Xv, yv, member_idx=m)
            self._members.append(net)
            if self.verbose and last is not None:
                loss_val = last.get("val_loss", last["train_loss"])
                _LOG.info("Member %d/%d done — loss=%.4f", m + 1, self.n_ensembles, loss_val)

        self.training_history.close()
        self.is_fitted = True
        return self

    def _predict_raw(self, X) -> np.ndarray:
        """Return (n_ensembles, n_samples, n_out) array in original scale."""
        X_s = self.scaler_X.transform(np.asarray(X, dtype=float)).astype("float32")
        Xt = torch.from_numpy(X_s).to(self.device)
        preds = []
        for net in self._members:
            net.eval()
            with torch.no_grad():
                p_scaled = net(Xt).cpu().numpy()  # (n, n_out)
            p_orig = self.scaler_y.inverse_transform(p_scaled)
            preds.append(p_orig)
        return np.stack(preds, axis=0)  # (n_ensembles, n, n_out)

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        preds = self._predict_raw(X)          # (n_ens, n, n_out)
        mean = preds.mean(axis=0)             # (n, n_out)
        if self._is_1d:
            return mean.ravel()
        return mean

    def predict_with_uncertainty(self, X, **_) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) across ensemble members."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_with_uncertainty()")
        preds = self._predict_raw(X)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        if self._is_1d:
            return mean.ravel(), std.ravel()
        return mean, std
