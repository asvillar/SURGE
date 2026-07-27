"""PyTorch MLP classifier backend for SURGE.

Reuses the same training loop structure as :mod:`surge.model.pytorch_impl`
but with:
- Output head: ``nn.Linear(hidden, n_classes)`` (logits, no activation)
- Loss: ``nn.CrossEntropyLoss`` (multi-class) / ``nn.BCEWithLogitsLoss`` (binary)
- ``predict()`` → ``argmax(logits, dim=1)`` (class indices)
- ``predict_proba()`` → ``softmax(logits, dim=1)``
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.mlp_classifier")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _MLPClassifierNet(nn.Module if TORCH_AVAILABLE else object):
    """Plain MLP with a softmax-compatible logit head."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        layers: list[Any] = []
        prev = input_size
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifierModel:
    """
    sklearn-compatible MLP classifier backed by PyTorch.

    Parameters
    ----------
    hidden_layers:
        Hidden layer sizes.  Default ``[128, 64]``.
    n_epochs, learning_rate, batch_size, dropout_rate:
        Standard training knobs.
    patience:
        Early-stopping patience on val-loss.  ``0`` disables.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        n_epochs: int = 150,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        dropout_rate: float = 0.1,
        patience: int = 15,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_kwargs: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.hidden_layers = hidden_layers or [128, 64]
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file

        self._net: Any = None
        self.scaler_X = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.n_classes_: Optional[int] = None
        self.is_fitted = False
        self.training_history: list[dict] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "MLPClassifierModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_enc = self.label_encoder.fit_transform(np.asarray(y_train))
        self.n_classes_ = len(self.label_encoder.classes_)
        X_s = self.scaler_X.fit_transform(np.asarray(X_train))

        n_in = X_s.shape[1]
        binary = self.n_classes_ == 2
        n_out = 1 if binary else self.n_classes_

        self._net = _MLPClassifierNet(
            input_size=n_in,
            hidden_layers=self.hidden_layers,
            n_classes=n_out,
            dropout=self.dropout_rate,
        ).to(self.device)

        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X_s.astype("float32"))
        if binary:
            yt = torch.from_numpy(y_enc.astype("float32")).unsqueeze(1)
        else:
            yt = torch.from_numpy(y_enc.astype("int64"))

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            y_val_enc = self.label_encoder.transform(np.asarray(y_val))
            Xv = torch.from_numpy(
                self.scaler_X.transform(np.asarray(X_val)).astype("float32")
            ).to(self.device)
            if binary:
                yv = torch.from_numpy(y_val_enc.astype("float32")).unsqueeze(1).to(self.device)
            else:
                yv = torch.from_numpy(y_val_enc.astype("int64")).to(self.device)

        best_val_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0
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
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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
        self._binary = binary
        return self

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        if self._binary:
            idx = (proba[:, 1] >= 0.5).astype(int)
        else:
            idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(idx)

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError("Model not fitted")
        self._net.eval()
        Xs = self.scaler_X.transform(np.asarray(X))
        Xt = torch.from_numpy(Xs.astype("float32")).to(self.device)
        with torch.no_grad():
            logits = self._net(Xt).cpu().numpy()
        if self._binary:
            p1 = 1 / (1 + np.exp(-logits.ravel()))
            return np.column_stack([1 - p1, p1])
        # softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X,
            "label_encoder": self.label_encoder,
            "config": {
                "hidden_layers": self.hidden_layers,
                "dropout_rate": self.dropout_rate,
                "n_in": self._net.net[0].in_features if self._net else None,
                "n_out": list(self._net.net.children())[-1].out_features if self._net else None,
            },
            "is_fitted": self.is_fitted,
            "_binary": getattr(self, "_binary", False),
        }, path)

    def load(self, path: str) -> None:
        import joblib

        data = joblib.load(path)
        self.scaler_X = data["scaler_X"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = data["is_fitted"]
        self._binary = data.get("_binary", False)
        cfg = data["config"]
        if data["state_dict"] is not None:
            self._net = _MLPClassifierNet(
                input_size=cfg["n_in"],
                hidden_layers=cfg["hidden_layers"],
                n_classes=cfg["n_out"],
                dropout=cfg["dropout_rate"],
            ).to(self.device)
            self._net.load_state_dict(data["state_dict"])
