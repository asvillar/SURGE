"""Feature Tokenizer + Transformer (FT-Transformer) for tabular data.

Architecture
------------
1. ``FeatureTokenizer``: maps each numerical feature x_i → Linear(1, d_model)
   producing a token per feature.
2. Prepend learnable ``[CLS]`` token.
3. ``nn.TransformerEncoder`` (pre-norm) with multi-head attention.
4. Regression head (or classification head) applied to the ``[CLS]`` token.

Supports both regression and binary / multi-class classification.

Reference
---------
Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
NeurIPS 2021. https://arxiv.org/abs/2106.11959
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

_LOG = logging.getLogger("surge.pytorch.ft_transformer")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _FeatureTokenizer(nn.Module if TORCH_AVAILABLE else object):
    """Projects each feature dimension independently to d_model."""

    def __init__(self, n_features: int, d_model: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        # one linear per feature: weight (d_model,), bias (d_model,)
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, n_features)
        # out: (B, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class _FTTransformerNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_factor: float = 4.0,
        dropout: float = 0.1,
        n_outputs: int = 1,
        task: str = "regression",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.task = task
        self.n_outputs = n_outputs
        self.tokenizer = _FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * ffn_factor),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        # x: (B, n_features)
        tokens = self.tokenizer(x)                              # (B, n_feat, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)          # (B, 1, d)
        seq = torch.cat([cls, tokens], dim=1)                   # (B, n_feat+1, d)
        out = self.transformer(seq)                             # (B, n_feat+1, d)
        cls_out = self.norm(out[:, 0])                          # (B, d)
        return self.head(cls_out)                               # (B, n_out)


class FTTransformerModel:
    """FT-Transformer surrogate for tabular regression or classification.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_heads : int
        Number of attention heads (must divide d_model).
    n_layers : int
        Number of Transformer encoder layers.
    ffn_factor : float
        Hidden-dim multiplier for the FFN inside each layer.
    dropout : float
        Dropout applied in attention + FFN.
    task : str
        ``"regression"`` or ``"classification"``.
    n_epochs, learning_rate, batch_size, patience :
        Training knobs.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_factor: float = 4.0,
        dropout: float = 0.1,
        task: str = "regression",
        n_epochs: int = 200,
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.d_model = d_model
        self.n_heads = min(n_heads, d_model)  # guard against d_model < n_heads
        self.n_layers = n_layers
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.task = task
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._net: Any = None
        self.is_fitted = False
        self._n_outputs = 1
        self.training_history: list[dict] = []

    def fit(self, X, y, **_: Any) -> "FTTransformerModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        n_features = X_arr.shape[1]

        Xs = self.scaler_X.fit_transform(X_arr)
        if self.task == "regression":
            self._n_outputs = y_arr.shape[1]
            ys = self.scaler_y.fit_transform(y_arr)
        else:
            # For classification, n_outputs = number of classes
            y_flat = y_arr.ravel().astype(np.int64)
            self._n_outputs = int(y_flat.max()) + 1
            ys = y_arr  # class indices

        self._net = _FTTransformerNet(
            n_features=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_factor=self.ffn_factor,
            dropout=self.dropout,
            n_outputs=self._n_outputs,
            task=self.task,
        ).to(self.device)

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
                if self.task == "regression":
                    loss = criterion(pred, yb)
                else:
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
                "d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "ffn_factor": self.ffn_factor,
                "dropout": self.dropout, "task": self.task,
                "n_outputs": self._n_outputs,
                "n_features": self.scaler_X.n_features_in_,
            },
            "net_state": self._net.state_dict() if self._net else None,
            "scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        cfg = d["config"]
        self.scaler_X = d["scaler_X"]
        self.scaler_y = d["scaler_y"]
        self.is_fitted = d["is_fitted"]
        self._n_outputs = cfg["n_outputs"]
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_layers = cfg["n_layers"]
        self.task = cfg["task"]
        if d["net_state"] is not None:
            self._net = _FTTransformerNet(
                n_features=cfg["n_features"],
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                ffn_factor=cfg.get("ffn_factor", 4.0),
                dropout=cfg.get("dropout", 0.1),
                n_outputs=self._n_outputs,
                task=self.task,
            ).to(self.device)
            self._net.load_state_dict(d["net_state"])
