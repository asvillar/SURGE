"""Variational Autoencoder (VAE) with a regression / classification head.

Architecture
------------
Encoder MLP:  x → [μ(x), log σ²(x)]  (latent_dim each)
Reparameterize: z = μ + ε·σ,  ε ~ N(0,I)
Decoder MLP:  z → x̂  (reconstruction)
Head MLP:     z → ŷ  (task prediction)

Loss = reconstruction MSE + β·KL divergence + task loss

Exposes ``predict_with_uncertainty(X)`` by sampling from the posterior.

Reference
---------
Kingma & Welling (2014) "Auto-Encoding Variational Bayes" ICLR 2014.
https://arxiv.org/abs/1312.6114
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

_LOG = logging.getLogger("surge.pytorch.vae")

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
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU() if act is None else act())
    return nn.Sequential(*layers)


class _VAENet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self, n_in: int, latent_dim: int, hidden_dim: int, n_out: int, task: str
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.task = task
        h = hidden_dim
        # Encoder
        self.enc_shared = _mlp([n_in, h, h])
        self.enc_act = nn.ReLU()
        self.mu_layer = nn.Linear(h, latent_dim)
        self.logvar_layer = nn.Linear(h, latent_dim)
        # Decoder
        self.decoder = _mlp([latent_dim, h, h, n_in])
        # Task head
        self.head = nn.Linear(latent_dim, n_out)

    def encode(self, x):
        h = self.enc_act(self.enc_shared(x))
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return y_hat, x_hat, mu, logvar


class VAEModel:
    """VAE latent regression / classification surrogate.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space z.
    hidden_dim : int
        Width of encoder/decoder/head hidden layers.
    beta : float
        Weight of the KL divergence term in the loss.
    task : str
        ``"regression"`` or ``"classification"``.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        beta: float = 1.0,
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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
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

    def fit(self, X, y, **_: Any) -> "VAEModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        self._n_outputs = y_arr.shape[1]

        Xs = self.scaler_X.fit_transform(X_arr)
        if self.task == "regression":
            ys = self.scaler_y.fit_transform(y_arr)
        else:
            ys = y_arr

        n_in = Xs.shape[1]
        n_out = self._n_outputs

        self._net = _VAENet(n_in, self.latent_dim, self.hidden_dim, n_out, self.task).to(self.device)
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)

        Xt = torch.from_numpy(Xs.astype(np.float32))
        if self.task == "regression":
            yt = torch.from_numpy(ys.astype(np.float32))
        else:
            yt = torch.from_numpy(ys.ravel().astype(np.int64))

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        recon_loss = nn.MSELoss(reduction="sum")
        task_loss_fn = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        best_loss = float("inf")
        no_improve = 0
        n = len(Xt)
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
                y_hat, x_hat, mu, logvar = self._net(xb)
                rl = recon_loss(x_hat, xb) / len(xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                tl = task_loss_fn(y_hat, yb)
                loss = rl + self.beta * kl + tl
                loss.backward()
                optimizer.step()
                eloss += loss.item() * len(xb)
            epoch_loss = eloss / n
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
            mu, _ = self._net.encode(Xt)
            y_hat = self._net.head(mu).cpu().numpy()
        if self.task == "regression":
            out = self.scaler_y.inverse_transform(y_hat)
            return out.ravel() if self._n_outputs == 1 else out
        return y_hat.argmax(axis=1)

    def predict_with_uncertainty(self, X, n_samples: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Sample from posterior and return mean + std."""
        self._net.train()  # enable dropout (none here, but keep consistent)
        Xs = self.scaler_X.transform(np.asarray(X, dtype=np.float64))
        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, logvar = self._net.encode(Xt)
                z = self._net.reparameterize(mu, logvar)
                y_hat = self._net.head(z).cpu().numpy()
                samples.append(y_hat)
        self._net.eval()
        stacked = np.stack(samples)  # (n_samples, B, n_out)
        mean = stacked.mean(0)
        std = stacked.std(0)
        if self.task == "regression":
            mean = self.scaler_y.inverse_transform(mean)
            std = std * self.scaler_y.scale_
        return mean.ravel() if self._n_outputs == 1 else mean, \
               std.ravel() if self._n_outputs == 1 else std

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "latent_dim": self.latent_dim, "hidden_dim": self.hidden_dim,
                "beta": self.beta, "task": self.task, "n_outputs": self._n_outputs,
                "n_in": self.scaler_X.n_features_in_,
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
        self._net = _VAENet(cfg["n_in"], cfg["latent_dim"], cfg["hidden_dim"], cfg["n_outputs"], cfg["task"]).to(self.device)
        if d["net_state"]:
            self._net.load_state_dict(d["net_state"])
