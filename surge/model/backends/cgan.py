"""Conditional Generative Adversarial Network (CGAN) for 1D fields.

Generator maps ``(noise z, condition x) → field y``.
Discriminator scores ``(x, y)`` pairs.

At inference, ``predict(X)`` averages multiple generator samples.

Reference
---------
Mirza & Osindero (2014) "Conditional Generative Adversarial Nets"
arXiv:1411.1784. https://arxiv.org/abs/1411.1784
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

_LOG = logging.getLogger("surge.pytorch.cgan")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


def _mlp_net(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> "nn.Sequential":
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2)]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class CGANModel:
    """Conditional GAN for 1D field-to-field surrogate.

    Parameters
    ----------
    latent_dim : int
        Noise vector dimension.
    hidden_channels : int
        Hidden layer width for generator and discriminator.
    n_gen_layers, n_disc_layers : int
        Depth of generator / discriminator MLPs.
    learning_rate_g, learning_rate_d : float
        Separate learning rates for G and D.
    n_predict_samples : int
        Number of generator samples averaged at inference.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_channels: int = 128,
        n_gen_layers: int = 3,
        n_disc_layers: int = 3,
        learning_rate_g: float = 2e-4,
        learning_rate_d: float = 2e-4,
        n_epochs: int = 200,
        batch_size: int = 64,
        n_predict_samples: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.n_gen_layers = n_gen_layers
        self.n_disc_layers = n_disc_layers
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_predict_samples = n_predict_samples
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._G: Any = None
        self._D: Any = None
        self.is_fitted = False
        self._nx: int = 0
        self.training_history: list[dict] = []

    def fit(self, X, y, **_: Any) -> "CGANModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        self._nx = y_arr.shape[1]
        nx = self._nx
        n_cond = X_arr.shape[1]

        Xs = self.scaler_X.fit_transform(X_arr).astype(np.float32)
        ys = self.scaler_y.fit_transform(y_arr).astype(np.float32)

        h = self.hidden_channels
        # Generator: (z + cond) → y
        self._G = _mlp_net(self.latent_dim + n_cond, nx, h, self.n_gen_layers).to(self.device)
        # Discriminator: (cond + y) → scalar
        self._D = _mlp_net(n_cond + nx, 1, h, self.n_disc_layers).to(self.device)
        # Sigmoid at end of D
        self._D = nn.Sequential(self._D, nn.Sigmoid())

        opt_G = optim.Adam(self._G.parameters(), lr=self.learning_rate_g, betas=(0.5, 0.999))
        opt_D = optim.Adam(self._D.parameters(), lr=self.learning_rate_d, betas=(0.5, 0.999))
        bce = nn.BCELoss()

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(ys)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        from ._progress import ProgressList
        self.training_history = ProgressList(
            self.n_epochs, verbose=self.verbose,
            log_file=self.log_file, desc=type(self).__name__,
        )
        for epoch in range(self.n_epochs):
            g_loss_e = 0.0
            d_loss_e = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B = xb.shape[0]
                real_labels = torch.ones(B, 1, device=self.device)
                fake_labels = torch.zeros(B, 1, device=self.device)

                # --- Train D ---
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_y = self._G(torch.cat([z, xb], dim=1)).detach()
                d_real = bce(self._D(torch.cat([xb, yb], dim=1)), real_labels)
                d_fake = bce(self._D(torch.cat([xb, fake_y], dim=1)), fake_labels)
                d_loss = 0.5 * (d_real + d_fake)
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

                # --- Train G ---
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_y = self._G(torch.cat([z, xb], dim=1))
                g_loss = bce(self._D(torch.cat([xb, fake_y], dim=1)), real_labels)
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

                g_loss_e += g_loss.item() * B
                d_loss_e += d_loss.item() * B

            n = len(Xt)
            self.training_history.append({
                "epoch": epoch + 1,
                "g_loss": g_loss_e / n,
                "d_loss": d_loss_e / n,
            })

        self.training_history.close()
        self.is_fitted = True
        return self

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._G is None:
            raise ValueError("Not fitted")
        self._G.eval()
        Xs = self.scaler_X.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        Xt = torch.from_numpy(Xs).to(self.device)
        B = Xt.shape[0]
        samples = []
        for _ in range(self.n_predict_samples):
            z = torch.randn(B, self.latent_dim, device=self.device)
            y_hat = self._G(torch.cat([z, Xt], dim=1)).cpu().numpy()
            samples.append(y_hat)
        mean = np.stack(samples).mean(0)  # (B, nx)
        return self.scaler_y.inverse_transform(mean)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "latent_dim": self.latent_dim, "hidden_channels": self.hidden_channels,
                "n_gen_layers": self.n_gen_layers, "n_disc_layers": self.n_disc_layers,
                "nx": self._nx, "n_cond": self.scaler_X.n_features_in_,
            },
            "G_state": self._G.state_dict() if self._G else None,
            "D_state": self._D.state_dict() if self._D else None,
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
        self._nx = cfg["nx"]
        h = self.hidden_channels
        n_cond = cfg["n_cond"]
        self._G = _mlp_net(self.latent_dim + n_cond, cfg["nx"], h, cfg["n_gen_layers"]).to(self.device)
        self._D = nn.Sequential(
            _mlp_net(n_cond + cfg["nx"], 1, h, cfg["n_disc_layers"]), nn.Sigmoid()
        ).to(self.device)
        if d["G_state"]:
            self._G.load_state_dict(d["G_state"])
        if d["D_state"]:
            self._D.load_state_dict(d["D_state"])
