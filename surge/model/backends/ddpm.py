"""Conditional Denoising Diffusion Probabilistic Model (DDPM) for 1D fields.

A minimal conditional DDPM with a UNet-1D denoising backbone.
Conditioning: the input field u₀(x) is concatenated to the noisy field
at each denoising step.

Usage
-----
- Input X: (B, nx) — initial condition / boundary data
- Target y: (B, nx) — solution field
- Predict: runs reverse diffusion from pure noise conditioned on X

Reference
---------
Ho et al. (2020) "Denoising Diffusion Probabilistic Models" NeurIPS 2020.
https://arxiv.org/abs/2006.11239
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.ddpm")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _SinusoidalEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """Sinusoidal time-step embedding."""

    def __init__(self, dim: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


class _ResBlock1D(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, channels: int, time_emb_dim: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class _UNet1D(nn.Module if TORCH_AVAILABLE else object):
    """Minimal 1D U-Net denoiser for conditional DDPM."""

    def __init__(self, cond_channels: int, hidden_channels: int, time_emb_dim: int = 64) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.time_emb = _SinusoidalEmbedding(time_emb_dim)
        # Input: (noisy_y + condition) => 2*cond_channels (both have same nx)
        in_ch = 2  # noisy field (1 ch) + condition (1 ch)
        ch = hidden_channels
        self.lift = nn.Conv1d(in_ch, ch, 1)
        self.down1 = _ResBlock1D(ch, time_emb_dim)
        self.pool = nn.AvgPool1d(2)
        self.down2 = _ResBlock1D(ch, time_emb_dim)
        self.mid = _ResBlock1D(ch, time_emb_dim)
        self.up2 = nn.ConvTranspose1d(ch, ch, 2, stride=2)
        self.up_blk2 = _ResBlock1D(ch * 2, time_emb_dim)
        self.project = nn.Conv1d(ch * 2, 1, 1)

    def forward(self, x, cond, t):
        # x: (B, 1, nx), cond: (B, 1, nx), t: (B,) int
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        inp = torch.cat([x, cond], dim=1)  # (B, 2, nx)
        h = self.lift(inp)                  # (B, ch, nx)
        h1 = self.down1(h, t_emb)           # (B, ch, nx)
        h2 = self.down2(self.mid(self.pool(h1), t_emb), t_emb)  # (B, ch, nx/2)
        h3 = self.up_blk2(
            torch.cat([self.up2(h2), h1[:, :, :h2.shape[2] * 2]], dim=1), t_emb
        )
        return self.project(h3)             # (B, 1, nx)


class DDPMModel:
    """Conditional DDPM for 1D field-to-field prediction.

    Parameters
    ----------
    n_timesteps : int
        Number of diffusion steps.
    hidden_channels : int
        U-Net channel width.
    beta_start, beta_end : float
        Noise schedule endpoints.
    n_epochs, learning_rate, batch_size :
        Training knobs.
    """

    def __init__(
        self,
        n_timesteps: int = 200,
        hidden_channels: int = 64,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        n_epochs: int = 100,
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
        self.n_timesteps = n_timesteps
        self.hidden_channels = hidden_channels
        self.beta_start = beta_start
        self.beta_end = beta_end
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
        self._betas: Any = None
        self._alphas_cumprod: Any = None
        self.is_fitted = False
        self._nx: int = 0
        self.training_history: list[dict] = []

    def _make_schedule(self) -> None:
        T = self.n_timesteps
        betas = torch.linspace(self.beta_start, self.beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self._betas = betas
        self._alphas_cumprod = alphas_cumprod
        self._alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

    def fit(self, X, y, **_: Any) -> "DDPMModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if y_arr.ndim == 2 and y_arr.shape[1] > 1:
            self._nx = y_arr.shape[1]
        else:
            self._nx = y_arr.shape[1]

        # Fit scalers on flat versions
        self.scaler_X.fit(X_arr)
        self.scaler_y.fit(y_arr)

        Xs = self.scaler_X.transform(X_arr).astype(np.float32)
        ys = self.scaler_y.transform(y_arr).astype(np.float32)

        self._make_schedule()
        self._net = _UNet1D(cond_channels=1, hidden_channels=self.hidden_channels).to(self.device)
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)

        Xt = torch.from_numpy(Xs[:, None, :] if Xs.ndim == 2 else Xs)   # (B, 1, nx)
        yt = torch.from_numpy(ys[:, None, :] if ys.ndim == 2 else ys)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        no_improve = 0
        T = self.n_timesteps
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
                B = yb.shape[0]
                t = torch.randint(0, T, (B,), device=self.device)
                noise = torch.randn_like(yb)
                ac = self._alphas_cumprod[t][:, None, None]
                y_noisy = ac.sqrt() * yb + (1 - ac).sqrt() * noise
                pred_noise = self._net(y_noisy, xb, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                eloss += loss.item() * B
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

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError("Not fitted")
        self._net.eval()
        X_arr = self.scaler_X.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        if X_arr.ndim == 2:
            X_arr = X_arr[:, None, :]
        cond = torch.from_numpy(X_arr).to(self.device)
        B, _, nx = cond.shape
        y = torch.randn(B, 1, nx, device=self.device)
        T = self.n_timesteps
        for t_val in reversed(range(T)):
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            noise_pred = self._net(y, cond, t)
            beta_t = self._betas[t_val]
            alpha_t = 1.0 - beta_t
            ac = self._alphas_cumprod[t_val]
            y = (1.0 / alpha_t.sqrt()) * (y - beta_t / (1 - ac).sqrt() * noise_pred)
            if t_val > 0:
                y = y + beta_t.sqrt() * torch.randn_like(y)
        out = y.squeeze(1).cpu().numpy()  # (B, nx)
        return self.scaler_y.inverse_transform(out)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "n_timesteps": self.n_timesteps, "hidden_channels": self.hidden_channels,
                "beta_start": self.beta_start, "beta_end": self.beta_end,
                "nx": self._nx,
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
        self._nx = cfg["nx"]
        self.n_timesteps = cfg["n_timesteps"]
        self.hidden_channels = cfg["hidden_channels"]
        self._make_schedule()
        self._net = _UNet1D(cond_channels=1, hidden_channels=self.hidden_channels).to(self.device)
        if d["net_state"]:
            self._net.load_state_dict(d["net_state"])
