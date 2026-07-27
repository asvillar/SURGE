"""Simformer: all-in-one simulation-based inference (score-based transformer).

Learns the JOINT distribution p(θ, x) of simulation parameters θ and
observables x with a score-based diffusion model whose network is a
transformer over per-variable tokens. Because the condition mask is part
of the model input (and resampled during training), one trained model can
sample ANY conditional of the joint — posterior p(θ|x), likelihood
p(x|θ), or arbitrary subsets — by clamping observed variables during the
reverse diffusion.

Reference
---------
Gloeckler, Deistler, Weilbach, Wood, Macke (2024),
"All-in-one simulation-based inference", ICML 2024.
https://arxiv.org/abs/2404.09636  (reference implementation in JAX:
https://github.com/mackelab/simformer — this is an independent PyTorch
implementation of the method for the SURGE registry.)

Method notes (matching the paper):
* each variable is one token = value embedding + learned identity
  embedding + learned condition-state embedding; diffusion time enters
  through a Fourier-feature embedding added to every token;
* VE-SDE denoising score matching; observed variables are kept CLEAN in
  the network input (``x_t^(M) = (1-M)·x_t + M·x_0``) and the loss is
  masked to latent variables only;
* the condition mask is resampled per pair per iteration among
  joint / posterior / likelihood / random-Bernoulli masks;
* sampling runs reverse-SDE Euler–Maruyama on latent variables with the
  observed ones clamped (~50-100 steps suffice).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from surge.utils import resolve_device

_LOG = logging.getLogger("surge.pytorch.simformer")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = None  # type: ignore


class _TimeEmbed(nn.Module if TORCH_AVAILABLE else object):
    """Gaussian Fourier features of diffusion time -> d_model vector."""

    def __init__(self, d_model: int, scale: float = 16.0) -> None:
        super().__init__()
        half = d_model // 2
        self.register_buffer("freqs", torch.randn(half) * scale)
        self.proj = nn.Sequential(
            nn.Linear(2 * half, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))

    def forward(self, t):                      # t: (B,)
        ang = t[:, None] * self.freqs[None, :] * 2 * math.pi
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))


class _SimformerNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_vars: int, d_model: int, n_heads: int,
                 n_layers: int, dim_feedforward: int) -> None:
        super().__init__()
        self.value_in = nn.Linear(1, d_model)
        self.id_embed = nn.Embedding(n_vars, d_model)
        self.cond_embed = nn.Embedding(2, d_model)
        self.time_embed = _TimeEmbed(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
            activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.n_vars = n_vars

    def forward(self, x, cond_mask, t):
        # x: (B, D) values; cond_mask: (B, D) in {0,1}; t: (B,)
        B, D = x.shape
        ids = torch.arange(D, device=x.device).expand(B, D)
        tok = (self.value_in(x[..., None])
               + self.id_embed(ids)
               + self.cond_embed(cond_mask.long())
               + self.time_embed(t)[:, None, :])
        return self.head(self.encoder(tok))[..., 0]      # (B, D) score


class SimformerModel:
    """All-in-one SBI surrogate over the joint (θ, x).

    ``fit(X, y)`` follows the SURGE convention: ``X`` are the observables
    / conditioning data (n, D_x) and ``y`` the parameters (n, D_θ). The
    model learns the joint; ``predict`` returns the posterior mean of θ
    given x and ``predict_with_uncertainty`` its posterior std. The
    all-in-one API is exposed via :meth:`sample_posterior`,
    :meth:`sample_likelihood`, :meth:`sample_joint` and the fully general
    :meth:`sample_conditional`.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 128,
        n_epochs: int = 300,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        sigma_min: float = 1e-2,
        sigma_max: float = 4.0,
        n_sample_steps: int = 64,
        n_posterior_samples: int = 128,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_sample_steps = n_sample_steps
        self.n_posterior_samples = n_posterior_samples
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.log_file = log_file
        self._net: Any = None
        self.is_fitted = False
        self.training_history: Any = []
        self.d_theta = 0
        self.d_x = 0
        self._mu: Any = None
        self._sd: Any = None

    # ── diffusion helpers (VE-SDE) ──────────────────────────────────────
    def _sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    # ── training ────────────────────────────────────────────────────────
    def _sample_cond_mask(self, B: int, D: int) -> "torch.Tensor":
        """Per-sample mask type: joint / posterior / likelihood / random."""
        mask = torch.zeros(B, D, device=self.device)
        kind = torch.randint(0, 4, (B,), device=self.device)
        mask[kind == 1, self.d_theta:] = 1.0          # posterior: x observed
        mask[kind == 2, :self.d_theta] = 1.0          # likelihood: θ observed
        rnd = kind == 3
        if rnd.any():
            p = torch.rand(int(rnd.sum()), 1, device=self.device)
            bern = (torch.rand(int(rnd.sum()), D,
                               device=self.device) < p).float()
            # never condition on everything (nothing left to denoise)
            all_on = bern.sum(dim=1) == D
            bern[all_on] = 0.0
            mask[rnd] = bern
        return mask

    def fit(self, X, y, **_: Any) -> "SimformerModel":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if X.ndim == 1:
            X = X[:, None]
        joint = np.concatenate([y, X], axis=1)        # [θ | x]
        self.d_theta, self.d_x = y.shape[1], X.shape[1]
        D = joint.shape[1]

        self._mu = joint.mean(0)
        self._sd = joint.std(0) + 1e-8
        z = (joint - self._mu) / self._sd

        torch.manual_seed(self.random_state)
        self._net = _SimformerNet(D, self.d_model, self.n_heads,
                                  self.n_layers,
                                  self.dim_feedforward).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        data = torch.from_numpy(z).to(self.device)

        from ._progress import ProgressList
        self.training_history = ProgressList(
            self.n_epochs, verbose=self.verbose,
            log_file=self.log_file, desc=type(self).__name__)

        n = len(data)
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            eloss, nb = 0.0, 0
            for i in range(0, n, self.batch_size):
                x0 = data[perm[i:i + self.batch_size]]
                B = len(x0)
                t = torch.rand(B, device=self.device).clamp_(1e-3, 1.0)
                sig = self._sigma(t)[:, None]
                eps = torch.randn(B, x0.shape[1], device=self.device)
                x_t = x0 + sig * eps
                mask = self._sample_cond_mask(B, x0.shape[1])
                x_in = (1 - mask) * x_t + mask * x0   # observed stay clean
                score = self._net(x_in, mask, t)
                # weighted DSM: || σ·s + ε ||² on latent variables only
                resid = (sig * score + eps) * (1 - mask)
                loss = resid.pow(2).sum() / (1 - mask).sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                eloss += float(loss)
                nb += 1
            self.training_history.append(
                {"epoch": epoch + 1, "train_loss": eloss / max(nb, 1)})
        self.training_history.close()
        self.is_fitted = True
        return self

    # ── all-in-one sampling ────────────────────────────────────────────
    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def sample_conditional(self, values: np.ndarray, cond_mask: np.ndarray,
                           n_samples: int = 128,
                           n_steps: int | None = None) -> np.ndarray:
        """Sample latent variables of the joint given observed ones.

        Parameters
        ----------
        values:
            (D,) joint-ordered vector ``[θ | x]``; entries where
            ``cond_mask == 1`` are the conditioning values (others ignored).
        cond_mask:
            (D,) binary; 1 = observed.
        Returns
        -------
        (n_samples, D) full joint vectors (observed entries fixed).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        n_steps = n_steps or self.n_sample_steps
        D = self.d_theta + self.d_x
        vals = ((np.asarray(values, dtype=np.float32) - self._mu)
                / self._sd)
        mask = torch.from_numpy(
            np.asarray(cond_mask, dtype=np.float32)).to(self.device)
        mask_b = mask[None, :].expand(n_samples, D)
        obs = torch.from_numpy(vals).to(self.device)[None, :] \
            .expand(n_samples, D)

        x = torch.randn(n_samples, D, device=self.device) * self.sigma_max
        x = (1 - mask_b) * x + mask_b * obs
        ts = torch.linspace(1.0, 1e-3, n_steps + 1, device=self.device)
        for k in range(n_steps):
            t0, t1 = ts[k], ts[k + 1]
            dt = t0 - t1
            tb = torch.full((n_samples,), float(t0), device=self.device)
            sig = self._sigma(tb)[:, None]
            # VE: g(t)² = dσ²/dt = 2 σ² ln(σ_max/σ_min)
            g2 = 2.0 * sig ** 2 * math.log(self.sigma_max / self.sigma_min)
            score = self._net(x, mask_b, tb)
            x_new = x + g2 * score * dt
            if k < n_steps - 1:
                x_new = x_new + torch.sqrt(g2 * dt) * torch.randn_like(x)
            x = (1 - mask_b) * x_new + mask_b * obs
        out = x.cpu().numpy() * self._sd + self._mu
        return out

    def sample_posterior(self, x_obs, n_samples: int = 128) -> np.ndarray:
        """p(θ | x): (n_samples, d_theta)."""
        x_obs = np.asarray(x_obs, dtype=np.float32).ravel()
        D = self.d_theta + self.d_x
        values = np.zeros(D, dtype=np.float32)
        values[self.d_theta:] = x_obs
        mask = np.zeros(D, dtype=np.float32)
        mask[self.d_theta:] = 1.0
        joint = self.sample_conditional(values, mask, n_samples)
        return joint[:, :self.d_theta]

    def sample_likelihood(self, theta, n_samples: int = 128) -> np.ndarray:
        """p(x | θ): (n_samples, d_x)."""
        theta = np.asarray(theta, dtype=np.float32).ravel()
        D = self.d_theta + self.d_x
        values = np.zeros(D, dtype=np.float32)
        values[:self.d_theta] = theta
        mask = np.zeros(D, dtype=np.float32)
        mask[:self.d_theta] = 1.0
        joint = self.sample_conditional(values, mask, n_samples)
        return joint[:, self.d_theta:]

    def sample_joint(self, n_samples: int = 128) -> np.ndarray:
        """p(θ, x): (n_samples, d_theta + d_x)."""
        D = self.d_theta + self.d_x
        return self.sample_conditional(np.zeros(D), np.zeros(D), n_samples)

    # ── SURGE regression contract: posterior mean / std ────────────────
    def _posterior_stats(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        means, stds = [], []
        for row in X:
            s = self.sample_posterior(row, self.n_posterior_samples)
            means.append(s.mean(0))
            stds.append(s.std(0))
        return np.stack(means), np.stack(stds)

    def predict(self, X) -> np.ndarray:
        mean, _ = self._posterior_stats(X)
        return mean.ravel() if self.d_theta == 1 else mean

    def predict_with_uncertainty(self, X, **_: Any):
        mean, std = self._posterior_stats(X)
        if self.d_theta == 1:
            return mean.ravel(), std.ravel()
        return mean, std

    # ── persistence ────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "config": {
                "d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dim_feedforward": self.dim_feedforward,
                "sigma_min": self.sigma_min, "sigma_max": self.sigma_max,
                "n_sample_steps": self.n_sample_steps,
                "n_posterior_samples": self.n_posterior_samples,
            },
            "d_theta": self.d_theta, "d_x": self.d_x,
            "mu": self._mu, "sd": self._sd,
            "net_state": (self._net.state_dict()
                          if self._net is not None else None),
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> None:
        import joblib
        d = joblib.load(path)
        cfg = d["config"]
        for k, v in cfg.items():
            setattr(self, k, v)
        self.d_theta, self.d_x = d["d_theta"], d["d_x"]
        self._mu, self._sd = d["mu"], d["sd"]
        self.is_fitted = d["is_fitted"]
        if d["net_state"] is not None:
            D = self.d_theta + self.d_x
            self._net = _SimformerNet(
                D, self.d_model, self.n_heads, self.n_layers,
                self.dim_feedforward).to(self.device)
            self._net.load_state_dict(d["net_state"])
            self._net.eval()
