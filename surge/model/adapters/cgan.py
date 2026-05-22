"""Adapter registering ``pytorch.cgan``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_CGAN_INFO = ModelInfo(
    architecture=(
        "Conditional Generative Adversarial Network (CGAN) for 1D field prediction: "
        "a generator G(z, x) maps a latent noise vector z concatenated with "
        "conditioning input x to a predicted output field; a discriminator D(y, x) "
        "distinguishes real (y, x) pairs from generated ones. Training alternates "
        "between maximising the discriminator's cross-entropy and minimising the "
        "generator's adversarial loss. The generator is a 1D transposed-convolution "
        "network; the discriminator uses strided convolutions with spectral "
        "normalisation for training stability."
    ),
    use_cases=[
        "Stochastic 1D field generation conditioned on input parameters",
        "Learning multi-modal output distributions (where DDPM/VAE may average modes)",
        "Scientific surrogates where sharp, realistic field samples are needed",
    ],
    not_for=[
        "Deterministic regression — GANs provide stochastic samples, not point estimates",
        "Small datasets (<2k samples) — adversarial training is unstable",
        "Tabular scalar prediction — latent sampling overhead not justified",
    ],
    strengths=[
        "Produces sharp, high-quality samples without blurring artefacts",
        "Can represent multi-modal output distributions",
        "No T-step reverse process — faster inference than DDPM",
    ],
    weaknesses=[
        "Mode collapse — generator may ignore parts of the output distribution",
        "Training instability — requires careful learning rate and architecture tuning",
        "No explicit likelihood — harder to calibrate uncertainty than VAE/DDPM",
    ],
    references=[
        "Mirza & Osindero (2014) 'Conditional Generative Adversarial Nets' "
        "arXiv:1411.1784. https://arxiv.org/abs/1411.1784",
        "Goodfellow et al. (2014) 'Generative Adversarial Networks' "
        "NeurIPS 2014. https://arxiv.org/abs/1406.2661",
    ],
)

_CGAN_PROFILE = ResourceProfile(
    name="pytorch.cgan",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Conditional GAN for 1D field prediction. Mirza & Osindero 2014.",
)


class CGANAdapter(BaseModelAdapter):
    """Conditional GAN 1D surrogate."""

    name = "pytorch.cgan"
    backend = "pytorch"
    task_type = "regression"
    uses_internal_preprocessing = True
    resource_profile = _CGAN_PROFILE
    _INFO = _CGAN_INFO
    default_params: dict[str, Any] = {
        "latent_dim": 64,
        "hidden_channels": 256,
        "n_gen_layers": 4,
        "n_disc_layers": 3,
        "learning_rate_g": 1e-4,
        "learning_rate_d": 4e-4,
        "n_epochs": 500,
        "batch_size": 64,
        "n_predict_samples": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.cgan")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.CGANModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
