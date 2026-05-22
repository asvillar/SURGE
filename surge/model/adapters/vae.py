"""Adapter registering ``pytorch.vae``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_VAE_INFO = ModelInfo(
    architecture=(
        "Variational Autoencoder (VAE) with a regression head: an encoder MLP "
        "maps inputs x to a Gaussian posterior q(z|x) via mean and log-variance "
        "vectors; a sample z ~ q(z|x) feeds a decoder MLP that reconstructs x "
        "and a separate regression head that predicts y. Trained end-to-end with "
        "ELBO = reconstruction_loss + β·KL(q||p) + regression_loss. The β "
        "parameter (β-VAE) controls the degree of disentanglement in the latent "
        "space. At inference, the mean of q(z|x) is used for deterministic "
        "prediction; multiple samples give a Monte Carlo uncertainty estimate."
    ),
    use_cases=[
        "Regression with built-in latent uncertainty quantification via sampling",
        "Learning structured latent representations of scientific inputs "
        "(e.g. plasma shape parameters, material configurations)",
        "Anomaly detection via reconstruction error in scientific datasets",
    ],
    not_for=[
        "Large-scale image generation — use DDPM or dedicated image VAEs",
        "Tasks where reconstruction quality is the primary objective",
    ],
    strengths=[
        "Provides prediction uncertainty via latent sampling (predict_with_uncertainty)",
        "Learns compact latent representations useful for downstream tasks",
        "β parameter allows trading off reconstruction quality vs. disentanglement",
    ],
    weaknesses=[
        "Blurry reconstructions compared to GANs/diffusion models",
        "Posterior collapse in deep networks — KL term can vanish",
        "Regression head accuracy lower than a plain MLP on the same data",
    ],
    references=[
        "Kingma & Welling (2014) 'Auto-Encoding Variational Bayes' "
        "ICLR 2014. https://arxiv.org/abs/1312.6114",
        "Higgins et al. (2017) 'β-VAE: Learning Basic Visual Concepts with a "
        "Constrained Variational Framework' ICLR 2017.",
    ],
)

_VAE_PROFILE = ResourceProfile(
    name="pytorch.vae",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Variational Autoencoder with regression head. Kingma & Welling ICLR 2014.",
)


class VAEAdapter(BaseModelAdapter):
    """VAE latent regression surrogate."""

    name = "pytorch.vae"
    backend = "pytorch"
    task_type = "regression"
    uses_internal_preprocessing = True
    resource_profile = _VAE_PROFILE
    _INFO = _VAE_INFO
    default_params: dict[str, Any] = {
        "latent_dim": 16,
        "hidden_dim": 128,
        "beta": 1.0,
        "regression_weight": 1.0,
        "task": "regression",
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.vae")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.VAEModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any) -> tuple:
        return self._model.predict_with_uncertainty(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
