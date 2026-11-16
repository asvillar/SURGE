"""Adapter registering ``pytorch.ddpm``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_DDPM_INFO = ModelInfo(
    architecture=(
        "Conditional denoising diffusion model (DDPM) for 1D field-to-field "
        "prediction. The denoiser is a small 1D U-Net with residual convolution "
        "blocks, one average-pool downsampling stage and one transposed-convolution "
        "upsampling stage, and sinusoidal timestep embeddings; the conditioning "
        "field is concatenated to the noisy field as an input channel (no "
        "attention). Training minimises the mean squared error on the predicted "
        "noise with a linear beta schedule. predict() runs the T-step ancestral "
        "reverse process from Gaussian noise and returns a single sample."
    ),
    use_cases=[
        "Stochastic 1D field generation conditioned on an input (e.g. forcing → response)",
        "Uncertainty quantification via sample diversity",
        "Scientific data augmentation — generating plausible physical fields",
    ],
    not_for=[
        "Deterministic regression benchmarks — slower and less accurate than MLP/FNO",
        "Tabular scalar prediction — sampling overhead is not justified",
        "Low-latency inference — T=200 reverse steps per prediction is expensive",
    ],
    strengths=[
        "State-of-the-art sample quality for 1D field distributions",
        "Natural UQ via sampling — no auxiliary training objective required",
        "Stable training compared to GANs (no mode collapse)",
    ],
    weaknesses=[
        "Very slow inference: T forward passes of the U-Net per sample",
        "High GPU memory during training — large batch sizes needed for stability",
        "Hyperparameter sensitive: T, beta schedule, conditioning strength",
    ],
    references=[
        "Ho, Jain & Abbeel (2020) 'Denoising Diffusion Probabilistic Models' "
        "NeurIPS 2020. https://arxiv.org/abs/2006.11239",
        "Song et al. (2021) 'Score-Based Generative Modeling through SDEs' "
        "ICLR 2021. https://arxiv.org/abs/2011.13456",
    ],
)

_DDPM_PROFILE = ResourceProfile(
    name="pytorch.ddpm",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Conditional 1D DDPM for field-to-field prediction. Ho et al. NeurIPS 2020.",
)


class DDPMAdapter(BaseModelAdapter):
    """Conditional DDPM 1D surrogate."""

    name = "pytorch.ddpm"
    backend = "pytorch"
    task_type = "regression"
    uses_internal_preprocessing = True
    resource_profile = _DDPM_PROFILE
    _INFO = _DDPM_INFO
    default_params: dict[str, Any] = {
        "n_timesteps": 200,
        "hidden_channels": 64,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "n_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.ddpm")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.DDPMModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
