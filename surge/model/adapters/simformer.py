"""Adapter registering ``pytorch.simformer`` in the SURGE model registry.

Simformer (Gloeckler et al., ICML 2024): all-in-one simulation-based
inference — one score-based transformer over the joint (θ, x) that can
sample the posterior, the likelihood, or any conditional. Through the
registry contract it behaves as a probabilistic regressor (posterior
mean / std); the full SBI API lives on the underlying model
(``adapter._model.sample_posterior`` / ``sample_likelihood`` /
``sample_conditional``).
"""

from __future__ import annotations

from typing import Any

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE

_SIMFORMER_PROFILE = ResourceProfile(
    name="pytorch.simformer",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Score-based transformer for all-in-one SBI (Gloeckler 2024).",
)


class SimformerAdapter(BaseModelAdapter):
    """All-in-one SBI adapter: fit(X=observables, y=parameters)."""

    name = "pytorch.simformer"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _SIMFORMER_PROFILE
    task_type = "regression"

    default_params: dict[str, Any] = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 4,
        "dim_feedforward": 128,
        "n_epochs": 300,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "n_sample_steps": 64,
        "n_posterior_samples": 128,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.simformer import SimformerModel

        params = dict(self.default_params)
        params.update(kwargs)
        return SimformerModel(**params)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any, **kwargs: Any) -> Any:
        return self._model.predict_with_uncertainty(X)

    def sample_posterior(self, x_obs: Any, n_samples: int = 128) -> Any:
        return self._model.sample_posterior(x_obs, n_samples)

    def sample_likelihood(self, theta: Any, n_samples: int = 128) -> Any:
        return self._model.sample_likelihood(theta, n_samples)

    def sample_conditional(self, values: Any, cond_mask: Any,
                           n_samples: int = 128) -> Any:
        return self._model.sample_conditional(values, cond_mask, n_samples)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
