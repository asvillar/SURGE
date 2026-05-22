"""Adapters registering ``pytorch.kan``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ...hpc import ResourceProfile

_KAN_INFO = ModelInfo(
    architecture=(
        "Kolmogorov-Arnold Network (KAN): replaces fixed activation functions with "
        "learnable univariate B-spline functions on each edge of the network. Each "
        "weight in a traditional MLP becomes a trainable spline parameterised by "
        "grid_size control points and spline_order (polynomial degree). This is "
        "motivated by the Kolmogorov-Arnold representation theorem, which states "
        "that any multivariate continuous function can be expressed as a composition "
        "of univariate functions and addition."
    ),
    use_cases=[
        "Tabular regression and classification where interpretability is desired",
        "Scientific function approximation where the learned activation shapes "
        "carry physical meaning (e.g., plasma transport, equilibrium surrogates)",
        "Low- to medium-dimensional problems where spline flexibility outweighs cost",
    ],
    not_for=[
        "Large-scale datasets (>100k rows) — spline computation is significantly "
        "slower than standard MLPs at the same parameter count",
        "High-dimensional inputs without feature selection — grid grows with features",
    ],
    strengths=[
        "Learned activation functions are visualisable and can reveal symbolic structure",
        "Often achieves competitive accuracy with fewer parameters than MLPs on "
        "low-dimensional scientific problems",
        "The spline grid can be refined post-training for higher accuracy",
    ],
    weaknesses=[
        "Requires the optional `efficient-kan` package "
        "(pip install git+https://github.com/Blealtan/efficient-kan)",
        "Training is slower than MLP — O(grid_size) overhead per neuron per forward pass",
        "Sensitive to grid_size and spline_order hyperparameters",
    ],
    references=[
        "Liu et al. (2024) 'KAN: Kolmogorov-Arnold Networks' arXiv:2404.19756",
        "Liu et al. (2024) 'KAN 2.0: Kolmogorov-Arnold Networks Meet Science' arXiv:2408.10205",
        "Blealtan (2024) efficient-kan: https://github.com/Blealtan/efficient-kan",
    ],
)

_KAN_PROFILE = ResourceProfile(
    name="pytorch.kan",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Kolmogorov-Arnold Network (B-spline activations on edges). Liu et al. 2024.",
)


class KANRegressorAdapter(BaseModelAdapter):
    """KAN regressor."""

    name = "pytorch.kan"
    backend = "pytorch"
    task_type = "regression"
    uses_internal_preprocessing = True
    resource_profile = _KAN_PROFILE
    _INFO = _KAN_INFO
    default_params: dict[str, Any] = {
        "hidden_dims": [64, 64],
        "grid_size": 5,
        "spline_order": 3,
        "task": "regression",
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.kan")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.KANModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))


class KANClassifierAdapter(BaseModelAdapter):
    """KAN classifier."""

    name = "pytorch.kan_classifier"
    backend = "pytorch"
    task_type = "classification"
    uses_internal_preprocessing = True
    resource_profile = _KAN_PROFILE
    _INFO = _KAN_INFO
    default_params: dict[str, Any] = {
        "hidden_dims": [64, 64],
        "grid_size": 5,
        "spline_order": 3,
        "task": "classification",
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.kan")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.KANModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
