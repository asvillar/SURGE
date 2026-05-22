"""Adapters registering ``pytorch.ft_transformer`` and
``pytorch.ft_transformer_classifier``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ...hpc import ResourceProfile

_FTT_PROFILE = ResourceProfile(
    name="pytorch.ft_transformer",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Feature Tokenizer + Transformer for tabular data. Gorishniy et al. NeurIPS 2021.",
)


class _FTTransformerBase(BaseModelAdapter):
    resource_profile = _FTT_PROFILE
    uses_internal_preprocessing = True

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.ft_transformer")
        params = dict(self.default_params)
        params.update(kwargs)
        return mod.FTTransformerModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))


class FTTransformerAdapter(_FTTransformerBase):
    """FT-Transformer regressor for tabular data."""

    name = "pytorch.ft_transformer"
    backend = "pytorch"
    task_type = "regression"
    default_params: dict[str, Any] = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "ffn_factor": 2.0,
        "dropout": 0.1,
        "task": "regression",
        "n_epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "patience": 20,
        "random_state": 42,
    }


class FTTransformerClassifierAdapter(_FTTransformerBase):
    """FT-Transformer classifier for tabular data."""

    name = "pytorch.ft_transformer_classifier"
    backend = "pytorch"
    task_type = "classification"
    default_params: dict[str, Any] = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "ffn_factor": 2.0,
        "dropout": 0.1,
        "task": "classification",
        "n_epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "patience": 20,
        "random_state": 42,
    }

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)
