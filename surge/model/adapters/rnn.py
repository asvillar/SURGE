"""Adapters registering ``pytorch.lstm`` and ``pytorch.gru``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_RNN_PROFILE = ResourceProfile(
    name="pytorch.rnn",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="LSTM / GRU surrogate for temporal sequences.",
)


class _RNNAdapterBase(BaseModelAdapter):
    resource_profile = _RNN_PROFILE
    task_type = "regression"
    default_params: dict[str, Any] = {
        "hidden_size": 128,
        "n_layers": 2,
        "dropout": 0.1,
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "patience": 20,
        "random_state": 42,
    }
    _backend_cls_name: str  # "LSTMModel" or "GRUModel"

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        import importlib
        mod = importlib.import_module("surge.model.backends.rnn")
        cls = getattr(mod, self._backend_cls_name)
        params = dict(self.default_params)
        params.update(kwargs)
        return cls(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        self._model.fit(X, y, X_val, y_val)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))


class LSTMAdapter(_RNNAdapterBase):
    """LSTM surrogate for temporal sequences."""
    name = "pytorch.lstm"
    backend = "pytorch"
    uses_internal_preprocessing = True
    _backend_cls_name = "LSTMModel"


class GRUAdapter(_RNNAdapterBase):
    """GRU surrogate for temporal sequences."""
    name = "pytorch.gru"
    backend = "pytorch"
    uses_internal_preprocessing = True
    _backend_cls_name = "GRUModel"
