"""Adapter for the PyTorch MLP classifier backend (registers pytorch.mlp_classifier)."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_PROFILE = ResourceProfile(
    name="pytorch.mlp_classifier",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="PyTorch MLP classifier; CrossEntropy / BCEWithLogits loss.",
)


class MLPClassifierAdapter(BaseModelAdapter):
    """
    PyTorch MLP classifier — neural baseline for classification benchmarks.

    Supports ``predict_proba()`` for AUROC and log-loss computation.

    Default hyperparameters
    -----------------------
    hidden_layers : [128, 64]
    n_epochs      : 150
    learning_rate : 1e-3
    dropout_rate  : 0.1
    patience      : 15   (early-stopping; 0 to disable)
    """

    name = "pytorch.mlp_classifier"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _PROFILE
    task_type = "classification"

    default_params: dict[str, Any] = {
        "hidden_layers": [128, 64],
        "n_epochs": 150,
        "learning_rate": 1e-3,
        "dropout_rate": 0.1,
        "patience": 15,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.mlp_classifier import MLPClassifierModel

        params = dict(self.default_params)
        params.update(kwargs)
        return MLPClassifierModel(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        self._model.fit(X, y, X_val, y_val)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return class probability matrix (n_samples × n_classes)."""
        return self._model.predict_proba(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
