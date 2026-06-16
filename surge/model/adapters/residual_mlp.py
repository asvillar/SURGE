"""Adapter for the Residual MLP backend (registers pytorch.residual_mlp)."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_PROFILE = ResourceProfile(
    name="pytorch.residual_mlp",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Residual MLP; num_workers unused (single DataLoader worker for tabular data).",
)


class ResidualMLPAdapter(BaseModelAdapter):
    """
    Residual MLP regressor — skip-connection dense network.

    Uses :class:`~surge.model.backends.residual_mlp.ResidualMLPModel`.
    Appropriate for nonlinear tabular regression where a standard MLP
    under-fits or suffers from vanishing gradients in deeper networks.

    Default hyperparameters
    -----------------------
    hidden_layers : [128, 128]  (any widths in [1, max_hidden_width])
    max_hidden_width : 1024
    layer_schedule : explicit | geometric
    n_epochs      : 200
    learning_rate : 1e-3
    dropout_rate  : 0.1
    patience      : 20   (early-stopping; 0 to disable)
    """

    name = "pytorch.residual_mlp"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _PROFILE
    task_type = "regression"

    default_params: dict[str, Any] = {
        "hidden_layers": [128, 128],
        "layer_schedule": "explicit",
        "max_hidden_width": 1024,
        "min_hidden_width": 1,
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "dropout_rate": 0.1,
        "patience": 20,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")
        from surge.model.backends.residual_mlp import ResidualMLPModel

        params = dict(self.default_params)
        params.update(kwargs)
        return ResidualMLPModel(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None, **kwargs: Any) -> None:
        self._model.fit(X, y, X_val, y_val)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
