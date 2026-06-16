"""Geometric-schedule Residual MLP — hidden widths interpolate input→output."""

from __future__ import annotations

from typing import Any

from .residual_mlp import ResidualMLPAdapter
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_PROFILE = ResourceProfile(
    name="pytorch.geom_residual_mlp",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes=(
        "Residual MLP with geometric hidden-layer schedule: each width lies on "
        "a constant-ratio path from n_inputs to n_outputs."
    ),
)


class GeometricResidualMLPAdapter(ResidualMLPAdapter):
    """
    Residual MLP whose hidden layer sizes are computed at fit time.

    Given ``n_in`` inputs and ``n_out`` outputs, ``n_hidden_layers`` widths
    are placed along a geometric progression (equal ratio between consecutive
    sizes) and clamped to ``[min_hidden_width, max_hidden_width]``.

    Example (90 → 12, 4 hidden layers, max 1024) might yield
    ``[45, 28, 18, 11]`` before clamping.
    """

    name = "pytorch.geom_residual_mlp"
    backend = "pytorch"
    resource_profile = _PROFILE

    default_params: dict[str, Any] = {
        "layer_schedule": "geometric",
        "n_hidden_layers": 3,
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
        params["layer_schedule"] = "geometric"
        return ResidualMLPModel(**params)
