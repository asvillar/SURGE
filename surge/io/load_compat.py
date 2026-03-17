"""Compatibility loaders for models saved with older sklearn/PyTorch versions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib

LOG = logging.getLogger(__name__)

# Track if sklearn patch was applied (avoid applying twice)
_SKLEARN_PATCH_APPLIED = False


def _apply_sklearn_compat_patch() -> None:
    """Patch sklearn tree classes to handle missing monotonic_cst in old pickles (1.3.x -> 1.4+)."""
    global _SKLEARN_PATCH_APPLIED
    if _SKLEARN_PATCH_APPLIED:
        return
    try:
        import sklearn.tree

        def _make_compat_setstate(orig_setstate: Any) -> Any:
            def _compat_setstate(self: Any, state: Any) -> Any:
                if isinstance(state, dict) and "monotonic_cst" not in state:
                    state = dict(state)
                    state["monotonic_cst"] = None
                return orig_setstate(self, state)

            return _compat_setstate

        for cls_name in ("DecisionTreeRegressor", "DecisionTreeClassifier"):
            cls = getattr(sklearn.tree, cls_name, None)
            if cls is None or not hasattr(cls, "__setstate__"):
                continue
            cls.__setstate__ = _make_compat_setstate(cls.__setstate__)
        _SKLEARN_PATCH_APPLIED = True
    except Exception as e:
        LOG.debug("Could not apply sklearn compat patch: %s", e)


def _load_torch_model(path: Path, model_entry: dict) -> Optional[Any]:
    """Load a PyTorch model saved with torch.save (avoids joblib/pickle protocol 0 issues)."""
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint, dict):
        return None

    # Format 1: PyTorchMLPModel (model_state_dict + model_config)
    if "model_state_dict" in checkpoint and "model_config" in checkpoint:
        try:
            from ..model.pytorch_impl import PyTorchMLPModel

            model = PyTorchMLPModel()
            model.load(str(path))
            return _TorchAdapter(model)
        except Exception as e:
            LOG.debug("Could not load PyTorchMLPModel format: %s", e)

    # Format 2: state_dict + params (generic/custom MLP with net, scalers)
    if "state_dict" in checkpoint and "params" in checkpoint:
        try:
            return _reconstruct_net_mlp(checkpoint)
        except Exception as e:
            LOG.debug("Could not reconstruct state_dict+params format: %s", e)

    return None


def _TorchAdapter(model: Any) -> Any:
    """Thin wrapper with predict for torch models."""

    class _Adapter:
        def __init__(self, m: Any):
            self._model = m

        def predict(self, X: Any) -> Any:
            return self._model.predict(X)

    return _Adapter(model)


def _reconstruct_net_mlp(checkpoint: dict) -> Optional[Any]:
    """Reconstruct MLP from state_dict + params (net architecture).
    Expects X in scaled space (run's input_scaler); returns predictions in scaled space.
    """
    import torch
    import torch.nn as nn
    import numpy as np

    params = checkpoint["params"]
    state_dict = checkpoint["state_dict"]
    input_dim = params.get("input_dim")
    output_dim = params.get("output_dim")
    hidden_layers = params.get("hidden_layers", (64, 32))
    if isinstance(hidden_layers, tuple):
        hidden_layers = list(hidden_layers)
    dropout = params.get("dropout", 0.1)
    activation = params.get("activation", "relu")

    if input_dim is None or output_dim is None:
        return None

    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
    act_fn = act_map.get(activation, nn.ReLU)

    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))

    class _NetMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

        def predict(self, X):
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            self.eval()
            with torch.no_grad():
                out = self.forward(X)
            return out.cpu().numpy()

    model = _NetMLP()
    model.load_state_dict(state_dict, strict=True)
    return _TorchAdapter(model)


def load_model_compat(
    model_path: Path,
    model_entry: Optional[dict] = None,
) -> Any:
    """
    Load a model with compatibility for older sklearn (monotonic_cst) and PyTorch (persistent IDs).

    Tries joblib.load first (with sklearn patch). On persistent ID errors for torch models,
    falls back to torch.load and reconstructs the adapter.
    """
    _apply_sklearn_compat_patch()
    model_entry = model_entry or {}
    backend = model_entry.get("backend", "")

    # Try joblib first (works for sklearn models and some torch if no protocol 0 issues)
    try:
        return joblib.load(model_path)
    except Exception as e:
        err_msg = str(e).lower()
        is_persistent_id_error = "persistent" in err_msg and "ascii" in err_msg
        is_torch_model = backend in ("torch", "pytorch") or "mlp" in model_entry.get("name", "").lower()

        if is_persistent_id_error and is_torch_model:
            adapter = _load_torch_model(model_path, model_entry)
            if adapter is not None:
                return adapter
        raise
