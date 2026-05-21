"""Adapters registering ``pytorch.lstm`` and ``pytorch.gru``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter, ModelInfo
from ..pytorch import PYTORCH_AVAILABLE
from ...hpc import ResourceProfile

_RNN_INFO = ModelInfo(
    architecture=(
        "Stacked LSTM/GRU with configurable hidden size and depth. "
        "Input shape (B, T, features) → output (B, T, targets). "
        "StandardScaler applied to X and y; early stopping on validation loss."
    ),
    use_cases=[
        "Time-series forecasting (Lorenz-63/96, plasma control signals)",
        "Sequence-to-sequence regression where temporal order matters",
        "Surrogate models for dynamical systems with memory effects",
    ],
    not_for=[
        "Static tabular regression — tree models and MLPs are faster and more accurate",
        "Image or 2-D spatial data — use CNN2D or ViT",
        "Very long sequences (> 1000 steps) — use Transformer instead",
    ],
    strengths=[
        "Captures temporal dependencies and long-range order",
        "Works well on plasma disruption precursor sequences",
        "Bidirectional option doubles representational capacity",
    ],
    weaknesses=[
        "Slower to train than MLP on same data size",
        "Vanishing gradients on sequences > 200 steps despite LSTM gating",
        "Underperforms tree ensembles on non-temporal tabular benchmarks",
    ],
    references=[
        "Hochreiter & Schmidhuber (1997) Long Short-Term Memory. Neural Computation.",
        "Cho et al. (2014) Learning Phrase Representations using RNN Encoder-Decoder. EMNLP.",
        "Grinsztajn et al. (2022) Why tree-based models still outperform deep learning on "
        "tabular data. NeurIPS. https://arxiv.org/abs/2207.08815",
    ],
    notes=(
        "Grinsztajn et al. (2022) showed that on static tabular benchmarks, "
        "gradient-boosted trees and MLPs consistently outperform LSTMs/GRUs. "
        "Use these models only when temporal ordering is meaningful."
    ),
)

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
    _INFO = _RNN_INFO


class GRUAdapter(_RNNAdapterBase):
    """GRU surrogate for temporal sequences."""
    name = "pytorch.gru"
    backend = "pytorch"
    uses_internal_preprocessing = True
    _backend_cls_name = "GRUModel"
    _INFO = _RNN_INFO
