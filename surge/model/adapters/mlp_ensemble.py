"""Adapter for the MLP ensemble backend (registers pytorch.mlp_ensemble).

Architecture matches Cadena et al. (2025) ConStellaration paper — Appendix A.4:
10-member MLP ensemble, 3 hidden layers × 256 units, tanh activation, MSE/Adam.
"""
from __future__ import annotations

from typing import Any

from surge.model.base import BaseModelAdapter, ModelInfo

_ENSEMBLE_INFO = ModelInfo(
    architecture=(
        "MLP Ensemble: N independently initialised and trained fully-connected "
        "networks (default N=10). Each member has the same architecture "
        "(default: 3 hidden layers × 256 units, tanh activations, MSE+Adam) "
        "but different random seeds. Predictions are aggregated as the "
        "pointwise mean; uncertainty is the pointwise standard deviation across "
        "members. This is the 'deep ensemble' approach of Lakshminarayanan et al. "
        "(2017), which is consistently competitive with Bayesian methods on "
        "well-calibrated UQ benchmarks. Default hyperparameters reproduce the "
        "surrogate baseline in the ConStellaration paper (Cadena et al. 2025)."
    ),
    use_cases=[
        "Tabular and scalar regression with calibrated uncertainty estimates",
        "Scientific surrogates where prediction intervals matter "
        "(plasma stability, material properties, transport coefficients)",
        "Benchmark baseline for UQ comparison against Bayesian / dropout methods",
    ],
    not_for=[
        "Very large datasets (>500k rows) — training N models multiplies cost",
        "Low-latency inference — N forward passes per prediction",
    ],
    strengths=[
        "Calibrated uncertainty: PICP typically >90% on held-out test sets",
        "Trivially parallelisable — members are independent",
        "No approximate inference — each member is a standard MLP",
        "Consistently outperforms MC-Dropout on calibration benchmarks "
        "(Lakshminarayanan et al. 2017)",
    ],
    weaknesses=[
        "N× training cost and N× inference cost compared to a single MLP",
        "Does not capture epistemic uncertainty from a single posterior",
    ],
    references=[
        "Lakshminarayanan, Pritzel & Blundell (2017) 'Simple and Scalable "
        "Predictive Uncertainty Estimation using Deep Ensembles' "
        "NeurIPS 2017. https://arxiv.org/abs/1612.01474",
        "Cadena et al. (2025) 'ConStellaration: a dataset of QI-like "
        "stellarator plasma boundaries and optimization benchmarks' "
        "arXiv:2506.19583. §Appendix A.4.",
    ],
)


class MLPEnsembleAdapter(BaseModelAdapter):
    """Ensemble of *n_ensembles* independent PyTorch MLPs.

    Default hyper-parameters reproduce the surrogate baseline reported in the
    ConStellaration paper (Cadena et al. 2025, arXiv:2506.19583):

    * ``n_ensembles=10``
    * ``hidden_dim=256``
    * ``n_layers=3``
    * ``activation="tanh"``
    * ``n_epochs=200``, ``learning_rate=1e-3``, ``batch_size=256``
    * ``patience=30`` (early stopping)

    The adapter exposes ``predict_with_uncertainty(X)`` returning (mean, std).
    """

    name = "pytorch.mlp_ensemble"
    backend = "pytorch"
    uses_internal_preprocessing = True
    handles_output_scaling = True
    _INFO = _ENSEMBLE_INFO

    _DEFAULT_PARAMS: dict[str, Any] = dict(
        n_ensembles=10,
        hidden_dim=256,
        n_layers=3,
        activation="tanh",
        n_epochs=200,
        learning_rate=1e-3,
        batch_size=256,
        patience=30,
        dropout=0.0,
    )

    def __init__(self, **kwargs: Any) -> None:
        # Merge defaults with any user-supplied overrides before calling super
        merged = {**self._DEFAULT_PARAMS, **kwargs}
        super().__init__(**merged)

    def _initialize(self) -> None:
        self._model = None

    def _build_model(self, **kwargs: Any):
        from surge.model.backends.mlp_ensemble import MLPEnsembleModel

        return MLPEnsembleModel(**kwargs)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_val: Any = None,
        y_val: Any = None,
        **kwargs: Any,
    ) -> None:
        if self._model is None:
            self._model = self._build_model(**self.params)
        self._model.fit(X, y, X_val=X_val, y_val=y_val)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any, **kwargs: Any):
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict_with_uncertainty(X)

    @property
    def training_history(self):
        if self._model is None:
            return None
        return getattr(self._model, "training_history", None)
