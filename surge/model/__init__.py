"""SURGE model package."""
from __future__ import annotations

from .base import BaseModelAdapter, SklearnRegressorAdapter
from .sklearn import (
    GPRModel,
    GradientBoostingClassifierAdapter,
    GradientBoostingRegressorModel,
    LogisticRegressionAdapter,
    MLPModel,
    RandomForestClassifierAdapter,
    RandomForestModel,
    SklearnClassifierAdapter,
)
from .pytorch import PYTORCH_AVAILABLE, PyTorchMLPAdapter
from .gpflow import GPflowGPRAdapter, GPflowMultiKernelAdapter
from .ensembles import EnsemblePrediction, FNNEnsemble
from .registry import MODEL_REGISTRY, create_model, list_models, register_model

# Register default models
register_model(RandomForestModel, key='sklearn.random_forest', aliases=['random_forest', 'rfr'])
register_model(MLPModel, key='sklearn.mlp', aliases=['mlp'])
register_model(GPRModel, key='sklearn.gpr', aliases=['gpr'])
register_model(
    PyTorchMLPAdapter,
    key='pytorch.mlp',
    aliases=['torch_mlp', 'torch.mlp'],
)
register_model(GPflowGPRAdapter, key='gpflow.gpr', aliases=['gp_gpr'])
register_model(GPflowMultiKernelAdapter, key='gpflow.multi_kernel', aliases=['gpflow_mk'])
register_model(RandomForestClassifierAdapter, key='sklearn.random_forest_classifier', aliases=['rf_classifier', 'rfc'])
register_model(GradientBoostingClassifierAdapter, key='sklearn.gradient_boosting_classifier', aliases=['gbc', 'gradient_boosting'])
register_model(GradientBoostingRegressorModel, key='sklearn.gradient_boosting_regressor', aliases=['gbr'])
register_model(LogisticRegressionAdapter, key='sklearn.logistic_regression', aliases=['logistic_regression', 'lr'])

# New model adapters from surge/model/adapters/
try:
    from .adapters.residual_mlp import ResidualMLPAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA

    if _PTA:
        register_model(ResidualMLPAdapter, key='pytorch.residual_mlp', aliases=['residual_mlp'])
except Exception:
    pass

try:
    from .adapters.mlp_classifier import MLPClassifierAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA2

    if _PTA2:
        register_model(MLPClassifierAdapter, key='pytorch.mlp_classifier', aliases=['mlp_classifier'])
except Exception:
    pass

try:
    from .adapters.xgboost import XGBClassifierAdapter, XGBRegressorAdapter
    from .backends.xgboost import XGBOOST_AVAILABLE as _XGBA

    if _XGBA:
        register_model(XGBRegressorAdapter, key='xgboost.xgbregressor', aliases=['xgbr', 'xgboost'])
        register_model(XGBClassifierAdapter, key='xgboost.xgbclassifier', aliases=['xgbc'])
except Exception:
    pass


def __getattr__(name: str):  # PEP 562: avoid importing TensorFlow/GPflow at import time
    if name == "GPFLOW_AVAILABLE":
        from .gpflow import gpflow_runtime_available

        return gpflow_runtime_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseModelAdapter",
    "SklearnRegressorAdapter",
    "SklearnClassifierAdapter",
    "RandomForestModel",
    "MLPModel",
    "GPRModel",
    "RandomForestClassifierAdapter",
    "GradientBoostingClassifierAdapter",
    "GradientBoostingRegressorModel",
    "LogisticRegressionAdapter",
    "PyTorchMLPAdapter",
    "GPflowGPRAdapter",
    "GPflowMultiKernelAdapter",
    "FNNEnsemble",
    "EnsemblePrediction",
    "PYTORCH_AVAILABLE",
    "GPFLOW_AVAILABLE",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    "register_model",
]
