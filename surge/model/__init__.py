"""SURGE model package."""
from __future__ import annotations

from .base import BaseModelAdapter, SklearnRegressorAdapter
from .sklearn import (
    GPRModel,
    GradientBoostingClassifierAdapter,
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
register_model(LogisticRegressionAdapter, key='sklearn.logistic_regression', aliases=['logistic_regression', 'lr'])


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
