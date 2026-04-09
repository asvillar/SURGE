"""SURGE model package."""
from __future__ import annotations

from .base import BaseModelAdapter, SklearnRegressorAdapter
from .sklearn import GPRModel, MLPModel, RandomForestModel
from .pytorch import PYTORCH_AVAILABLE, PyTorchMLPAdapter
from .gpflow import GPFLOW_AVAILABLE, GPflowGPRAdapter, GPflowMultiKernelAdapter
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

__all__ = [
    "BaseModelAdapter",
    "SklearnRegressorAdapter",
    "RandomForestModel",
    "MLPModel",
    "GPRModel",
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
