"""SURGE model package."""
from __future__ import annotations

from .base import BaseModelAdapter, SklearnRegressorAdapter
from .plot_training import (
    compare_training_histories,
    load_training_history,
    plot_training_history,
)
from .sklearn import (
    GPRModel,
    GradientBoostingClassifierAdapter,
    GradientBoostingRegressorModel,
    LogisticRegressionAdapter,
    MLPModel,
    RandomForestClassifierAdapter,
    RandomForestModel,
    RidgeRegressorAdapter,
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
register_model(RidgeRegressorAdapter, key='sklearn.ridge', aliases=['ridge'])

# LightGBM (optional dependency)
try:
    from .sklearn import LGBMRegressorAdapter, LGBMClassifierAdapter
    register_model(LGBMRegressorAdapter, key='lgbm.regressor', aliases=['lgbm', 'lightgbm'])
    register_model(LGBMClassifierAdapter, key='lgbm.classifier', aliases=['lgbm_clf'])
except Exception:
    pass

# CatBoost (optional dependency)
try:
    from .sklearn import CatBoostRegressorAdapter, CatBoostClassifierAdapter
    register_model(CatBoostRegressorAdapter, key='catboost.regressor', aliases=['catboost'])
    register_model(CatBoostClassifierAdapter, key='catboost.classifier', aliases=['catboost_clf'])
except Exception:
    pass

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
    from .adapters.mlp_ensemble import MLPEnsembleAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA_ENS

    if _PTA_ENS:
        register_model(
            MLPEnsembleAdapter,
            key='pytorch.mlp_ensemble',
            aliases=['mlp_ensemble', 'ensemble_mlp'],
        )
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

try:
    from .adapters.cnn import CNN1DAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA3

    if _PTA3:
        register_model(CNN1DAdapter, key='pytorch.cnn1d', aliases=['cnn1d'])
except Exception:
    pass

try:
    from .adapters.rnn import GRUAdapter, LSTMAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA4

    if _PTA4:
        register_model(LSTMAdapter, key='pytorch.lstm', aliases=['lstm'])
        register_model(GRUAdapter, key='pytorch.gru', aliases=['gru'])
except Exception:
    pass

try:
    from .adapters.fno import FNO1dAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA5

    if _PTA5:
        register_model(FNO1dAdapter, key='pytorch.fno1d', aliases=['fno1d', 'fno'])
except Exception:
    pass

try:
    from .adapters.deeponet import DeepONetAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA6

    if _PTA6:
        register_model(DeepONetAdapter, key='pytorch.deeponet', aliases=['deeponet'])
except Exception:
    pass

try:
    from .adapters.lenet import LeNet5Adapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA7

    if _PTA7:
        register_model(LeNet5Adapter, key='pytorch.lenet5', aliases=['lenet5', 'lenet'])
except Exception:
    pass

try:
    from .adapters.resnet import ResNet20Adapter, ResNet56Adapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA8

    if _PTA8:
        register_model(ResNet20Adapter, key='pytorch.resnet20', aliases=['resnet20'])
        register_model(ResNet56Adapter, key='pytorch.resnet56', aliases=['resnet56'])
except Exception:
    pass

try:
    from .adapters.botorch_gp import BoTorchGPAdapter, BoTorchSparseGPAdapter
    register_model(BoTorchGPAdapter, key='botorch.gp', aliases=['botorch_gp', 'bgp'])
    register_model(BoTorchSparseGPAdapter, key='botorch.sparse_gp', aliases=['botorch_sgp', 'sparse_gp'])
except Exception:
    pass

try:
    from .adapters.kan import KANRegressorAdapter, KANClassifierAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA9

    if _PTA9:
        register_model(KANRegressorAdapter, key='pytorch.kan', aliases=['kan'])
        register_model(KANClassifierAdapter, key='pytorch.kan_classifier', aliases=['kan_classifier'])
except Exception:
    pass

try:
    from .adapters.ft_transformer import FTTransformerAdapter, FTTransformerClassifierAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA10

    if _PTA10:
        register_model(FTTransformerAdapter, key='pytorch.ft_transformer', aliases=['ft_transformer', 'ftt'])
        register_model(FTTransformerClassifierAdapter, key='pytorch.ft_transformer_classifier', aliases=['ft_transformer_classifier'])
except Exception:
    pass

try:
    from .adapters.vit import ViTAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA11

    if _PTA11:
        register_model(ViTAdapter, key='pytorch.vit', aliases=['vit'])
except Exception:
    pass

try:
    from .adapters.alexnet import AlexNetAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA12

    if _PTA12:
        register_model(AlexNetAdapter, key='pytorch.alexnet', aliases=['alexnet'])
except Exception:
    pass

try:
    from .adapters.fno2d import FNO2dAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA13

    if _PTA13:
        register_model(FNO2dAdapter, key='pytorch.fno2d', aliases=['fno2d'])
except Exception:
    pass

try:
    from .adapters.unet import UNetAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA14

    if _PTA14:
        register_model(UNetAdapter, key='pytorch.unet', aliases=['unet'])
except Exception:
    pass

try:
    from .adapters.vae import VAEAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA15

    if _PTA15:
        register_model(VAEAdapter, key='pytorch.vae', aliases=['vae'])
except Exception:
    pass

try:
    from .adapters.ddpm import DDPMAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA16

    if _PTA16:
        register_model(DDPMAdapter, key='pytorch.ddpm', aliases=['ddpm'])
except Exception:
    pass

try:
    from .adapters.cgan import CGANAdapter
    from .pytorch import PYTORCH_AVAILABLE as _PTA17

    if _PTA17:
        register_model(CGANAdapter, key='pytorch.cgan', aliases=['cgan'])
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
