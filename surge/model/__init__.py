"""SURGE model package."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Adapter registration
# ---------------------------------------------------------------------------
# Core (always-available) adapters register directly; optional groups go
# through _register_optional, which records WHY a group is unavailable
# instead of hiding it. Inspect the outcome with
# `surge.model.registration_report()` / `registration_table()`, or
# `surge models --verbose` on the CLI. Set SURGE_STRICT_REGISTRY=1 (as CI
# does) to turn adapter *bugs* (anything other than a missing optional
# dependency) into hard import errors.
import os as _os
import warnings as _warnings

from .base import BaseModelAdapter, SklearnRegressorAdapter
from .ensembles import EnsemblePrediction, FNNEnsemble
from .gpflow import GPflowGPRAdapter, GPflowMultiKernelAdapter
from .plot_training import (
    compare_training_histories,
    load_training_history,
    plot_training_history,
)
from .pytorch import PYTORCH_AVAILABLE, PyTorchMLPAdapter
from .registry import (
    MODEL_REGISTRY,
    REGISTRATION_LOG,
    RegistrationRecord,
    create_model,
    list_models,
    register_model,
    registration_report,
    registration_table,
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

_STRICT = _os.environ.get("SURGE_STRICT_REGISTRY", "").lower() in ("1", "true", "yes")


def _record(keys, status, reason="", requires=()):
    reason = " ".join(str(reason).split())          # single line
    if len(reason) > 220:
        reason = reason[:217] + "..."
    for key in keys:
        REGISTRATION_LOG.append(
            RegistrationRecord(key=key, status=status, reason=reason,
                               requires=tuple(requires)))


def _register_core(adapter_cls, key, aliases=()):
    register_model(adapter_cls, key=key, aliases=list(aliases))
    _record([key], "registered")


def _register_optional(keys, requires, loader, *, needs_torch=False):
    """Attempt one optional adapter group.

    ImportError/OSError => the optional dependency is missing or its native
    library is broken: record "skipped" with the real reason. Any other
    exception is a SURGE bug: record "error", warn loudly, and re-raise
    under SURGE_STRICT_REGISTRY.
    """
    if needs_torch and not PYTORCH_AVAILABLE:
        _record(keys, "skipped", "torch not installed", requires)
        return
    try:
        loader()
    except (ImportError, OSError) as exc:
        _record(keys, "skipped", f"{'/'.join(requires) or 'dependency'} unavailable: {exc}",
                requires)
    except Exception as exc:
        # An exception class defined inside the optional dependency itself
        # (e.g. xgboost.core.XGBoostError on a broken libomp) is still a
        # dependency problem, not a SURGE bug.
        exc_root = type(exc).__module__.split(".")[0]
        if exc_root in {r.replace("-", "_") for r in requires}:
            _record(keys, "skipped",
                    f"{exc_root} broken: {type(exc).__name__}: {exc}", requires)
            return
        _record(keys, "error", f"{type(exc).__name__}: {exc}", requires)
        _warnings.warn(
            f"SURGE adapter group {list(keys)} failed to register with an "
            f"internal error (not a missing dependency): {exc!r}. "
            "Set SURGE_STRICT_REGISTRY=1 to raise.",
            RuntimeWarning, stacklevel=3)
        if _STRICT:
            raise
    else:
        _record(keys, "registered", requires=requires)


# -- core adapters (base install) -------------------------------------------
_register_core(RandomForestModel, 'sklearn.random_forest', ['random_forest', 'rfr'])
_register_core(MLPModel, 'sklearn.mlp', ['mlp'])
_register_core(GPRModel, 'sklearn.gpr', ['gpr'])
_register_core(GPflowGPRAdapter, 'gpflow.gpr', ['gp_gpr'])
_register_core(GPflowMultiKernelAdapter, 'gpflow.multi_kernel', ['gpflow_mk'])
_register_core(RandomForestClassifierAdapter, 'sklearn.random_forest_classifier', ['rf_classifier', 'rfc'])
_register_core(GradientBoostingClassifierAdapter, 'sklearn.gradient_boosting_classifier', ['gbc', 'gradient_boosting'])
_register_core(GradientBoostingRegressorModel, 'sklearn.gradient_boosting_regressor', ['gbr'])
_register_core(LogisticRegressionAdapter, 'sklearn.logistic_regression', ['logistic_regression', 'lr'])
_register_core(RidgeRegressorAdapter, 'sklearn.ridge', ['ridge'])

if PYTORCH_AVAILABLE:
    _register_core(PyTorchMLPAdapter, 'pytorch.mlp', ['torch_mlp', 'torch.mlp'])
else:
    _record(['pytorch.mlp'], "skipped", "torch not installed", ("torch",))


# -- optional adapter groups -------------------------------------------------

def _load_lgbm():
    import lightgbm  # noqa: F401 - fail HERE, not at fit() time

    from .sklearn import LGBMClassifierAdapter, LGBMRegressorAdapter
    register_model(LGBMRegressorAdapter, key='lgbm.regressor', aliases=['lgbm', 'lightgbm'])
    register_model(LGBMClassifierAdapter, key='lgbm.classifier', aliases=['lgbm_clf'])


def _load_catboost():
    import catboost  # noqa: F401

    from .sklearn import CatBoostClassifierAdapter, CatBoostRegressorAdapter
    register_model(CatBoostRegressorAdapter, key='catboost.regressor', aliases=['catboost'])
    register_model(CatBoostClassifierAdapter, key='catboost.classifier', aliases=['catboost_clf'])


def _load_xgboost():
    import xgboost  # noqa: F401

    from .adapters.xgboost import XGBClassifierAdapter, XGBRegressorAdapter
    register_model(XGBRegressorAdapter, key='xgboost.xgbregressor', aliases=['xgbr', 'xgboost'])
    register_model(XGBClassifierAdapter, key='xgboost.xgbclassifier', aliases=['xgbc'])


def _load_botorch():
    from .adapters.botorch_gp import BoTorchGPAdapter, BoTorchSparseGPAdapter
    register_model(BoTorchGPAdapter, key='botorch.gp', aliases=['botorch_gp', 'bgp'])
    register_model(BoTorchSparseGPAdapter, key='botorch.sparse_gp', aliases=['botorch_sgp', 'sparse_gp'])


def _load_tabpfn():
    import tabpfn  # noqa: F401

    from .adapters.tabpfn import TabPFNClassifierAdapter, TabPFNRegressorAdapter
    register_model(TabPFNRegressorAdapter, key='tabpfn.regressor', aliases=['tabpfn'])
    register_model(TabPFNClassifierAdapter, key='tabpfn.classifier', aliases=['tabpfn_clf'])


def _load_keras():
    import tensorflow  # noqa: F401 - fail HERE with the real reason

    from .adapters.keras import KerasMLPAdapter
    register_model(KerasMLPAdapter, key='keras.mlp', aliases=['keras', 'tf.mlp'])


def _torch_group(module, names_keys_aliases):
    def _loader():
        import importlib
        mod = importlib.import_module(module, package=__name__)
        for cls_name, key, aliases in names_keys_aliases:
            register_model(getattr(mod, cls_name), key=key, aliases=list(aliases))
    return _loader


_register_optional(('lgbm.regressor', 'lgbm.classifier'), ('lightgbm',), _load_lgbm)
_register_optional(('catboost.regressor', 'catboost.classifier'), ('catboost',), _load_catboost)
_register_optional(('xgboost.xgbregressor', 'xgboost.xgbclassifier'), ('xgboost',), _load_xgboost)
_register_optional(('botorch.gp', 'botorch.sparse_gp'), ('torch', 'botorch'), _load_botorch)
_register_optional(('tabpfn.regressor', 'tabpfn.classifier'), ('tabpfn',), _load_tabpfn)
_register_optional(('keras.mlp',), ('tensorflow',), _load_keras)

_TORCH_GROUPS = [
    ('.adapters.residual_mlp', [('ResidualMLPAdapter', 'pytorch.residual_mlp', ['residual_mlp'])]),
    ('.adapters.geometric_residual_mlp', [('GeometricResidualMLPAdapter', 'pytorch.geom_residual_mlp',
                                           ['geom_residual_mlp', 'geometric_residual_mlp'])]),
    ('.adapters.mlp_classifier', [('MLPClassifierAdapter', 'pytorch.mlp_classifier', ['mlp_classifier'])]),
    ('.adapters.mlp_ensemble', [('MLPEnsembleAdapter', 'pytorch.mlp_ensemble', ['mlp_ensemble', 'ensemble_mlp'])]),
    ('.adapters.cnn', [('CNN1DAdapter', 'pytorch.cnn1d', ['cnn1d'])]),
    ('.adapters.rnn', [('LSTMAdapter', 'pytorch.lstm', ['lstm']),
                       ('GRUAdapter', 'pytorch.gru', ['gru'])]),
    ('.adapters.fno', [('FNO1dAdapter', 'pytorch.fno1d', ['fno1d', 'fno'])]),
    ('.adapters.deeponet', [('DeepONetAdapter', 'pytorch.deeponet', ['deeponet'])]),
    ('.adapters.simformer', [('SimformerAdapter', 'pytorch.simformer', ['simformer'])]),
    ('.adapters.convnext_unet', [('ConvNeXtUNetAdapter', 'pytorch.convnext_unet', ['convnext_unet', 'cunet'])]),
    ('.adapters.lenet', [('LeNet5Adapter', 'pytorch.lenet5', ['lenet5', 'lenet'])]),
    ('.adapters.resnet', [('ResNet20Adapter', 'pytorch.resnet20', ['resnet20']),
                          ('ResNet56Adapter', 'pytorch.resnet56', ['resnet56'])]),
    ('.adapters.kan', [('KANRegressorAdapter', 'pytorch.kan', ['kan']),
                       ('KANClassifierAdapter', 'pytorch.kan_classifier', ['kan_classifier'])]),
    ('.adapters.ft_transformer', [('FTTransformerAdapter', 'pytorch.ft_transformer', ['ft_transformer', 'ftt']),
                                  ('FTTransformerClassifierAdapter', 'pytorch.ft_transformer_classifier',
                                   ['ft_transformer_classifier'])]),
    ('.adapters.vit', [('ViTAdapter', 'pytorch.vit', ['vit'])]),
    ('.adapters.alexnet', [('AlexNetAdapter', 'pytorch.alexnet', ['alexnet'])]),
    ('.adapters.fno2d', [('FNO2dAdapter', 'pytorch.fno2d', ['fno2d'])]),
    ('.adapters.unet', [('UNetAdapter', 'pytorch.unet', ['unet'])]),
    ('.adapters.vae', [('VAEAdapter', 'pytorch.vae', ['vae'])]),
    ('.adapters.ddpm', [('DDPMAdapter', 'pytorch.ddpm', ['ddpm'])]),
    ('.adapters.cgan', [('CGANAdapter', 'pytorch.cgan', ['cgan'])]),
]

for _module, _entries in _TORCH_GROUPS:
    _register_optional(tuple(key for _, key, _a in _entries), ('torch',),
                       _torch_group(_module, _entries), needs_torch=True)


def __getattr__(name: str):  # PEP 562: avoid importing TensorFlow/GPflow at import time
    if name == "GPFLOW_AVAILABLE":
        from .gpflow import gpflow_runtime_available

        return gpflow_runtime_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GPFLOW_AVAILABLE",
    "MODEL_REGISTRY",
    "PYTORCH_AVAILABLE",
    "BaseModelAdapter",
    "EnsemblePrediction",
    "FNNEnsemble",
    "GPRModel",
    "GPflowGPRAdapter",
    "GPflowMultiKernelAdapter",
    "GradientBoostingClassifierAdapter",
    "GradientBoostingRegressorModel",
    "LogisticRegressionAdapter",
    "MLPModel",
    "PyTorchMLPAdapter",
    "RandomForestClassifierAdapter",
    "RandomForestModel",
    "SklearnClassifierAdapter",
    "SklearnRegressorAdapter",
    "compare_training_histories",
    "create_model",
    "list_models",
    "load_training_history",
    "plot_training_history",
    "register_model",
    "registration_report",
    "registration_table",
]
