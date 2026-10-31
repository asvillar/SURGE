# -*- coding: utf-8 -*-
"""SURGE package public exports."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Álvaro Sánchez Villar"

from .metrics import summarize
from .model import MODEL_REGISTRY, PYTORCH_AVAILABLE
from .engine import (
    EngineRunConfig,
    ModelRunResult,
    ModelSpec,
    SurrogateEngine,
)
from .model.base import BaseModelAdapter
from .model.registry import ModelRegistry, registry_summary
from .dataset import SurrogateDataset
from .datasets import M3DC1Dataset, XGCDataset
from .utils import get_data_path, setup_surge_path
from .datagen import DataGenerator
from .workflow.run import run_surrogate_workflow
from .workflow.spec import HPOConfig, ModelConfig, SurrogateWorkflowSpec

# Adapter registration happens eagerly inside `surge.model.__init__`
# (see line 10 above); no extra import is required.


def __getattr__(name: str):  # PEP 562: lazy GPflow probe (keeps lightweight CLIs quiet)
    if name == "GPFLOW_AVAILABLE":
        from . import model as _model

        return _model.GPFLOW_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Optional dataset utilities
try:
    from .datagen.utils import (
        validate_dataset,
        get_dataset_summary,
        dataframe_to_pytorch_dataset,
        print_validation_report,
    )
    DATASET_UTILS_AVAILABLE = True
except ImportError:
    DATASET_UTILS_AVAILABLE = False

# M3DC1 HDF5 loaders live in scripts/m3dc1/loader.py and are NOT part of
# the installable wheel: they are application-specific HDF5 readers that
# depend on the scripts/m3dc1 tree alongside a working SURGE checkout.
# `M3DC1Dataset` (exported below) imports them lazily when a dataset is
# actually constructed, so pip-installed users get a clean `import surge`
# and a clear ImportError only if they try to use the M3DC1 loader
# without the scripts/ tree on sys.path. Prior versions attempted an
# eager sys.path injection here; that silently failed for wheel installs
# and polluted sys.path from import time, so it was removed.
M3DC1_LOADER_AVAILABLE = False

__all__ = [
    "SurrogateEngine",
    "SurrogateDataset",
    "M3DC1Dataset",
    "XGCDataset",
    "EngineRunConfig",
    "ModelSpec",
    "ModelRunResult",
    "BaseModelAdapter",
    "ModelRegistry",
    "registry_summary",
    "MODEL_REGISTRY",
    "run_surrogate_workflow",
    "SurrogateWorkflowSpec",
    "ModelConfig",
    "HPOConfig",
    "summarize",
    "setup_surge_path",
    "get_data_path",
    "DataGenerator",
    "__version__",
    "__author__",
    "PYTORCH_AVAILABLE",
    "GPFLOW_AVAILABLE",
]

# Add dataset utilities if available
if DATASET_UTILS_AVAILABLE:
    __all__.extend([
        "validate_dataset",
        "get_dataset_summary",
        "dataframe_to_pytorch_dataset",
        "print_validation_report",
    ])

# M3DC1 HDF5 loader functions are intentionally NOT exported: they are
# application-specific and live under scripts/m3dc1/. Use `M3DC1Dataset`
# from `surge.datasets.m3dc1` (exported below) which imports them lazily.

# Print availability status when package is imported
def _print_availability():
    """Print information about available model backends."""
    print("🚀 SURGE - Surrogate Unified Robust Generation Engine")
    print(f"📦 Version: {__version__}")
    print(f"👨‍💻 Author: {__author__}")
    print()
    print("Available Model Backends:")
    print("  🔹 Scikit-learn: ✅ (RFR, MLP, GPR)")
    print(f"  🔹 PyTorch: {'✅' if PYTORCH_AVAILABLE else '❌'} (Enhanced MLP)")
    print(f"  🔹 GPflow: {'✅' if GPFLOW_AVAILABLE else '❌'} (Advanced GP)")
    print()
    if not PYTORCH_AVAILABLE:
        print("💡 Install PyTorch for enhanced MLP models: pip install torch")
    if not GPFLOW_AVAILABLE:
        print("💡 Install GPflow for advanced GP models: pip install gpflow")
    print()

# Uncomment the line below to show availability on import
# _print_availability()
