# -*- coding: utf-8 -*-
"""SURGE package public exports."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Álvaro Sánchez Villar"

from .metrics import summarize
from .model import GPFLOW_AVAILABLE, MODEL_REGISTRY, PYTORCH_AVAILABLE
from .engine import (
    EngineRunConfig,
    ModelRunResult,
    ModelSpec,
    SurrogateEngine,
)
from .registry import BaseModelAdapter, ModelRegistry, RegistryEntry, registry_summary
from .dataset import SurrogateDataset
from .datasets import MHD-codeDataset, XGCDataset
from .utils import get_data_path, setup_surge_path
from .datagen import DataGenerator
from .workflow.run import run_surrogate_workflow
from .workflow.spec import HPOConfig, ModelConfig, SurrogateWorkflowSpec

# Ensure adapter modules register themselves on import
from .models import adapters as _adapters  # noqa: F401

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

# Optional MHD-code loader (now in scripts/mhd_code/)
try:
    import sys
    from pathlib import Path
    # Add scripts/mhd_code to path for optional imports
    scripts_mhd_code = Path(__file__).parent.parent / "scripts" / "mhd_code"
    if str(scripts_mhd_code) not in sys.path:
        sys.path.insert(0, str(scripts_mhd_code))
    from loader import (
        load_mhd_code_hdf5,
        convert_to_dataframe,
        convert_sdata_complex_v2_to_dataframe,
        read_mhd_code_hdf5_structure,
        read_sdata_complex_v2_structure,
    )
    MHD-code_LOADER_AVAILABLE = True
except ImportError:
    MHD-code_LOADER_AVAILABLE = False

# Import helper functions for convenience
try:
    from .helpers import quick_train, train_and_compare, load_and_train
    HELPERS_AVAILABLE = True
except ImportError:
    HELPERS_AVAILABLE = False

__all__ = [
    "SurrogateEngine",
    "SurrogateDataset",
    "MHD-codeDataset",
    "XGCDataset",
    "EngineRunConfig",
    "ModelSpec",
    "ModelRunResult",
    "BaseModelAdapter",
    "ModelRegistry",
    "RegistryEntry",
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

# Add MHD-code loader if available
if MHD-code_LOADER_AVAILABLE:
    __all__.extend([
        "load_mhd_code_hdf5",
        "convert_to_dataframe",
        "convert_sdata_complex_v2_to_dataframe",
        "read_mhd_code_hdf5_structure",
        "read_sdata_complex_v2_structure",
    ])

# Backward compatibility aliases
MLTrainer = SurrogateEngine  # Legacy alias
SurrogateTrainer = SurrogateEngine  # Legacy alias

# Add helpers if available
if HELPERS_AVAILABLE:
    __all__.extend(["quick_train", "train_and_compare", "load_and_train"])

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
