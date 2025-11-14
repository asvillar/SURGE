# surge/src/__init__.py

__version__ = "0.1.0"
__author__ = "Álvaro Sánchez Villar"  # Expose the core classes/functions at package level
from .metrics import summarize
from .model import GPFLOW_AVAILABLE, PYTORCH_AVAILABLE, MODEL_REGISTRY
from .engine import SurrogateEngine
from .dataset import SurrogateDataset
from .utils import setup_surge_path, get_data_path
from .datagen import DataGenerator

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
        read_mhd_code_hdf5_structure,
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
    "summarize",
    "setup_surge_path",
    "get_data_path",
    "DataGenerator",
    "MODEL_REGISTRY",
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
        "read_mhd_code_hdf5_structure",
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
