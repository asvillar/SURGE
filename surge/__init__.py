# surge/src/__init__.py

__version__ = "0.1.0"
__author__ = "Álvaro Sánchez Villar"  # Expose the core classes/functions at package level
from .metrics import summarize
from .models import GPFLOW_AVAILABLE, PYTORCH_AVAILABLE, MODEL_REGISTRY
from .engine import SurrogateEngine
from .dataset import SurrogateDataset
from .utils import setup_surge_path, get_data_path
from .datagen import DataGenerator

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
