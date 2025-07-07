# surge/src/__init__.py

__version__ = "0.1.0"
__author__ = "Álvaro Sánchez Villar"

# Expose the core classes/functions at package level
from .trainer import SurrogateTrainer
from .metrics import summarize

# Import model availability information
from .models import PYTORCH_AVAILABLE, GPFLOW_AVAILABLE

__all__ = [
    "SurrogateTrainer",
    "summarize",
    "__version__",
    "__author__",
    "PYTORCH_AVAILABLE",
    "GPFLOW_AVAILABLE",
]

# Print availability status when package is imported
def _print_availability():
    """Print information about available model backends."""
    print("🚀 SURGE - Surrogate Unified Robust Generation Engine")
    print(f"📦 Version: {__version__}")
    print(f"👨‍💻 Author: {__author__}")
    print()
    print("Available Model Backends:")
    print(f"  🔹 Scikit-learn: ✅ (RFR, MLP, GPR)")
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
