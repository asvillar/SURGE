# surge/src/__init__.py

__version__ = "0.1.0"
__author__  = "Álvaro Sánchez Villar"

# Expose the core classes/functions at package level
from .trainer import SurrogateTrainer
from .metrics import summarize

__all__ = [
    "SurrogateTrainer",
    "summarize",
    "__version__",
    "__author__",
]
