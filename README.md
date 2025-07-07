
# SURGE – Surrogate Unified Robust Generation Engine

**SURGE** is a modular AI/ML framework for building fast, accurate, and uncertainty-aware surrogate models that emulate complex scientific simulations. Designed for flexibility and scientific rigor, it supports ensemble regressors, neural networks, and Gaussian processes, streamlining the development and deployment of surrogates for inference, optimization, and control.

## 🔧 Features

- Unified interface for RFR, MLP, and GPR
- Configurable training and validation workflows
- Uncertainty quantification support (GPR)
- Cross-validation and performance reporting
- Ready for scientific computing environments

## 🚀 Getting Started

### Installation

#### From PyPI (when published)
```bash
pip install surge-surrogate
```

#### For Development
Clone the repository and install in development mode:

```bash
git clone https://github.com/your-username/SURGE.git
cd SURGE
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

### Quick Usage

```python
from surge import SurrogateTrainer
import numpy as np

# Generate sample data
X = np.random.random((100, 5))
y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

# Create and train a surrogate model
trainer = SurrogateTrainer(model_type='rfr', n_estimators=100)
trainer.fit(X, y)

# Perform cross-validation
results = trainer.cross_validate(n_splits=5)
print(f"Average R² score: {results['r2_mean']:.3f}")

# Save the model
trainer.save('my_model')
