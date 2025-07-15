
<div align="center">
  <img src="data/logos/surge_logo_panoramic.png" alt="SURGE Logo" width="800"/>
</div>

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
```

## 🧪 Testing and Development

### Running Tests

SURGE includes a comprehensive test suite covering all core functionality:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=surge --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run specific test file
pytest tests/test_enhanced_models.py -v
```

### Continuous Integration

The project uses GitHub Actions for automated testing across multiple Python versions (3.8-3.11). The CI pipeline includes:

- **Testing**: Full test suite with pytest and coverage reporting
- **Linting**: Code quality checks with ruff, black, and isort
- **Type Checking**: Static type analysis with mypy
- **Documentation**: Automated docs building and validation

### Development Setup

For contributors, set up the development environment:

```bash
# Clone and install in development mode
git clone https://github.com/your-username/SURGE.git
cd SURGE
pip install -e ".[dev]"

# Install pre-commit hooks for code quality
pre-commit install

# Run the full development check
black surge/ tests/     # Format code
isort surge/ tests/     # Sort imports  
ruff check surge/ tests/ # Lint code
mypy surge/             # Type checking
pytest                  # Run tests
```

### Code Quality Standards

- **Code Formatting**: Black with 88-character line length
- **Import Sorting**: isort with black profile
- **Linting**: ruff for fast Python linting
- **Type Hints**: mypy for static type checking
- **Testing**: pytest with >90% coverage target

## 📊 Performance and Resource Monitoring

SURGE includes built-in resource monitoring for training and inference:

```python
from surge import MLTrainer

trainer = MLTrainer(model_type='pytorch_mlp')
trainer.fit(X, y, monitor_resources=True)

# View resource usage during training
trainer.plot_resource_usage()

# Get detailed performance metrics
performance = trainer.get_performance_summary()
print(performance)
```

## 🔧 Hyperparameter Tuning

Multiple optimization strategies are supported:

```python
# Random search (memory efficient)
best_params = trainer.tune(
    X, y, 
    method='random', 
    n_trials=50,
    memory_efficient=True
)

# Bayesian optimization with scikit-optimize
best_params = trainer.tune(
    X, y, 
    method='bayesian', 
    n_trials=30
)

# Advanced optimization with Optuna + BoTorch
best_params = trainer.tune(
    X, y, 
    method='optuna_botorch', 
    n_trials=100,
    study_name='my_optimization'
)
```

## 📚 Examples and Notebooks

Check the `notebooks/` directory for comprehensive examples:

- **RF_Heating_Surrogate_Demo.ipynb**: Complete workflow for RF heating data
- **Demo1.ipynb**: Basic usage and model comparison

Example scripts in `examples/`:

- **simple_optuna_demo.py**: Hyperparameter optimization
- **comprehensive_optimization_demo.py**: Multi-method comparison
- **bayesian_optimization_demo.py**: Advanced Bayesian techniques

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes with appropriate tests
5. Ensure all tests pass: `pytest`
6. Run code quality checks: `pre-commit run --all-files`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with scikit-learn, PyTorch, TensorFlow, and GPflow
- Hyperparameter optimization powered by Optuna and scikit-optimize
- Resource monitoring with psutil
