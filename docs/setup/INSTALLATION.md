# SURGE Installation Guide

## Quick Installation (Recommended)

### 1. Clone the Repository

```bash
git clone <your-surge-repo-url>
cd SURGE
```

### 2. Create Conda Environment

```bash
# Create environment with all dependencies
conda env create -f environment.yml

# Activate the environment
conda activate surge
```

### 3. Install SURGE Package

```bash
# Install SURGE in development mode
pip install -e .
```

The `-e` flag installs SURGE in "editable" mode, meaning:
- ✅ You can import SURGE from anywhere: `from surge import MLTrainer`
- ✅ Changes to the code are immediately available (no reinstall needed)
- ✅ No need to manually set PYTHONPATH or modify sys.path

## Verify Installation

```bash
# Test that SURGE is installed
python -c "import surge; print(surge.__version__)"

# Check available backends
python -c "from surge import PYTORCH_AVAILABLE; print(f'PyTorch: {PYTORCH_AVAILABLE}')"

# Run tests
pytest tests/ -v
```

## Alternative: Manual Path Setup (Not Recommended)

If you prefer not to install SURGE as a package, you can set the environment variable:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export PYTHONPATH="${PYTHONPATH}:/path/to/SURGE"
export SURGE_DIR="/path/to/SURGE"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Directory Structure After Installation

```
SURGE/                          # Your cloned repository
├── environment.yml             # Main conda environment
├── envs/                       # Alternative environments
│   ├── environment_minimal.yml
│   └── environment_gpu.yml
├── setup.py                    # Python package setup
├── surge/                      # Main package code
│   ├── __init__.py
│   ├── trainer.py
│   ├── models.py
│   └── ...
├── tests/                      # Test suite
├── examples/                   # Example scripts
└── docs/                       # Documentation
```

## How Python Finds SURGE

After `pip install -e .`, Python automatically knows where SURGE is because:

1. A `.egg-link` file is created in your conda environment's `site-packages/`
2. This file points to your SURGE directory
3. Python can now import SURGE from anywhere

## Uninstalling

```bash
# Remove SURGE package
pip uninstall surge-surrogate

# Remove conda environment
conda deactivate
conda env remove -n surge
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'surge'"

**Solution**: Install SURGE as a package:
```bash
cd /path/to/SURGE
pip install -e .
```

### "Command 'pip install -e .' fails"

**Solution**: Make sure you're in the SURGE directory and conda environment is activated:
```bash
conda activate surge
cd /path/to/SURGE
pip install -e .
```

### "ImportError: cannot import name 'MLTrainer'"

**Solution**: Check your import statement:
```python
# Correct
from surge.trainer import MLTrainer

# Also correct
from surge import MLTrainer
```

## Next Steps

1. ✅ Repository cloned
2. ✅ Environment created
3. ✅ SURGE installed
4. 👉 **Try the examples**: `python examples/simple_optuna_demo.py`
5. 👉 **Read the docs**: Check `docs/` for tutorials
6. 👉 **Run tests**: `pytest tests/ -v`

---

**Ready to build surrogate models? 🚀**
