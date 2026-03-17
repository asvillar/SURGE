# SURGE Documentation

## Navigation

Use this page to navigate the SURGE documentation and repository.

### Getting Started

| Document | Description |
|----------|-------------|
| [SURGE_OVERVIEW.md](SURGE_OVERVIEW.md) | Framework overview, philosophy, key modules |
| [quickstart.rst](quickstart.rst) | Installation and first run |
| [setup/QUICKSTART.md](setup/QUICKSTART.md) | Environment setup quick reference |
| [setup/INSTALLATION.md](setup/INSTALLATION.md) | Full installation |

### Framework Usage

| Document | Description |
|----------|-------------|
| [overview.rst](overview.rst) | Architecture, entry points, dataset/engine |
| [ADDING_NEW_MODEL_ADAPTER.md](ADDING_NEW_MODEL_ADAPTER.md) | How to add a new model adapter |
| [QUICK_START_NEW_MODEL.md](QUICK_START_NEW_MODEL.md) | Quick start for new models |
| [MODEL_EXTENSION_SUMMARY.md](MODEL_EXTENSION_SUMMARY.md) | Model extension summary |

### Workflows & Data

| Document | Description |
|----------|-------------|
| [BATCH_GENERATION_WORKFLOW.md](BATCH_GENERATION_WORKFLOW.md) | Batch generation workflow |
| [BATCH_GENERATION_RECIPE.md](BATCH_GENERATION_RECIPE.md) | Batch generation recipe |
| [DATAGEN_USER_MANUAL.md](DATAGEN_USER_MANUAL.md) | Data generation manual |

### Application-Specific

| Document | Description |
|----------|-------------|
| [mhd_code/](mhd_code/) | MHD-code-specific docs (delta-p spectra, etc.) |

### Development & Modifications

| Document | Description |
|----------|-------------|
| [dev/README.md](dev/README.md) | Modification tracking, status, potential features |

### Build Documentation

```bash
make -C docs html
# Open docs/_build/html/index.html
```

## Repository Layout

```
SURGE/
├── surge/           # Core framework
├── configs/         # YAML configs (templates, application-specific)
├── examples/        # Example scripts
├── notebooks/       # Jupyter notebooks
├── scripts/         # Utility scripts (e.g. mhd_code loaders)
├── tests/           # Test suite
└── docs/            # Documentation (this folder)
```
