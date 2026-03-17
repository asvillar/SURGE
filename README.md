
<div align="center">
  <img src="data/logos/surge_logo_panoramic.png" alt="SURGE Logo" width="800"/>
</div>

# SURGE – Surrogate Unified Robust Generation Engine

**SURGE** is a modular AI/ML framework for building fast, accurate, and uncertainty-aware surrogate models that emulate complex scientific simulations. The current earlier refactor collapses the entire engine into the `surge/` package, providing a single configuration-driven workflow that spans dataset ingestion, splitting/standardization, model training, Optuna/Bayesian HPO, validation, artifact export, and visualization.

## 🔧 Features

- Unified workflow runner (`surge.workflow.run.run_surrogate_workflow`)
- Dataset auto-detection + metadata overrides via `surge.dataset.SurrogateDataset`
- Built-in model registry (RFR, Torch MLP with MC-Dropout, GPflow GPR)
- Optuna-based HPO with TPE or BoTorch samplers
- Standardized artifact layout under `runs/<tag>/` (models, scalers, predictions, metrics, HPO logs, env snapshot, git rev)
- Visualization helpers (`surge.viz`) for GT vs. prediction density, per-mode scatter, violin/SNR panels, etc.
- **kinetic-code support**: Datastreamset evaluation, drift detection (OOD inputs/outputs, accuracy drop), SHAP on target, input range evolution plots.

## 🚀 Getting Started

> **Prereqs**
> - Python 3.11 (see `envs/surge-env-devel.yml` for Conda env)
> - Dataset: `data/datasets/device/case_data` + `mhd_code_metadata.yaml`

### 1. Clone & install (development mode)

```bash
git clone https://github.com/your-username/SURGE.git
cd SURGE
pip install -e ".[dev]"
# or conda env create -f envs/surge-env-devel.yml
```

### 2. Run the baseline MHD-code workflow (1k-sample smoke test)

```bash
conda run -n surge python -m examples.mhd_code_workflow \
    --spec configs/mhd_code_demo.yaml \
    --run-tag mhd_code_demo_cli
```

### 3. Run the augmented workflow (all 9,981 samples, 50 HPO trials/model)

```bash
conda run -n surge python -m examples.mhd_code_workflow \
    --spec configs/mhd_code_demo_augmented.yaml \
    --run-tag mhd_code_demo_full
```

### 4. Regenerate the R² > 0.88 growth-rate surrogates

To rebuild the same surrogates as in `runs/mhd_code_aug_r75/` (full dataset, 75 HPO trials per model):

```bash
conda run -n surge python -m examples.mhd_code_workflow \
    --spec configs/mhd_code_aug_r75.yaml \
    --run-tag mhd_code_aug_r75_rebuild
```

Artifacts go to `runs/mhd_code_aug_r75_rebuild/`. Use `--run-tag mhd_code_aug_r75` and set `overwrite_existing_run: true` in the spec to replace the original run.

### 5. Explore in the notebook

```bash
conda run -n surge jupyter lab notebooks/MHD-code_demo.ipynb
```

The notebook mirrors the CLI workflow (baseline spec by default) and layers on dataset analysis, timing/metric tables, GT vs. prediction density plots, per-mode scatter, violin/SNR dashboards, etc. Switch to the augmented config by changing `spec_path = Path("configs/mhd_code_demo_augmented.yaml")`.

### 6. Programmatic access

```python
from pathlib import Path
import yaml
from surge.workflow.spec import SurrogateWorkflowSpec
from surge.workflow.run import run_surrogate_workflow

spec_path = Path("configs/mhd_code_demo.yaml")
spec = SurrogateWorkflowSpec.from_dict(yaml.safe_load(spec_path.read_text()))
spec.run_tag = "mhd_code_programmatic"
summary = run_surrogate_workflow(spec)
print(summary["models"][0]["metrics"]["test"])
```

### Workflow outputs

Each run produces a self-contained directory under `runs/<tag>/`:

```
runs/<tag>/
├── models/                         # joblib-serialized adapters
├── scalers/                        # StandardScaler bundles for X/Y
├── predictions/<model>_<split>.csv # y_true/y_pred (+ uncertainty JSON)
├── metrics.json                    # aggregate metrics
├── workflow_summary.json           # dataset/split/registry/hardware summary
├── spec.yaml                       # copy of the workflow spec
├── environment.txt                 # pip freeze
└── git.txt                         # git describe for provenance
```

Point plotting/notebook scripts at these CSV/JSON artifacts to build custom dashboards.

### Legacy API

`SurrogateTrainer`, `MLTrainer`, and the old module layout remain importable for backwards compatibility, but new work should go through the unified workflow described above.

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

SURGE includes comprehensive testing infrastructure:

- **Testing**: Full test suite with pytest and coverage reporting
- **Code Quality**: Manual code quality checks with ruff, black, and isort
- **Type Checking**: Static type analysis with mypy available
- **Documentation**: Automated docs building with Sphinx

### Development Setup

For contributors, set up the development environment:

```bash
# Clone and install in development mode
git clone https://github.com/your-username/SURGE.git
cd SURGE
pip install -e ".[dev]"

# Run development quality checks
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

## 🔧 Hyperparameter Tuning

Optuna search spaces live directly inside each `ModelConfig` (see the configs under `configs/`). Enable BoTorch by setting `sampler: botorch` in the YAML or by passing `HPOConfig(sampler="botorch")` programmatically. Workflow summaries record the winning trial and persist the full study JSON under `runs/<tag>/hpo/<model>.json`.

## 📚 Examples and Notebooks

Check the `notebooks/` directory for comprehensive examples:

- **`MHD-code_demo.ipynb`**: End-to-end device MHD-code workflow (baseline spec, optional augmented spec)
- **`RF_Heating_Surrogate_Demo.ipynb`**: Legacy training utilities on example-data datasets

Example scripts in `examples/`:

- **`examples/mhd_code_workflow.py`**: CLI wrapper for the YAML-driven workflow
- **`examples/target_.py`**: Lightweight RF heating surrogate example
- Additional Optuna/Bayesian comparison scripts under `examples/`

## 📊 Datasets

SURGE includes real-world datasets for immediate experimentation:

### RF Heating Data (data/datasets/)
- **`example-data/target_.pkl`**: Electron heating power profiles from Reference simulations (~41MB)
- **`example-data/PwIF_.pkl`**: Ion/fast particle heating profiles (~41MB)

Perfect for:
```python
from surge import SurrogateTrainer

# Load RF heating dataset
trainer = SurrogateTrainer()
inputs, outputs = trainer.load_dataset_pickle('data/datasets/example-data/target_.pkl')
# Automatically identifies 100+ input parameters → 60+ output power profiles
```

See `data/datasets/README.md` for detailed dataset documentation.

### device MHD-code Campaign (data/datasets/device/)

- **`case_data`**: Curated growth-rate dataset used in the new MHD-code workflow demos.
- **`mhd_code_metadata.yaml`**: Input/output definitions consumed by `SurrogateDataset`.
- **`configs/mhd_code_demo.yaml`**: 1k-sample smoke workflow.
- **`configs/mhd_code_demo_augmented.yaml`**: Full 9,981-sample workflow with 50-trial HPO per model.

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

## MHD-code Surrogate Workflow (Recap)

1. **Dataset** – `data/datasets/device/case_data` + `mhd_code_metadata.yaml`
2. **Spec** – edit `configs/mhd_code_demo.yaml` (baseline) or `configs/mhd_code_demo_augmented.yaml` (full sweep)
3. **Workflow run** – `python -m examples.mhd_code_workflow --spec ... --run-tag ...`
4. **Notebook** – `notebooks/MHD-code_demo.ipynb` for visual inspection (switch specs via `spec_path`)
5. **Artifacts** – `runs/<tag>/` contains everything needed for reproducibility and publication plots

---

## kinetic-code target Workflow

SURGE supports kinetic-code (kinetic-code-O) target surrogate training and generalization analysis. Full documentation: [`docs/stacked/XGC_FEATURES_2025-03.md`](docs/stacked/XGC_FEATURES_2025-03.md).

### 1. Train an kinetic-code surrogate

```bash
surge run configs/xgc_target_set1.yaml --run-tag xgc_target_set1_v3
```

Artifacts go to `runs/xgc_target_set1_v3/` (models, scalers, `train_data_ranges.json`, etc.).

### 2. Visualization and inference comparison

```bash
surge viz runs/xgc_target_set1_v3
```

Produces inference comparison plots (GT vs prediction), HPO convergence, and R² summary.

### 3. SHAP importance (target)

```bash
surge viz runs/xgc_target_set1_v3 --shap
```

SHAP targets **output_1 (target)** by default. Override with `--shap-output-index 0`. Outputs: `plots/shap_<model>/shap_summary_tree_*.png`, including grouped bar (14 stages + 5 extra columns).

### 4. Datastreamset evaluation and drift detection

Evaluate the trained model on held-out contiguous regions (datastreamsets) of the full dataset:

```bash
surge viz runs/xgc_target_set1_v3 --datastreamset-eval
```

**Options:**
- `--datastreamset-size 50000` – rows per datastreamset (default: 50000)
- `--datastreamset-max 10` – max datastreamsets to evaluate (default: 10)

**Outputs:**
- `plots/datastreamset_evaluation_r2_rmse.png` – R², RMSE, in-range bar chart, input range evolution
- `plots/datastreamset_evaluation_stats.json` – per-model, per-datastreamset metrics and OOD info
- Drift summary in console (OOD inputs/outputs, accuracy drop)

**Drift policy:** [`docs/stacked/DRIFT_DETECTION_POLICY.md`](docs/stacked/DRIFT_DETECTION_POLICY.md)

### 5. Train-on-datastreamset (generalization analysis)

Train a simple RF on one datastreamset and evaluate on others:

```bash
python scripts/xgc_datastreamset_generalization.py \
  --data-dir /path/to/stacked_data \
  --train-on-datastreamset --hpo-trials 10 \
  --datastreamset-size 50000 --max-datastreamsets 10
```

### 6. Combined viz (SHAP + datastreamset eval)

```bash
surge viz runs/xgc_target_set1_v3 --shap --datastreamset-eval
```

### kinetic-code workflow outputs

```
runs/<tag>/
├── models/                    # RF, MLP adapters (load_model_compat for version mismatch)
├── scalers/                   # input/output StandardScaler
├── train_data_ranges.json     # min/max for in-distribution checks
├── workflow_summary.json
└── plots/
    ├── inference_comparison_*.png
    ├── hpo_convergence.png
    ├── datastreamset_evaluation_r2_rmse.png   # R² zoom, input range evolution
    ├── datastreamset_evaluation_stats.json
    └── shap_<model>/                          # SHAP on target
```
