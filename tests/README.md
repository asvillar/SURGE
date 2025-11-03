# SURGE Test Suite

This directory contains the comprehensive test suite for SURGE. Tests are organized into modular, focused test files that demonstrate specific capabilities.

## Test Organization

### Core Tests

#### `test_core.py`
**Purpose:** Basic imports, initialization, and core functionality  
**Tests:**
- Package imports (SurrogateEngine, SurrogateDataset, etc.)
- Availability flags (PYTORCH_AVAILABLE, GPFLOW_AVAILABLE)
- Basic initialization of classes
- Simple operations

**Run:** `pytest tests/test_core.py -v`

---

#### `test_dataset.py`
**Purpose:** SurrogateDataset class functionality  
**Tests:**
- Loading from various file formats (CSV, pickle, parquet, Excel)
- Auto-detection of input/output columns
- Pattern-based detection (var_1, var_2, ... patterns)
- Manual specification of columns
- Dataset statistics and analysis

**Run:** `pytest tests/test_dataset.py -v`

---

#### `test_engine.py`
**Purpose:** SurrogateEngine workflow - complete training pipeline  
**Tests:**
- Load data (from file, dataset, or DataFrame)
- Train-test split
- Data standardization
- Model initialization (via integer or registry key)
- Cross-validation
- Model training
- Prediction and evaluation
- Saving results (JSON)
- Saving models (joblib with scalers)
- Model loading and inference

**Run:** `pytest tests/test_engine.py -v`

---

### Model Tests

#### `test_models.py`
**Purpose:** Individual model type functionality  
**Tests:**
- Random Forest Regressor (sklearn)
- MLP Regressor (sklearn)
- PyTorch MLP (if available)
- GPflow GPR (if available)
- Model-specific parameters
- Model prediction functionality

**Run:** `pytest tests/test_models.py -v`

---

### Advanced Features

#### `test_helpers.py`
**Purpose:** Helper/convenience functions  
**Tests:**
- `quick_train()` - Rapid prototyping
- `train_and_compare()` - Multi-model training and comparison
- `load_and_train()` - File-based workflow

**Run:** `pytest tests/test_helpers.py -v`

---

#### `test_model_registry.py`
**Purpose:** Model registry extensibility  
**Tests:**
- Listing registered models
- Creating models from registry
- Custom model registration
- Model metadata access

**Run:** `pytest tests/test_model_registry.py -v`

---

#### `test_visualization.py`
**Purpose:** Visualization functions  
**Tests:**
- `plot_regression_results()` - Ground Truth vs Prediction plots
- `plot_all_outputs()` - Multi-output comparison
- Plot saving functionality
- Different dataset subsets (train/test/both)

**Run:** `pytest tests/test_visualization.py -v`

---

#### `test_model_comparison.py`
**Purpose:** Multi-model comparison and benchmarking  
**Tests:**
- Training multiple models side-by-side
- Comparing performance metrics
- Generating comparison plots
- Saving comparison results

**Run:** `pytest tests/test_model_comparison.py -v`

---

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_engine.py -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=surge --cov-report=html
```

### Run Only Fast Tests (Skip Slow Integration Tests)
```bash
pytest tests/ -m "not slow"
```

### Run Tests for Specific Feature
```bash
# Test only visualization
pytest tests/test_visualization.py tests/test_model_comparison.py

# Test only models
pytest tests/test_models.py tests/test_model_registry.py
```

## Test Data

Tests use synthetic datasets generated with fixed random seeds for reproducibility. No external data files are required.

## Test Outputs

Some tests generate outputs:
- **Plots:** Saved to `tests/plots/` directory
- **Models:** Saved to temporary directories (cleaned up after tests)
- **Results:** Saved to temporary directories (cleaned up after tests)

## Adding New Tests

When adding new functionality:
1. Add tests to the appropriate existing test file
2. If testing a new major feature, create a new test file following the naming convention
3. Document the test file purpose in this README
4. Ensure tests are modular and focused on a single capability

## Test Philosophy

- **Modular:** Each test file focuses on a specific capability
- **Essential:** Tests demonstrate core functionality users need
- **Non-redundant:** Avoid duplicate tests across files
- **Clear:** Test names and structure make capabilities obvious
- **Maintainable:** Easy to understand what each test demonstrates

