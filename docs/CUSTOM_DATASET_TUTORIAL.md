# Custom dataset tutorial

Train one or more SURGE surrogate models on **your own tabular data** stored as
CSV, pickle (`.pkl`), or HDF5 (`.h5`). This page is the canonical guide for
that workflow.

**Try it now (from the repo root, with SURGE installed):**

```bash
python examples/custom_dataset_tutorial.py
ls runs/custom_dataset_tutorial/
```

That command writes a small synthetic dataset, runs the bundled YAML spec, and
prints a JSON summary. The same paths and files are exercised in CI via
`tests/test_custom_dataset_tutorial.py`.

---

## What SURGE does for you

SURGE runs the full surrogate pipeline in one call:

1. Load your file
2. Split into train / validation / test
3. Standardize inputs (and optionally outputs)
4. Train each model you list
5. Write artifacts under `runs/<run_tag>/` (metrics, predictions, scalers, models, config snapshot)

You describe *what* to run in a **workflow spec** — a YAML file or a Python
`SurrogateWorkflowSpec`.

> **Note:** The top-level `surge` CLI (`surge run`, `surge list`) is for
> **built-in benchmarks** (CIFAR, QLKNN, Iris, …). For your own CSV/PKL/H5
> file, use the workflow API described here — not `surge run -b ...`.

---

## 0. Install

From a SURGE checkout:

```bash
cd SURGE
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"    # torch = neural nets; dev adds pytest + h5py
```

Verify:

```bash
python -c "import surge; print(surge.__version__)"
```

---

## 1. Prepare your data

SURGE expects a **tabular dataset**: one row per sample, columns = features and
targets.

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Easiest to inspect |
| Pickle | `.pkl`, `.pickle` | pandas DataFrame on disk |
| Parquet | `.parquet` | Good for large tables |
| HDF5 | `.h5`, `.hdf5` | 2D numeric table (PyTables `key=` or auto-detected 2D dataset) |

Example layout:

```text
input_a,input_b,input_c,output_x,output_y
1.2,0.5,-0.3,0.61,-0.12
...
```

**Multi-output** is supported — list every target column under `outputs`.

**Tips**

- Remove or fix NaNs before training.
- For nested simulation HDF5 (MHD-code batch trees, kinetic-code directories), use the
  specialized loaders documented in [`SURGE_OVERVIEW.md`](SURGE_OVERVIEW.md);
  this tutorial focuses on flat tabular files.

### Bundled sample data

The repo ships a generator used by this tutorial:

```bash
# Write examples/data/tutorial_sample.csv only
python examples/custom_dataset_tutorial.py --prepare-only

# Also try pickle or HDF5
python examples/custom_dataset_tutorial.py --format pkl --prepare-only
python examples/custom_dataset_tutorial.py --format h5 --prepare-only
```

---

## 2. Declare inputs and outputs

SURGE can **auto-detect** columns when names follow scientific conventions
(`input_*`, `x_*`, `target_*`, `y_*`, `output_*`, …). For generic column names,
declare I/O explicitly.

### Option A — metadata YAML (recommended)

The bundled file `examples/configs/custom_dataset_meta.yaml`:

```yaml
inputs:
  - input_a
  - input_b
  - input_c
outputs:
  - output_x
  - output_y
```

Reference it from your workflow spec with `metadata_path:`.

### Option B — inline overrides

```yaml
metadata_overrides:
  inputs: [input_a, input_b, input_c]
  outputs: [output_x, output_y]
```

Use Option A when metadata is shared across runs; Option B for quick one-offs.

---

## 3. Create a workflow spec (YAML)

The bundled tutorial spec is `examples/configs/custom_dataset_tutorial.yaml`:

```yaml
dataset_path: examples/data/tutorial_sample.csv
dataset_format: auto
metadata_path: examples/configs/custom_dataset_meta.yaml

test_fraction: 0.2
val_fraction: 0.1
standardize_inputs: true
standardize_outputs: false
seed: 42

output_dir: .
run_tag: custom_dataset_tutorial
overwrite_existing_run: true

resources:
  device: cpu
  num_workers: 2

models:
  - key: sklearn.random_forest
    name: rf_tutorial
    params:
      n_estimators: 50
      max_depth: 8
      n_jobs: -1
```

Copy this file and edit `dataset_path`, `metadata_path`, `run_tag`, and the
`models` list for your project.

### Pickle example

```yaml
dataset_path: data/my_simulations.pkl
dataset_format: pkl
metadata_path: configs/my_dataset_meta.yaml
run_tag: sims_pkl_rf
models:
  - key: sklearn.random_forest
    params: {n_estimators: 100}
```

### HDF5 example

```yaml
dataset_path: data/tabular_results.h5
dataset_format: h5
metadata_path: configs/my_dataset_meta.yaml
run_tag: h5_run
models:
  - key: sklearn.ridge
    params: {alpha: 1.0}
```

### Train multiple models

Add more entries under `models:` — SURGE trains all of them in one run:

```yaml
models:
  - key: sklearn.random_forest
    name: rf_baseline
    params: {n_estimators: 200}
  - key: pytorch.mlp
    name: mlp_neural
    params:
      hidden_layers: [128, 64]
      learning_rate: 0.001
      n_epochs: 100
      batch_size: 256
```

(`pytorch.mlp` requires `pip install -e ".[torch]"`.)

---

## 4. Run training

### Option A — bundled tutorial script (fastest)

```bash
python examples/custom_dataset_tutorial.py
```

### Option B — generic CLI wrapper

Run from the **repo root** so relative paths in the YAML resolve correctly:

```bash
python examples/custom_dataset_tutorial.py --prepare-only
python examples/run_workflow.py --spec examples/configs/custom_dataset_tutorial.yaml
```

Override run tag or output directory:

```bash
python examples/run_workflow.py \
    --spec examples/configs/custom_dataset_tutorial.yaml \
    --run-tag my_experiment_v2 \
    --output-dir /tmp/surge_runs
```

### Option C — Python API

```python
import yaml
from pathlib import Path
import surge
from surge import SurrogateWorkflowSpec, run_surrogate_workflow

spec = SurrogateWorkflowSpec.from_dict(
    yaml.safe_load(Path("examples/configs/custom_dataset_tutorial.yaml").read_text())
)
summary = run_surrogate_workflow(spec)

model = summary["models"][0]
print(f"test R² = {model['metrics']['test']['r2']:.3f}")
print(f"artifacts → {summary['artifacts']['root']}")
```

---

## 5. List available models

```bash
surge models
```

Common starting points:

| Key | Description | Extra needed |
|-----|-------------|--------------|
| `sklearn.random_forest` | Random forest regressor | core |
| `sklearn.ridge` | Linear ridge | core |
| `sklearn.mlp` | sklearn MLP | core |
| `pytorch.mlp` | PyTorch MLP | `[torch]` |
| `pytorch.residual_mlp` | Residual MLP | `[torch]` |
| `sklearn.gpr` | Gaussian process | core |

See also [`QUICK_START_NEW_MODEL.md`](QUICK_START_NEW_MODEL.md) for adding a
custom adapter.

---

## 6. Optional: hyperparameter optimization (HPO)

Add an `hpo` block to any model entry:

```yaml
models:
  - key: pytorch.mlp
    name: mlp_hpo
    params:
      n_epochs: 50
    hpo:
      enabled: true
      n_trials: 20
      direction: minimize
      metric: val_rmse
      search_space:
        learning_rate:
          type: loguniform
          low: 1e-4
          high: 1e-2
        dropout_rate:
          type: float
          low: 0.0
          high: 0.5
        hidden_layers:
          type: categorical
          choices:
            - [64, 64]
            - [128, 64]
```

See [`examples/qlknn_workflow.py`](../examples/qlknn_workflow.py) for a full
working HPO example.

---

## 7. Inspect results

After a run, artifacts land in `runs/<run_tag>/`:

```text
runs/custom_dataset_tutorial/
├── spec.yaml
├── workflow_summary.json
├── metrics.json
├── train_data_ranges.json
├── scalers/
│   └── inputs.joblib
├── models/
│   └── rf_tutorial.joblib
├── predictions/
└── run.log
```

Quick metrics check:

```bash
python -c "
import json
print(json.dumps(json.load(open('runs/custom_dataset_tutorial/metrics.json')), indent=2))
"
```

### Score new rows (inference round-trip)

```python
import json
import joblib
import pandas as pd

run_dir = "runs/custom_dataset_tutorial"
input_cols = json.load(open(f"{run_dir}/train_data_ranges.json"))["inputs"]["columns"]

scaler = joblib.load(f"{run_dir}/scalers/inputs.joblib")
model = joblib.load(f"{run_dir}/models/rf_tutorial.joblib")

df = pd.read_csv("examples/data/tutorial_sample.csv")[input_cols].head(5)
y_hat = model.predict(scaler.transform(df.values))
print(y_hat)
```

For PyTorch models, load via the adapter class (see the README section
*Round-trip inference from a saved run*).

---

## 8. Explore your dataset before training

```python
from surge import SurrogateDataset

ds = SurrogateDataset.from_path(
    "examples/data/tutorial_sample.csv",
    metadata_path="examples/configs/custom_dataset_meta.yaml",
)
print(ds.summary())
print(ds.input_columns, "→", ds.output_columns)
```

---

## 9. When to use `surge run` instead

Use **`surge run`** only for SURGE's curated benchmark suite:

```bash
surge list
surge run -b iris -m sklearn.random_forest
surge run -b qlknn -m all --seeds 3
```

That path downloads standard datasets and writes leaderboard-style results. It
does **not** accept arbitrary local CSV files.

---

## 10. Checklist for your own project

1. Place data at e.g. `data/my_dataset.csv` (or `.pkl` / `.h5`)
2. Create `configs/my_dataset_meta.yaml` with `inputs` and `outputs`
3. Copy `examples/configs/custom_dataset_tutorial.yaml` → `configs/my_surrogate.yaml`
4. Edit paths and `models`
5. Run:

   ```bash
   python examples/run_workflow.py --spec configs/my_surrogate.yaml
   ```

6. Open `runs/<run_tag>/metrics.json`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Wrong columns picked as inputs/outputs | Add `metadata_path` or `metadata_overrides` |
| `ModuleNotFoundError: torch` | `pip install -e ".[torch]"` |
| Run directory already exists | Set `overwrite_existing_run: true` or change `run_tag` |
| HDF5 won't load | Ensure a 2D dataset exists, or convert to CSV/Parquet |
| Relative paths not found | Run `run_workflow.py` from the repo root (or use absolute paths) |
| Used `surge run -b my_csv` | That CLI is for built-in benchmarks; use this workflow instead |

---

## Related docs

- [`quickstart.rst`](quickstart.rst) — diabetes / California housing demos
- [`SURGE_OVERVIEW.md`](SURGE_OVERVIEW.md) — codebase tour
- [`QUICK_START_NEW_MODEL.md`](QUICK_START_NEW_MODEL.md) — register a new model adapter
- [`setup/WALKTHROUGH.md`](setup/WALKTHROUGH.md) — first-run environment setup
