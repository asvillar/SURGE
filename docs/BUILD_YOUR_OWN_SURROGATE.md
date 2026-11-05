# Build your own surrogate

Two workflows live under this guide:

| Goal | Jump to |
|------|---------|
| Train **existing** SURGE models on **your data** (CSV, PKL, H5) | [Part I — Your dataset](#part-i--train-on-your-own-data) |
| **Embed a new model type** into SURGE (new adapter) | [Part II — Embed a new model](#part-ii--embed-a-new-model-in-surge) |

> **CLI note:** `surge run` is for **built-in benchmarks** only. Custom data and
> custom workflow specs use `run_surrogate_workflow` / `examples/run_workflow.py`.

---

# Part I — Train on your own data

## Try it now

From the repo root (SURGE installed, Python 3.10+):

```bash
python examples/custom_dataset_tutorial.py
ls runs/custom_dataset_tutorial/
```

CI coverage: `tests/test_custom_dataset_tutorial.py`.

## Install

```bash
git clone https://github.com/S-Villar/SURGE.git && cd SURGE
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"
python -c "import surge; print(surge.__version__)"
```

PyPI: `pip install "surge-ml[torch]"` (import name: `surge`). Example scripts
require a GitHub clone.

## 1. Prepare tabular data

One row per sample; columns = inputs and targets.

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Pickle | `.pkl` |
| Parquet | `.parquet` |
| HDF5 | `.h5` (2D numeric table) |

```text
input_a,input_b,input_c,output_x,output_y
1.2,0.5,-0.3,0.61,-0.12
```

Remove NaNs before training. Nested simulation HDF5 (M3DC1 batches, XGC dirs)
uses specialized loaders — see [`SURGE_OVERVIEW.md`](SURGE_OVERVIEW.md).

Generate the bundled tutorial dataset:

```bash
python examples/custom_dataset_tutorial.py --prepare-only
python examples/custom_dataset_tutorial.py --format pkl --prepare-only
```

## 2. Declare inputs and outputs

SURGE auto-detects columns when names follow conventions (`input_*`, `x_*`,
`target_*`, `y_*`, `output_*`). For generic names, declare I/O explicitly.

**Prefer `metadata_path`** when the file has many extra columns — it
**replaces** auto-detection. **`metadata_overrides`** only **adds hints** on
top of auto-detection and can pull in unwanted columns on wide tables.

`examples/configs/custom_dataset_meta.yaml`:

```yaml
inputs:
  - input_a
  - input_b
  - input_c
outputs:
  - output_x
  - output_y
```

Inline alternative:

```yaml
metadata_overrides:
  inputs: [input_a, input_b, input_c]
  outputs: [output_x, output_y]
```

## 3. Workflow spec (YAML)

Copy `examples/configs/custom_dataset_tutorial.yaml` and edit paths, `run_tag`,
and `models`:

```yaml
dataset_path: examples/data/tutorial_sample.csv
dataset_format: auto
metadata_path: examples/configs/custom_dataset_meta.yaml

test_fraction: 0.2
val_fraction: 0.1
standardize_inputs: true
seed: 42
run_tag: my_surrogate
overwrite_existing_run: true

models:
  - key: sklearn.random_forest
    name: rf_baseline
    params: {n_estimators: 200, n_jobs: -1}
  - key: pytorch.residual_mlp
    name: residual_mlp
    params:
      hidden_layers: [128, 64]
      n_epochs: 100
```

List **multiple models** under `models:` — SURGE trains all of them in one run
(same splits and scalers).

### HPO on your data

Add an `hpo` block per model (each gets its own Optuna study):

```yaml
models:
  - key: sklearn.random_forest
    name: rf_hpo
    params: {n_jobs: -1}
    hpo:
      enabled: true
      n_trials: 15
      direction: maximize
      metric: val_r2
      search_space:
        n_estimators: {type: int, low: 50, high: 400}
        max_depth: {type: categorical, choices: [null, 8, 12, 16]}
        min_samples_leaf: {type: int, low: 1, high: 8}

  - key: pytorch.residual_mlp
    name: mlp_hpo
    params: {n_epochs: 50}
    hpo:
      enabled: true
      n_trials: 15
      direction: maximize
      metric: val_r2
      search_space:
        hidden_layers:
          type: categorical
          choices: [[128, 128], [256, 128]]
        learning_rate: {type: loguniform, low: 1e-4, high: 1e-2}
        dropout_rate: {type: float, low: 0.0, high: 0.4}
```

Scientific reference run (QLKNN plasma transport, RF + Residual MLP + HPO):

```bash
pip install fusion_surrogates   # Python ≥ 3.10; generates cache on first run
python examples/qlknn_multi_hpo_workflow.py --hpo-trials 10 --overwrite
```

Spec: `examples/configs/qlknn_multi_hpo.yaml`

## 4. Run

```bash
python examples/custom_dataset_tutorial.py --prepare-only
python examples/run_workflow.py --spec examples/configs/custom_dataset_tutorial.yaml
```

Python API:

```python
import yaml
from pathlib import Path
import surge
from surge import SurrogateWorkflowSpec, run_surrogate_workflow

spec = SurrogateWorkflowSpec.from_dict(
    yaml.safe_load(Path("examples/configs/custom_dataset_tutorial.yaml").read_text())
)
summary = run_surrogate_workflow(spec)
print(summary["artifacts"]["root"])
```

## 5. Inspect artifacts

```text
runs/<run_tag>/
├── metrics.json
├── workflow_summary.json
├── spec.yaml
├── scalers/inputs.joblib
├── models/<name>.joblib
├── predictions/
└── hpo/                    # when HPO enabled
```

```bash
python -c "import json; print(json.dumps(json.load(open('runs/my_surrogate/metrics.json')), indent=2))"
```

### Inference on new rows

```python
import json, joblib, pandas as pd

run_dir = "runs/my_surrogate"
cols = json.load(open(f"{run_dir}/train_data_ranges.json"))["inputs"]["columns"]
scaler = joblib.load(f"{run_dir}/scalers/inputs.joblib")
model = joblib.load(f"{run_dir}/models/rf_baseline.joblib")
df = pd.read_csv("my_data.csv")[cols].head(5)
print(model.predict(scaler.transform(df.values)))
```

PyTorch adapters: load via the adapter class (`model.load(path)`) — see
`surge/model/adapters/residual_mlp.py`.

## 6. Checklist

1. Put data at `data/my_dataset.csv` (or `.pkl` / `.h5`)
2. Write `examples/configs/my_meta.yaml` with `inputs` / `outputs`
3. Copy `examples/configs/custom_dataset_tutorial.yaml` → `examples/configs/my_surrogate.yaml`
4. `python examples/run_workflow.py --spec examples/configs/my_surrogate.yaml`
5. Open `runs/<run_tag>/metrics.json`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Wrong columns as I/O | Use `metadata_path`, not only `metadata_overrides` |
| `ModuleNotFoundError: torch` | `pip install -e ".[torch]"` |
| Run dir exists | `overwrite_existing_run: true` or new `run_tag` |
| Used `surge run -b my.csv` | Use workflow spec instead |

---

# Part II — Embed a new model in SURGE

Use this when **existing adapters** (`sklearn.random_forest`, `pytorch.mlp`,
`pytorch.residual_mlp`, …) are not enough and you need a **new architecture or
backend**.

## Policy overview

Every SURGE model is a **thin adapter** around your training code. The workflow
engine owns splits, scaling, metrics, HPO orchestration, and artifacts — your
adapter owns **fit** and **predict** on numpy arrays the engine provides.

```
Your data  →  SurrogateDataset  →  Engine (split + scale)
                                        ↓
                              BaseModelAdapter.fit / predict
                                        ↓
                              runs/<tag>/ (metrics, models, HPO, …)
```

### Decision checklist — what to implement

| Feature | Required? | When you need it |
|---------|-----------|------------------|
| **`fit(X, y)`** | **Yes** | Always |
| **`predict(X)`** | **Yes** | Always; return `(n_samples, n_outputs)` |
| **`mark_fitted()`** | **Yes** | End of `fit` if you track state manually |
| **`ensure_fitted()`** | **Yes** | Start of `predict` if you track state manually |
| **`save` / `load`** | **Strongly recommended** | Workflow writes `models/<name>.joblib`; without this, runs are not reloadable |
| **Multi-output `y`** | **Yes** (shape) | Support 1D and 2D targets even if you only use scalar outputs today |
| **`resource_profile`** | Recommended | CPU/GPU policy, worker semantics — drives `[surge.fit]` banner |
| **`uses_internal_preprocessing`** | If true | You normalize inside the backend (common for PyTorch tabular) |
| **`handles_output_scaling`** | If true | You inverse-transform targets inside `predict` |
| **`predict_with_uncertainty`** | Optional | Set `supports_uq`; workflow can store UQ JSON |
| **`training_history`** | Optional | Epoch losses for `surge.viz` training plots |
| **HPO `search_space`** | Optional | Expose tunable params in YAML; document sensible ranges |
| **`ModelInfo` (`_INFO`)** | Recommended | Architecture, use cases, strengths — shown in docs/tooling |
| **Optional import guard** | Recommended | `try: import torch` pattern so base install still works |
| **Unit test** | Recommended | At least fit → predict on synthetic `(N, F)` data |
| **Benchmark registration** | Optional | Add to `surge/benchmarks/registry.py` for `surge run` leaderboards |

### Naming and registration

- **Registry key:** `backend.model_name` (e.g. `pytorch.residual_mlp`)
- **File layout:** `surge/model/adapters/<name>.py` + optional `surge/model/backends/<name>.py`
- **Register** in `surge/model/__init__.py`:

```python
from .adapters.my_adapter import MyAdapter
register_model(MyAdapter, key="mybackend.mymodel", aliases=["mymodel"])
```

- **Use in YAML** immediately after registration:

```yaml
models:
  - key: mybackend.mymodel
    params: {hidden_dim: 128}
```

### Data contract

- **`X`, `y` passed to `fit`** are already **standardized** by the engine (unless
  your adapter sets `uses_internal_preprocessing = True`).
- **`predict(X)`** receives standardized inputs; return predictions in the
  **same units as `y` passed to fit** (typically original target scale if the
  engine applied output scaling, or standardized if not).
- Read **`self._last_fit_resources`** after `prepare_for_fit` for `n_jobs`,
  device hints, etc.

### HPO contract

Workflow HPO calls your adapter repeatedly with sampled params. Keep per-trial
training **fast enough for the trial budget** (lower `n_epochs` during search).

Supported search space types in YAML: `int`, `float`, `loguniform`, `categorical`,
`variable_list`.

Example:

```yaml
hpo:
  enabled: true
  n_trials: 20
  direction: maximize
  metric: val_r2
  search_space:
    learning_rate: {type: loguniform, low: 1e-4, high: 1e-2}
```

### PyTorch-specific hooks

- Set **`resource_profile`** with `supports_gpu=True` when CUDA helps.
- Respect env vars **`SURGE_TRAINING_PROGRESS_JSONL`** and
  **`SURGE_CHECKPOINT_DIR`** if you want live loss curves and checkpoints
  (see `surge/model/backends/residual_mlp.py`).
- Implement **`save` / `load`** via `torch.save` or your backend helper.

## Minimal adapter

```python
# surge/model/adapters/my_adapter.py
from __future__ import annotations

import numpy as np

from ..base import BaseModelAdapter
from ...hpc import ResourceProfile

class MyAdapter(BaseModelAdapter):
    name = "mybackend.mymodel"
    backend = "mybackend"
    resource_profile = ResourceProfile(
        name="mybackend.mymodel",
        supports_cpu=True,
        supports_gpu=False,
        worker_semantics="threads",  # or "processes" / "none"
    )
    default_params = {"alpha": 1.0}

    def _build_model(self, **kwargs):
        from my_package import MyRegressor
        params = {**self.default_params, **kwargs}
        return MyRegressor(**params)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return np.asarray(self._model.predict(np.asarray(X)))

    def save(self, path):
        self._model.save(str(path))

    def load(self, path):
        self._model.load(str(path))
        return self
```

Register in `surge/model/__init__.py`:

```python
register_model(MyAdapter, key="mybackend.mymodel", aliases=["mymodel"])
```

## Reference implementations

| Pattern | File |
|---------|------|
| sklearn wrapper | `surge/model/sklearn.py` |
| PyTorch tabular | `surge/model/adapters/residual_mlp.py` |
| Backend + adapter split | `surge/model/backends/residual_mlp.py` |
| Template | `examples/custom_cnn_adapter_template.py` |

## Further reading

- [`QUICK_START_NEW_MODEL.md`](QUICK_START_NEW_MODEL.md) — minimal adapter cheat sheet
- [`ADDING_NEW_MODEL_ADAPTER.md`](ADDING_NEW_MODEL_ADAPTER.md) — full CNN walkthrough
- [`MODEL_EXTENSION_SUMMARY.md`](MODEL_EXTENSION_SUMMARY.md) — architecture diagram

## Related

- [`quickstart.rst`](quickstart.rst) — diabetes / California demos
- [`SURGE_OVERVIEW.md`](SURGE_OVERVIEW.md) — codebase tour
