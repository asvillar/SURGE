# <img src="data/logos/surge_logo_os.png" width="56" alt="" align="absmiddle"/>&nbsp;&nbsp;SURGE

[![CI](https://github.com/S-Villar/SURGE/actions/workflows/ci.yml/badge.svg)](https://github.com/S-Villar/SURGE/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](./pyproject.toml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/uv-ready-261230.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![DOE CODE](https://img.shields.io/badge/DOE%20CODE-179819-1e4d2e?labelColor=0d2818)](https://www.osti.gov/doecode/biblio/179819)
[![DOI](https://img.shields.io/badge/DOI-10.11578%2Fdc.20260422.5-00758f?labelColor=004466)](https://doi.org/10.11578/dc.20260422.5)

<!--
  The two badges below (PyPI version and GitHub last-commit) depend on
  data that shields.io can only read when the repo is PUBLIC *and*
  the package is published. Uncomment them at the same time you flip
  the repo visibility to public and run `twine upload` for 0.1.0.

  [![PyPI](https://img.shields.io/pypi/v/surge-ml.svg?label=PyPI&color=informational)](https://pypi.org/project/surge-ml/)
  [![Last commit](https://img.shields.io/github/last-commit/S-Villar/SURGE.svg)](https://github.com/S-Villar/SURGE/commits)
-->


**Surrogate Unified Robust Generation Engine** — a surrogate modeling
framework for scientific workflows that integrates Scientific Machine
Learning (SciML) and AutoML features, uncertainty quantification (UQ),
and MLOps-style provenance in a single declarative pipeline.

SURGE unifies data generation and ingestion, an extensible registry of
model adapters (classical, neural, probabilistic, and ensemble), held-out
evaluation with UQ, automated hyperparameter optimization, structured
artifact and lineage tracking, diagnostic visualization, and portable
inference and deployment. Configuration-as-code specifications parameterize
a composable, end-to-end surrogate development cycle and emit
machine-readable provenance, exposing APIs suitable for agent-based
orchestration.

Concretely, the shipping pieces are:

- A unified **engine** (`SurrogateEngine`) covering scikit-learn, PyTorch, and
  GPflow backends through a single adapter interface.
- A declarative **workflow spec** (`SurrogateWorkflowSpec`) with `train → HPO →
  predict → export` in one YAML file.
- A **resource policy** (`ResourceSpec`) that lets you pick `cpu` vs `cuda` and
  set the per-model worker count, with one `[surge.fit] ...` banner per train
  so you always see what actually ran.
- A lightweight **post-training profile** per model: `model_size_bytes`,
  `parameter_count`, `inference_ms_per_sample`, throughput.
- Per-run **provenance artifacts**: `workflow_summary.json`, `metrics.json`,
  predictions (Parquet), scalers, and a model card.
- Optional **ONNX export** with a round-trip smoke test in CI.

See [`docs/SURGE_OVERVIEW.md`](docs/SURGE_OVERVIEW.md) for a tour of the
codebase and [`docs/RELEASE_DEMO_PLAN.md`](docs/RELEASE_DEMO_PLAN.md) for the
end-to-end user walkthrough the `0.1.0` release must support.

## Status

Pre-`0.1.0`. The core `engine → adapters → workflow → artifacts` path is
stable and green in CI; see
[`docs/PRERELEASE.md`](docs/PRERELEASE.md) for the exact scope.

**DOE record:** [DOE CODE 179819](https://www.osti.gov/doecode/biblio/179819) ·
[DOI 10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5)

Test suite (on a clean clone with `.[torch,onnx,dev]`): **`pytest -q`**
→ *52 passed, 1 skipped, 0 failed* on Python 3.11. The skip is a
legacy visualization scaffold pending migration to `surge.viz`
(`h5py` ships with the `dev` extra so MHD-code batch tests run).

## Install (extras reference)

Optional extras layer on top of the base install:

| Extra   | Adds                                            | Use when…                                    |
|---------|-------------------------------------------------|----------------------------------------------|
| (none)  | sklearn / pandas / parquet / MLflow / plotting  | you only want classical regressors           |
| `torch` | PyTorch 1.9+ (MLP adapter, MC-Dropout)          | you want neural surrogates                   |
| `onnx`  | `onnx`, `onnxscript`, `onnxruntime`             | you want to export + deploy trained models   |
| `dev`   | `pytest`, `pytest-cov`, `ruff`, `h5py`          | tests + lint + HDF5-backed MHD-code tests       |
| `docs`  | `sphinx`, `furo`, `myst-parser`                 | you plan to build the docs locally           |

From a published tag (once `0.1.0` is on PyPI):

```bash
python -m pip install "surge-ml[torch,onnx,dev]==0.1.0"   # dev adds pytest + h5py
```

`surge-ml` is the PyPI distribution name; the import name is `surge`.
A clean `pip install .` (no extras) pulls ~25 packages and completes
in under 5 seconds on a typical laptop. For the full recipe (clone →
venv → verify → tests → run), see the Quickstart immediately below.

## Quickstart

**Flow:** clone → [`uv`](https://github.com/astral-sh/uv) →
`uv pip install -e ".[torch,onnx,dev]"` (the `dev` extra includes
`pytest`, `ruff`, and **`h5py`** for MHD-code / HDF5 tests) → `pytest` →
[`examples/quickstart.py`](examples/quickstart.py). NERSC / `$SCRATCH`
details: [`docs/setup/WALKTHROUGH.md`](docs/setup/WALKTHROUGH.md).

<details>
<summary><b>1 — Environment (first time; skip if already installed)</b></summary>

```bash
export TEST_DIR=${TEST_DIR:-$HOME/surge-release-test}   # or $SCRATCH/...
mkdir -p "$TEST_DIR" && cd "$TEST_DIR"
git clone https://github.com/S-Villar/SURGE.git

# uv (once):  curl -LsSf https://astral.sh/uv/install.sh | sh
#             export PATH="$HOME/.local/bin:$PATH"    # add to ~/.bashrc
# if curl is blocked:  python3.11 -m pip install --user uv
# no uv at all:       python3.11 -m venv .venv  &&  use pip instead of uv pip

uv venv --python 3.11 .venv && source .venv/bin/activate
cd SURGE
uv pip install -e ".[torch,onnx,dev]"

python - <<'PY'
import surge, torch, onnx, onnxruntime, sklearn, h5py
print(f"surge {surge.__version__} | torch {torch.__version__} | h5py {h5py.__version__}")
PY
```

</details>

**2 — Tests** (run from the repo root, venv active):

`pytest -q` → expect **52 passed, 1 skipped** (skip = legacy viz scaffold
in `test_model_comparison.py`). CI smoke: `pytest -q tests/test_e2e_release_smoke.py`.

**3 — Demos** (must run **inside** `SURGE/` so `python -m examples.quickstart` resolves; outputs go to `runs/<tag>/` in the repo):

| Step | Command |
|------|---------|
| Diabetes + RF (~5 s) | `python -m examples.quickstart --dataset diabetes --model rf --viz --infer` |
| California + MLP (~60–90 s CPU) | `python -m examples.quickstart --dataset california --model mlp --viz --infer` |
| Optional HPO (5 trials × 50 epochs) | add `--n-trials 5` to the MLP line |

**4 — Inspect** — e.g. `ls runs/california_mlp/`, or:

```bash
python -c "import json; d=json.load(open('runs/california_mlp/metrics.json')); print(json.dumps(d['pytorch.mlp']['test'], indent=2))"
python -m surge.cli viz --run-dir runs/california_mlp   # parity plots only, no retrain
```

### Run it yourself in Python

The CLI is a thin wrapper over the public API. The same result in a
notebook:

```python
from sklearn.datasets import fetch_california_housing
from surge import SurrogateWorkflowSpec, run_surrogate_workflow
from surge.hpc import ResourceSpec

frame = fetch_california_housing(as_frame=True).frame
frame.to_csv("california.csv", index=False)

spec = SurrogateWorkflowSpec(
    dataset_path="california.csv",
    metadata_overrides={
        "inputs": [c for c in frame.columns if c != "MedHouseVal"],
        "outputs": ["MedHouseVal"],
    },
    models=[{"key": "sklearn.random_forest",
             "params": {"n_estimators": 200}}],
    resources=ResourceSpec(device="cpu", num_workers=4),
    output_dir=".",
    run_tag="california_rf",
    overwrite_existing_run=True,
)

summary = run_surrogate_workflow(spec)
m = summary["models"][0]
print(f"test R² = {m['metrics']['test']['r2']:.3f}")
```

### Round-trip inference from a saved run

Every run writes a self-contained bundle under `runs/<tag>/`. To score
new inputs from another process, load the input scaler and the model,
then apply them in the same order SURGE used:

```python
import json, joblib, numpy as np, pandas as pd

run_dir = "runs/california_rf"
# SURGE sorts input columns alphabetically; read the canonical order
# back from train_data_ranges.json.
input_cols = json.loads(open(f"{run_dir}/train_data_ranges.json").read())["inputs"]["columns"]

scaler = joblib.load(f"{run_dir}/scalers/inputs.joblib")
model  = joblib.load(f"{run_dir}/models/sklearn.random_forest.joblib")

df = pd.read_csv("california.csv")[input_cols].head(5)
y_hat = model.predict(scaler.transform(df.values))
print(np.round(y_hat, 2))
```

For PyTorch models (`pytorch.mlp.joblib` is a `torch.save` archive, not
a joblib pickle) use the adapter class, which re-instates the model's
internal `scaler_y` and inverse-transforms predictions for you:

```python
from surge.model.pytorch_impl import PyTorchMLPModel

model = PyTorchMLPModel()
model.load(f"{run_dir}/models/pytorch.mlp.joblib")
y_hat = model.predict(scaler.transform(df.values))   # returns original units
```

### What ends up on disk

```text
runs/<tag>/
├── spec.yaml                          # exact workflow spec (re-runnable)
├── env.txt                            # pip freeze at run time
├── git_rev.txt                        # HEAD of the repo, or "unknown"
├── run.log                            # stdout capture
├── workflow_summary.json              # metrics + profile + resources_used
├── metrics.json                       # per-model train/val/test + timings
├── train_data_ranges.json             # canonical input order + min/max
├── model_card_<key>.json              # data + model provenance card
├── scalers/inputs.joblib              # input scaler (needed for inference)
├── models/<key>.joblib                # trained estimator / torch archive
├── models/<key>.onnx                  # PyTorch: ONNX export (when extras installed)
├── predictions/<key>_{train,val,test}.parquet
└── hpo/                               # populated only when --n-trials > 0
```

During training SURGE prints a one-line banner per model so you always
know what backend, device, and worker count actually ran:

```text
[surge.fit] model=sklearn.random_forest backend=sklearn device=cpu \
            max_gpus=1 n_train=309 n_features=10 n_outputs=1 n_jobs=4
```

The same fields are persisted to `workflow_summary.json` under
`models[].resources_used`. `profile` carries `model_size_bytes`,
`parameter_count`, and `inference_ms_per_sample` for every run.

## Documentation

- [`docs/README.md`](docs/README.md) — documentation index.
- [`docs/setup/WALKTHROUGH.md`](docs/setup/WALKTHROUGH.md) —
  copy-paste-ready first-run walkthrough on `$SCRATCH` (clone → venv →
  install → tests → CLI quickstart).
- [`docs/setup/INSTALLATION.md`](docs/setup/INSTALLATION.md) — longer
  install reference (extras, troubleshooting, NERSC tips).
- [`docs/PRERELEASE.md`](docs/PRERELEASE.md) — what ships in `0.1.0`.
- [`docs/RELEASE_DEMO_PLAN.md`](docs/RELEASE_DEMO_PLAN.md) — end-to-end tour.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — scheduled post-`0.1.0` work.
- [`docs/REFACTORING_PLAN.md`](docs/REFACTORING_PLAN.md) — code cleanup plan.
- Sphinx build (requires `[docs]` extra): `make -C docs html`.

## Community

- Issues: use the templates under `.github/ISSUE_TEMPLATE/`.
- Security: please use the private channel in [`SECURITY.md`](SECURITY.md),
  **not** a public issue.
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Citing: registered in [DOE CODE](https://www.osti.gov/doecode) as **Code ID
  [179819](https://www.osti.gov/doecode/biblio/179819)**, with DOI
  [10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5). See
  also [`CITATION.cff`](CITATION.cff) (GitHub *Cite this repository* with a
  tagged release).

<p align="center">
  <img src="data/logos/surge_logo_os_expanded.png" alt="SURGE — Surrogate Unified Robust Generation Engine" width="640"/>
</p>

## License and funding

- **License:** BSD 3-Clause — see [`LICENSE`](LICENSE).
- **Notice / funding:** see [`NOTICE`](NOTICE) for the DOE acknowledgement
  and the customary disclaimer.

**DOE / OSTI:** the canonical software record is
[osti.gov/…/doecode/biblio/179819](https://www.osti.gov/doecode/biblio/179819)
(**DOE CODE 179819**; DOI
[10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5)). OSTI also
exposes [MLA, APA, Chicago, and BibTeX](https://www.osti.gov/doecode/biblio/179819)
on that page under *Citation Formats*.

Cite alongside [`CITATION.cff`](CITATION.cff) and the
[`pyproject.toml`](pyproject.toml) / GitHub *About* metadata for version and
URL consistency.
