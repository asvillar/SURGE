# <img src="data/logos/logoBlueSolid.png" width="56" alt="" align="absmiddle"/>&nbsp;&nbsp;SURGE

[![CI](https://github.com/S-Villar/SURGE/actions/workflows/ci.yml/badge.svg)](https://github.com/S-Villar/SURGE/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](./pyproject.toml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/uv-ready-261230.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)

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

Test suite (on a clean clone): **`pytest -q tests/`** → *51 passed,
38 skipped, 0 failed* on Python 3.11/3.13. Skipped tests are legacy
pre-refactor tests tracked in
[`docs/REFACTORING_PLAN.md`](docs/REFACTORING_PLAN.md) §1.9.

## Install

From a clone:

```bash
git clone https://github.com/S-Villar/SURGE.git
cd SURGE
python -m pip install -e ".[torch,onnx]"   # core + PyTorch + ONNX export
```

The extras are additive and optional:

| Extra   | Adds                                            | Use when…                                    |
|---------|-------------------------------------------------|----------------------------------------------|
| (none)  | sklearn / pandas / parquet / MLflow / plotting  | you only want classical regressors           |
| `torch` | PyTorch 1.9+ (MLP adapter, MC-Dropout)          | you want neural surrogates                   |
| `onnx`  | `onnx`, `onnxscript`, `onnxruntime`             | you want to export + deploy trained models   |
| `dev`   | `pytest`, `pytest-cov`, `ruff`                  | you plan to run the test suite / lint        |
| `docs`  | `sphinx`, `furo`, `myst-parser`                 | you plan to build the docs locally           |

With [`uv`](https://github.com/astral-sh/uv) (faster, isolated):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[torch,onnx]"
```

Or from a published tag (once `0.1.0` is on PyPI):

```bash
python -m pip install "surge-ml[torch,onnx]==0.1.0"
```

`surge-ml` is the PyPI distribution name; the import name is `surge`.
A clean `pip install .` (no extras) pulls ~25 packages and completes
in under 5 seconds on a typical laptop.

## Smoke test

```bash
python -c "import surge; print(surge.__version__)"
pytest -q tests/test_e2e_release_smoke.py
```

The smoke test exercises: CSV → `SurrogateEngine` → fit → predict → ONNX
export → `onnxruntime` round-trip parity. It is also the CI
`e2e-regression` job.

## Quickstart

A complete train → evaluate → profile cycle, end to end, on a public
442-row dataset. Runs in under 5 seconds on a laptop.

```python
import pandas as pd
from sklearn.datasets import load_diabetes

from surge import SurrogateWorkflowSpec, run_surrogate_workflow
from surge.hpc import ResourceSpec

# 1. Any CSV with named columns works — here: sklearn's diabetes dataset
#    (10 inputs, 1 output, 442 rows).
load_diabetes(as_frame=True).frame.to_csv("diabetes.csv", index=False)

# 2. Declare the workflow: dataset, model(s), compute budget, output dir.
spec = SurrogateWorkflowSpec(
    dataset_path="diabetes.csv",
    models=[{"key": "sklearn.random_forest",
             "params": {"n_estimators": 200}}],
    resources=ResourceSpec(device="cpu", num_workers=4),
    output_dir=".",
    run_tag="quickstart",
    overwrite_existing_run=True,
)

# 3. Run it.
summary = run_surrogate_workflow(spec)

m = summary["models"][0]
print(f"test R²   = {m['metrics']['test']['r2']:.3f}")
print(f"test RMSE = {m['metrics']['test']['rmse']:.2f}")
print(f"model     = {m['profile']['model_size_bytes']/1024:.1f} KB, "
      f"{m['profile']['parameter_count']:,} params")
print(f"inference = {m['profile']['inference_ms_per_sample']:.2f} ms/sample")
```

What you actually see when you run this (verbatim from the command
above, captured on CPU with 4 sklearn workers):

```text
SURGE workflow started.
[1/3] Loading dataset...
[2/3] Preparing splits and scalers...
[3/3] Training sklearn.random_forest...
[3/3] Done. Artifacts in runs/quickstart

test R²   = 0.444
test RMSE = 54.26
model     = 5352.0 KB, 75,470 params
inference = 0.75 ms/sample
```

The `runs/quickstart/` directory now contains everything you need to
reproduce or audit the run:

```text
runs/quickstart/
├── spec.yaml                                     # exact workflow spec (for re-running)
├── env.txt                                       # pip freeze at run time
├── git_rev.txt                                   # HEAD of the repo, or "unknown"
├── run.log                                       # stdout capture
├── workflow_summary.json                         # metrics + profile + resources_used
├── metrics.json                                  # per-model train/val/test + timings
├── train_data_ranges.json                        # input ranges (for OOD flagging)
├── model_card_sklearn.random_forest.json         # data + model provenance card
├── scalers/
│   └── inputs.joblib                             # input scaler (needed for inference)
├── models/
│   └── sklearn.random_forest.joblib              # the trained estimator
├── predictions/
│   ├── sklearn.random_forest_train.parquet
│   ├── sklearn.random_forest_val.parquet
│   └── sklearn.random_forest_test.parquet
└── hpo/                                          # populated only if HPO is enabled
```

PyTorch adapters additionally drop an ONNX export under
`models/<key>.onnx` when the `torch` + `onnx` extras are installed.

During training SURGE also prints a one-line banner per model so you
always know what backend, device, and worker count actually ran:

```text
[surge.fit] model=sklearn.random_forest backend=sklearn device=cpu \
            max_gpus=1 n_train=309 n_features=10 n_outputs=1 n_jobs=4
```

The same fields are persisted to `workflow_summary.json` under
`models[].resources_used` for auditing. `profile` carries
`model_size_bytes`, `parameter_count`, and `inference_ms_per_sample`
for every run.

## Documentation

- [`docs/README.md`](docs/README.md) — documentation index.
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
- Citing: [`CITATION.cff`](CITATION.cff) — GitHub surfaces this as
  *Cite this repository* once `0.1.0` is tagged.

<p align="center">
  <img src="data/logos/Fulllogo.png" alt="SURGE — Surrogate Unified Robust Generation Engine" width="520"/>
</p>

## License and funding

- **License:** BSD 3-Clause — see [`LICENSE`](LICENSE).
- **Notice / funding:** see [`NOTICE`](NOTICE) for the DOE acknowledgement
  and the customary disclaimer.

After DOE CODE registration, cite SURGE using the identifier and URL your
publications office provides, together with `CITATION.cff`.
