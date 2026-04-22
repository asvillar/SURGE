# SURGE

**Surrogate Regression Engine** — a surrogate modeling framework for
scientific workflows that integrates Scientific Machine Learning (SciML)
and AutoML features, uncertainty quantification (UQ), and MLOps-style
provenance in a single declarative pipeline.

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
python -m pip install -e ".[torch]"   # core + PyTorch MLP; omit [torch] if not needed
```

Or from a published tag (once `0.1.0` is on PyPI):

```bash
python -m pip install "surge-ml==0.1.0"
```

`surge-ml` is the PyPI distribution name; the import name is `surge`.

## Smoke test

```bash
python -c "import surge; print(surge.__version__)"
pytest -q tests/test_e2e_release_smoke.py
```

The smoke test exercises: CSV → `SurrogateEngine` → fit → predict → ONNX
export → `onnxruntime` round-trip parity. It is also the CI
`e2e-regression` job.

## Minimal example

```python
from surge import (
    EngineRunConfig, ModelSpec, SurrogateEngine,
    SurrogateWorkflowSpec, run_surrogate_workflow,
)
from surge.hpc import ResourceSpec

spec = SurrogateWorkflowSpec(
    dataset_path="data/datasets/MHD-code/mhd_code_synthetic_profiles_sample.csv",
    models=[{"key": "sklearn.random_forest", "params": {"n_estimators": 100}}],
    resources=ResourceSpec(device="cpu", num_workers=4),
    output_dir="runs/",
)
summary = run_surrogate_workflow(spec)
print(summary["models"][0]["profile"])
# {'model_size_bytes': ..., 'parameter_count': ...,
#  'inference_ms_per_sample': ..., ...}
```

During training you will see exactly one banner per model:

```
[surge.fit] model=sklearn.random_forest backend=sklearn device=cpu
            max_gpus=1 n_train=... n_features=... n_outputs=... n_jobs=4
```

The same fields are persisted to `workflow_summary.json` under
`models[].resources_used` for auditing.

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

## License and funding

- **License:** BSD 3-Clause — see [`LICENSE`](LICENSE).
- **Notice / funding:** see [`NOTICE`](NOTICE) for the DOE acknowledgement
  and the customary disclaimer.

After DOE CODE registration, cite SURGE using the identifier and URL your
publications office provides, together with `CITATION.cff`.
