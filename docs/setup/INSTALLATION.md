# Installation

SURGE is published as the `surge-ml` distribution on PyPI; the import name is
`surge`. The README covers the 30-second install; this page is the longer
reference for people setting up a fresh environment on a workstation or HPC
login node.

## 1. Pick an environment

SURGE targets **Python 3.10 or 3.11**. CI tests both. You can use any
environment manager you like; the two we test routinely are [`uv`](https://github.com/astral-sh/uv)
and `venv`+`pip`.

```bash
# Option A — uv (fastest, isolated; what the CI does)
uv venv --python 3.11 .venv
source .venv/bin/activate

# Option B — plain venv
python3.11 -m venv .venv
source .venv/bin/activate
```

## 2. Install SURGE

From a clone (recommended while `0.1.0` is still pre-release on PyPI):

```bash
git clone https://github.com/S-Villar/SURGE.git
cd SURGE
pip install -e ".[torch,onnx]"
```

Or, once `0.1.0` is tagged on PyPI:

```bash
pip install "surge-ml[torch,onnx]==0.1.0"
```

### Optional extras

SURGE keeps heavy dependencies opt-in. The extras are additive.

| Extra   | Pulls in                                         | Install when you want to…                   |
|---------|--------------------------------------------------|---------------------------------------------|
| (none)  | sklearn / pandas / pyarrow / optuna / matplotlib | train classical regressors (RF, MLP-sklearn)|
| `torch` | `torch>=1.9`                                     | train the PyTorch MLP surrogate             |
| `onnx`  | `onnx`, `onnxscript`, `onnxruntime`              | export trained models for deployment        |
| `dev`   | `pytest`, `pytest-cov`, `ruff`                   | run the test suite + lint locally           |
| `docs`  | `sphinx`, `furo`, `myst-parser`                  | build the documentation                     |

The full developer install used by CI is:

```bash
pip install -e ".[dev,torch,onnx]"
```

## 3. Verify the install

The bundled checker prints a one-screen diagnostic:

```bash
python scripts/utils/check_installation.py
```

Or quickly by hand:

```bash
python -c "import surge; print('surge', surge.__version__)"
python -c "from surge import SurrogateWorkflowSpec, run_surrogate_workflow; print('workflow API OK')"
```

Then run the quickstart (uses scikit-learn's 442-row diabetes dataset, under
5 seconds on a laptop):

```bash
python -m examples.quickstart --dataset diabetes
```

## 4. Run the tests

```bash
pytest -q
```

`pytest` is scoped to the `tests/` directory via `pyproject.toml`. CI runs the
same command plus a dedicated end-to-end smoke test that trains an sklearn
model, trains a tiny PyTorch MLP, exports to ONNX, and verifies round-trip
parity.

## Troubleshooting

**`ModuleNotFoundError: No module named 'surge'`** — you forgot `pip install
-e .` (or installed it into a different environment). Double-check with
`which python` and `which pip`.

**`torch.onnx.export` silently produces no file** — `torch>=2.11` dispatches
ONNX export through `onnxscript`, which lives in the `onnx` extra. Install
with `.[torch,onnx]`.

**`Warning: GPflow not available or incompatible`** — cosmetic. GPflow is an
optional backend that requires a legacy TensorFlow; the warning is emitted
once at import and the sklearn / torch / onnx paths are unaffected. If you
do need GPflow, uncomment its extra in `pyproject.toml` and install.

**Running on NERSC / Perlmutter** — create the venv on `$SCRATCH` rather than
`$HOME` (quota pressure); see `docs/internal/SURGE_SLURM_BEST_PRACTICES.md` for
worked examples if you have access.

## Next steps

- [`docs/setup/WALKTHROUGH.md`](WALKTHROUGH.md) — copy-paste-ready
  first-run flow (clone → venv → install → tests → CLI quickstart)
  against `$SCRATCH` or any other clean directory.
- `python -m examples.quickstart --dataset california` — a bigger public
  regression benchmark (20k samples, typically R² > 0.8).
- [`docs/RUNBOOK.md`](../RUNBOOK.md) — what each file under
  `runs/<tag>/` means and how to consume it downstream.
- The README's *Quickstart* section for a verbatim output transcript and a
  full artifact listing.
