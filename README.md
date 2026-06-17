# <img src="data/logos/surge_logo_os.png" width="56" alt="" align="absmiddle"/>&nbsp;&nbsp;SURGE

[![CI](https://github.com/S-Villar/SURGE/actions/workflows/ci.yml/badge.svg)](https://github.com/S-Villar/SURGE/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](./pyproject.toml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOE CODE](https://img.shields.io/badge/DOE%20CODE-179819-1e4d2e?labelColor=0d2818)](https://www.osti.gov/doecode/biblio/179819)
[![DOI](https://img.shields.io/badge/DOI-10.11578%2Fdc.20260422.5-00758f?labelColor=004466)](https://doi.org/10.11578/dc.20260422.5)

**Surrogate Unified Robust Generation Engine** — train, tune, evaluate, and
export scientific surrogates from a single workflow. One YAML spec (or Python
API) covers load → split → train → HPO → metrics → artifacts under
`runs/<tag>/`.

**v0.1.0** · [DOE CODE 179819](https://www.osti.gov/doecode/biblio/179819) ·
[DOI 10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5)

---

## Install

**Requirements:** Python **3.10 or 3.11** (recommended: 3.11).

PyPI package name: **`surge-ml`** · import name: **`surge`**

### From GitHub (recommended for examples)

Run all commands below from the repo root after cloning:

```bash
git clone https://github.com/S-Villar/SURGE.git
cd SURGE
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"
python -c "import surge; print('surge', surge.__version__)"
```

### From PyPI

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "surge-ml[torch,dev]"
```

> Examples such as `python -m examples.quickstart` and
> `examples/run_workflow.py` live in the GitHub repo — clone it for the
> commands in the next section.

### Optional extras

| Extra | Adds | Install |
|-------|------|---------|
| *(base)* | sklearn, pandas, Optuna, plotting | `pip install -e .` |
| `torch` | PyTorch MLP / Residual MLP | `pip install -e ".[torch]"` |
| `onnx` | ONNX export + runtime smoke tests | `pip install -e ".[onnx]"` |
| `dev` | pytest, ruff, h5py | `pip install -e ".[dev]"` |
| `docs` | Sphinx | `pip install -e ".[docs]"` |

Typical dev install: `pip install -e ".[torch,dev]"`

---

## Try it — copy-paste examples

All runs write artifacts to `runs/<tag>/` (`metrics.json`, trained models,
scalers, predictions, spec snapshot).

### 1 · Smoke test (~5 s) — tabular regression

```bash
python -m examples.quickstart --dataset diabetes --model rf --infer
python -c "import json; print(json.load(open('runs/diabetes_rf/metrics.json'))['sklearn.random_forest']['test'])"
```

### 2 · Neural net + optional HPO (~1–2 min CPU)

```bash
python -m examples.quickstart --dataset california --model mlp --n-trials 5 --infer
ls runs/california_mlp_hpo5/
```

### 3 · Scientific case — QLKNN plasma transport, two models, HPO

Predict **electron ITG heat flux** (`efeITG`) from 10 gyrokinetic inputs.
One workflow trains **Random Forest + Residual MLP**, each with its own
Optuna search. First run generates the dataset via `fusion_surrogates`
(**Python ≥ 3.10**).

```bash
pip install fusion_surrogates
python examples/qlknn_multi_hpo_workflow.py --hpo-trials 10 --overwrite
python -c "
import json
m = json.load(open('runs/qlknn_multi_hpo/metrics.json'))
for name in ('qlknn_rf', 'qlknn_residual_mlp'):
    print(name, m[name]['test'])
"
```

Cache: `data/datasets/benchmarks/plasma/qlknn_transport.npz` ·
YAML spec: `examples/configs/qlknn_multi_hpo.yaml`

### 4 · Your own CSV / PKL / H5 file

```bash
python examples/custom_dataset_tutorial.py
python examples/run_workflow.py --spec examples/configs/custom_dataset_tutorial.yaml
```

Full guide: [`docs/BUILD_YOUR_OWN_SURROGATE.md`](docs/BUILD_YOUR_OWN_SURROGATE.md)

### 5 · Benchmark suite (built-in datasets)

```bash
surge list                                          # available benchmarks
surge run -b iris -m sklearn.random_forest          # single benchmark + model
surge run -b qlknn -m pytorch.residual_mlp          # plasma transport
```

`surge run` uses curated benchmarks only — not arbitrary local CSV files.
For custom data, use section **4** or the workflow YAML path.

---

## Verify the install

```bash
pytest -q tests/test_e2e_release_smoke.py          # fast smoke (~seconds)
pytest -q                                          # full suite (dev extra)
```

---

## What you get after a run

```text
runs/<tag>/
├── metrics.json              # train / val / test metrics per model
├── workflow_summary.json     # full run summary + profiling
├── spec.yaml                 # exact config (re-runnable)
├── scalers/inputs.joblib
├── models/<name>.joblib
├── predictions/              # parquet per split
├── plots/                    # parity + HPO + training dashboards (see below)
└── hpo/                      # when HPO is enabled (*_hpo.json trial logs)
```

---

## Visualize results

After any workflow run, generate **regression parity plots** (predicted vs
ground truth, R² per split) and **HPO convergence** curves from the saved
artifacts. Requires matplotlib (included in the base install).

### Quick path — add `--viz` to quickstart

```bash
python -m examples.quickstart --dataset california --model mlp --n-trials 5 --viz
# writes runs/california_mlp_hpo5/plots/
```

### Any run directory — `surge.viz.viz_run`

Works for quickstart runs, QLKNN multi-HPO, custom YAML workflows, etc.:

```bash
python examples/qlknn_multi_hpo_workflow.py --hpo-trials 10 --overwrite

python -c "
from pathlib import Path
from surge.viz import viz_run
result = viz_run(Path('runs/qlknn_multi_hpo'), dpi=150, include_hpo=True)
for p in result['saved_paths']:
    print(p)
"
```

### What gets written to `runs/<tag>/plots/`

| Plot | Source | When |
|------|--------|------|
| `inference_comparison_output_*.png` | `predictions/*_{train,val,test}.parquet` | Every run with predictions |
| `inference_comparison_grid.png` | Same (combined layout) | Every run with predictions |
| `hpo_convergence.png` | `hpo/*_hpo.json` | Runs with Optuna HPO |
| `training_dashboard_*.png` | `training_history_*.json` | PyTorch / Residual MLP (auto during train) |

**Regression maps:** each panel is a 2D density scatter of ground truth vs
prediction with a diagonal reference and R² for train / val / test. Multi-model
runs (e.g. QLKNN RF + Residual MLP) show one row per model.

**HPO:** `hpo_convergence.png` tracks the best validation metric across trials
(R² when available, otherwise RMSE). Raw trial data stays in `hpo/`.

**Neural training curves:** PyTorch backends also write
`training_log_<model>.jsonl` and `training_progress_*.jsonl` under the run
root. Re-plot or compare epochs with:

```bash
python -c "
from pathlib import Path
from surge.viz.training import load_training_history, plot_training_dashboard
hist = load_training_history(Path('runs/california_mlp_hpo5/training_log_pytorch.mlp.jsonl'))
plot_training_dashboard(hist, model_name='pytorch.mlp', save_path='training_dashboard.png')
print('saved training_dashboard.png')
"
```

---

## Documentation

| Topic | Link |
|-------|------|
| Build your own surrogate | [`docs/BUILD_YOUR_OWN_SURROGATE.md`](docs/BUILD_YOUR_OWN_SURROGATE.md) |
| First-run walkthrough (HPC) | [`docs/setup/WALKTHROUGH.md`](docs/setup/WALKTHROUGH.md) |
| Install reference | [`docs/setup/INSTALLATION.md`](docs/setup/INSTALLATION.md) |
| Codebase tour | [`docs/SURGE_OVERVIEW.md`](docs/SURGE_OVERVIEW.md) |
| Doc index | [`docs/README.md`](docs/README.md) |

---

## Community

- **Issues:** `.github/ISSUE_TEMPLATE/`
- **Security:** [`SECURITY.md`](SECURITY.md) (private channel — not public issues)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Cite:** [DOE CODE 179819](https://www.osti.gov/doecode/biblio/179819),
  DOI [10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5),
  [`CITATION.cff`](CITATION.cff)

<p align="center">
  <img src="data/logos/surge_logo_os_expanded.png" alt="SURGE — Surrogate Unified Robust Generation Engine" width="640"/>
</p>

## License

BSD 3-Clause — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
