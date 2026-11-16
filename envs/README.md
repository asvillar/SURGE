# Environments (legacy conda files)

**The supported environment story is: one environment + extras**, managed
with uv (or pip) from `pyproject.toml` — see
[`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[torch,dev]"        # add extras as needed:
uv pip install -e ".[gpflow]"           # TensorFlow/GPflow models
uv pip install -e ".[onnx]"             # ONNX export/runtime
uv pip install -e ".[shap]"             # SHAP feature importance
```

Supported Python: **3.10 – 3.12**.

The conda YAMLs in this directory (`environment.yml`,
`environment_minimal.yml`, `environment_gpu.yml`)
predate that consolidation and are kept for HPC sites that require
conda. If you use them, still install SURGE itself with
`pip install -e ".[...]"` inside the activated environment so dependency
versions come from `pyproject.toml`, and verify with:

```bash
surge version && surge models --verbose
```

A separate TensorFlow-pinned environment is only necessary on hardware
where torch and TensorFlow builds conflict; on typical Linux/macOS,
`.[torch,gpflow]` coexist in one environment (TF ≥ 2.21 is NumPy-2
compatible — see the arm64 notes in `requirements.txt`).

On shared systems with a site-installed environment, activate by path
(`conda activate "$SURGE_CONDA_ENV"`), then run
`python -m examples.quickstart --dataset diabetes` to sanity-check.
