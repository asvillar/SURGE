Quick Start Guide
=================

This guide walks through the SURGE workflow end-to-end: dataset ingestion →
splitting & scaling → training → (optional) HPO → artifact export. It mirrors
the README but with a bit more narrative and a Sphinx-friendly layout.

Prerequisites
-------------

* Python 3.10 or 3.11
* SURGE installed in editable mode (see :doc:`setup/INSTALLATION`)::

    pip install -e ".[torch,onnx]"

* No dataset required — the bundled CLI can materialize scikit-learn's
  ``load_diabetes`` or ``fetch_california_housing`` on the fly.

One-minute quickstart (CLI)
---------------------------

.. code-block:: bash

   # 442-row regression benchmark, finishes in ~5 s on a laptop
   python -m examples.quickstart --dataset diabetes

   # 20,640-row housing benchmark, finishes in ~30 s
   python -m examples.quickstart --dataset california

   # PyTorch MLP instead of random forest
   python -m examples.quickstart --dataset california --model mlp

Everything lands under ``runs/<tag>/`` with a stable layout:

* ``models/`` — trained estimator(s), serialized with joblib
* ``scalers/`` — ``StandardScaler`` bundle(s) for inputs / outputs
* ``predictions/`` — per-split parquet (train / val / test) plus optional UQ JSON
* ``metrics.json`` / ``workflow_summary.json`` — metrics, timings, profiling
* ``model_card_<key>.json`` — provenance card for downstream citation
* ``spec.yaml`` / ``env.txt`` / ``git_rev.txt`` / ``run.log`` — reproducibility snapshot
* ``hpo/<model>_hpo.json`` — serialized Optuna trials (when HPO is enabled)

Programmatic use
----------------

The CLI is a thin wrapper around the workflow API. You can drive it directly:

.. code-block:: python

   from sklearn.datasets import load_diabetes
   from surge import SurrogateWorkflowSpec, run_surrogate_workflow
   from surge.hpc import ResourceSpec

   load_diabetes(as_frame=True).frame.to_csv("diabetes.csv", index=False)

   spec = SurrogateWorkflowSpec(
       dataset_path="diabetes.csv",
       models=[{"key": "sklearn.random_forest",
                "params": {"n_estimators": 200}}],
       resources=ResourceSpec(device="cpu", num_workers=4),
       output_dir=".",
       run_tag="quickstart",
       overwrite_existing_run=True,
   )
   summary = run_surrogate_workflow(spec)
   print(summary["models"][0]["metrics"]["test"])

Each entry in ``summary["models"]`` carries ``metrics``, ``timings``,
``profile`` (model size + inference latency), ``resources_used`` (device /
worker count actually used), and artifact paths.

Loading a trained surrogate for inference
-----------------------------------------

Once a run exists, you can round-trip the trained artifact without touching
the ``SurrogateWorkflowSpec`` API. Two things to know:

1. SURGE sorts input columns alphabetically during preprocessing. The
   canonical order for a given run is written to
   ``runs/<tag>/train_data_ranges.json`` under ``inputs.columns`` — read
   it back rather than trusting the CSV header order.
2. The scaler at ``runs/<tag>/scalers/inputs.joblib`` must be applied
   *before* the model's ``.predict`` call.

sklearn backends (``sklearn.random_forest`` and friends) are plain joblib
pickles:

.. code-block:: python

   import json, joblib
   import pandas as pd
   import numpy as np

   run_dir = "runs/quickstart"
   input_cols = json.loads(open(f"{run_dir}/train_data_ranges.json").read())["inputs"]["columns"]

   model  = joblib.load(f"{run_dir}/models/sklearn.random_forest.joblib")
   scaler = joblib.load(f"{run_dir}/scalers/inputs.joblib")

   df = pd.read_csv("diabetes.csv")[input_cols].head(5)
   y_hat = model.predict(scaler.transform(df.values))
   print("first 5 predictions:", np.round(y_hat, 1))

PyTorch backends (``pytorch.mlp``) save a ``torch.save`` archive under
the same ``.joblib`` name. Use the adapter class, which restores the
model's internal ``scaler_y`` and inverse-transforms predictions back to
the original target units:

.. code-block:: python

   from surge.model.pytorch_impl import PyTorchMLPModel

   model = PyTorchMLPModel()
   model.load(f"{run_dir}/models/pytorch.mlp.joblib")
   y_hat = model.predict(scaler.transform(df.values))   # original units

For PyTorch-backed models trained with the ``onnx`` extra installed, the
same run directory also contains ``models/<key>.onnx`` that
``onnxruntime`` can load without a Python dependency on SURGE or
PyTorch.

Both round-trips are implemented end-to-end in
``examples/quickstart.py`` under ``--infer``; read the
``_demo_inference`` helper there for a working reference.

Configuration-driven runs (YAML)
--------------------------------

For anything beyond a quick experiment you'll typically author a YAML spec
and invoke ``run_surrogate_workflow`` with it. The schema matches
``SurrogateWorkflowSpec`` one-for-one; see ``configs/`` for worked examples.

.. code-block:: python

   from pathlib import Path
   import yaml
   from surge.workflow.spec import SurrogateWorkflowSpec
   from surge.workflow.run import run_surrogate_workflow

   spec = SurrogateWorkflowSpec.from_dict(
       yaml.safe_load(Path("configs/my_experiment.yaml").read_text())
   )
   summary = run_surrogate_workflow(spec)

Next steps
----------

* :doc:`api_reference/index` — class and method reference
* :doc:`comparison` — how SURGE compares to neighbouring tools
