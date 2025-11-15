Quick Start Guide
=================

This guide covers the unified SURGE workflow (dataset ingestion → splitting → training → HPO → artifact export). Legacy helpers such as ``SurrogateTrainer`` remain available but are no longer highlighted in Quick Start.

Prerequisites
-------------

* Python 3.11 environment (see ``envs/surge-env-devel.yml``)
* device dataset: ``data/datasets/device/case_data`` and ``mhd_code_metadata.yaml`` (already tracked)

Baseline workflow (1k-sample smoke test)
----------------------------------------

.. code-block:: bash

   conda run -n surge python -m examples.mhd_code_workflow \
       --spec configs/mhd_code_demo.yaml \
       --run-tag mhd_code_demo_cli

This spec samples 1,000 rows for a rapid validation run (RFR + Torch MLP + GPflow GPR, with modest Optuna sweeps).

Augmented workflow (all 9,981 samples, 50 Optuna trials/model)
--------------------------------------------------------------

.. code-block:: bash

   conda run -n surge python -m examples.mhd_code_workflow \
       --spec configs/mhd_code_demo_augmented.yaml \
       --run-tag mhd_code_demo_full

The augmented spec removes sampling and increases Optuna sweeps to 50 trials per model, exercising the full dataset.

Notebook exploration
--------------------

.. code-block:: bash

   conda run -n surge jupyter lab notebooks/MHD-code_demo.ipynb

The notebook:

* Analyzes the dataset with ``SurrogateDataset``
* Runs the same workflow (baseline spec by default)
* Visualizes timing/metric tables and multiple comparison plots
* Documents how to switch ``spec_path`` to ``configs/mhd_code_demo_augmented.yaml``

Programmatic workflow invocation
--------------------------------

.. code-block:: python

   from pathlib import Path
   import yaml
   from surge.workflow.spec import SurrogateWorkflowSpec
   from surge.workflow.run import run_surrogate_workflow

   spec_path = Path("configs/mhd_code_demo.yaml")
   spec = SurrogateWorkflowSpec.from_dict(yaml.safe_load(spec_path.read_text()))
   spec.run_tag = "mhd_code_python_api"
   summary = run_surrogate_workflow(spec)
   print(summary["models"][0]["metrics"]["test"])

Each entry in ``summary["models"]`` contains metrics, timings, artifact paths, and Optuna/HPO information.

Inspecting artifacts
--------------------

All workflow outputs land under ``runs/<tag>/``:

* ``models/`` – joblib-serialized adapters
* ``scalers/`` – StandardScaler bundles for inputs/outputs
* ``predictions/`` – per-model CSVs (train/val/test) plus optional UQ JSON
* ``metrics.json`` / ``workflow_summary.json`` – aggregate metrics and dataset/split summary
* ``hpo/<model>_hpo.json`` – serialized Optuna trials
* ``spec.yaml`` / ``environment.txt`` / ``git.txt`` – provenance snapshot

Legacy APIs
-----------

If you still depend on the historical ``SurrogateTrainer`` / ``MLTrainer`` helpers, they remain importable. However, all new development should target ``surge.workflow`` so that CLI runs, notebooks, and programmatic usage stay unified.

Next Steps
----------

* Read the :doc:`examples/index` page for more demonstrations (RF heating, Optuna comparisons, etc.)
* Browse :doc:`api_reference/index` for detailed class/method documentation
