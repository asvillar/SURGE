SURGE Overview
==============

This page summarizes the key building blocks in SURGE after the Pre-ROSE refactor.
Use it as a map when deciding which entry point (CLI, notebook, or Python API)
best suits your workflow.

High-Level Architecture
-----------------------

* **Package layout** – all new code lives directly under ``surge/``:

  - ``surge/dataset.py`` – dataset ingestion + auto-detection
  - ``surge/engine.py`` – splitting, standardization, metrics, timing
  - ``surge/model/registry.py`` – unified model registry + adapter metadata
  - ``surge/workflow/`` – YAML spec + orchestration runner
  - ``surge/io/`` – artifact writers (models, scalers, metrics, predictions)
  - ``surge/viz/`` – reusable plotting helpers (density scatter, violin/SNR, etc.)

* **Entry points**

  - CLI: ``surge run spec.yaml`` (see :doc:`quickstart`)
  - Python API: import ``SurrogateWorkflowSpec`` + ``run_surrogate_workflow``

Dataset Handling
----------------

``SurrogateDataset`` can load CSV/Parquet/Excel/HDF5/NetCDF/Pickle files,
optionally applying metadata overrides (``inputs``/``outputs``) and sampling.
Key points:

* ``from_path`` auto-detects format and returns cleaned ``df``
* ``from_dataframe`` lets you plug in an existing pandas DataFrame
* ``summary()/describe()/stats()`` provide size/memory/missingness info

Engine & Splits
---------------

``SurrogateEngine`` is the core trainer used by the workflow:

* ``configure_dataframe`` / ``configure_from_dataset`` – register a dataset
* ``prepare`` – create deterministic train/val/test splits + scalers
* ``run`` – iterate over ``ModelSpec`` objects, training and recording results
* ``results`` / ``dataset_summary`` – introspection helpers for downstream tests

Model Registry & Adapters
-------------------------

Adapters register themselves in ``surge.model.MODEL_REGISTRY``:

* ``sklearn.random_forest`` – scikit-learn RandomForestRegressor with tree-based UQ
* ``torch.mlp`` – PyTorch FNN (flexible hidden layers, MC-Dropout UQ)
* ``gpflow.gpr`` – GPflow Gaussian Process Regressor with mean/variance output

Each adapter implements ``fit``, ``predict``, ``predict_with_uncertainty`` (optional),
``save``/``load`` for artifact serialization.

Workflow Runner
---------------

``surge.workflow.spec.SurrogateWorkflowSpec`` captures everything needed to train:

* Dataset metadata (paths, format, sampling)
* Split/scaling flags
* ``models`` array of ``ModelConfig`` entries (params + optional ``HPOConfig``)
* Artifact output directory + run tag

``run_surrogate_workflow(spec)``:

1. Loads the dataset via ``SurrogateDataset``
2. Configures ``SurrogateEngine`` and prepares splits
3. Executes Optuna HPO if enabled per model
4. Saves models/scalers/predictions/metrics/spec/environment/git revision under ``runs/<tag>/``
5. Returns a JSON-serializable summary (dataset info, splits, model metrics/timings, artifact paths)

Hyperparameter Optimization
---------------------------

* Optuna is wired through ``HPOConfig`` (per model):

  - ``enabled`` flag (default: ``False``)
  - ``n_trials``, ``timeout``, ``metric``, ``direction``
  - ``search_space`` describing int/float/loguniform/categorical distributions
  - ``sampler`` can be ``tpe`` or ``botorch``

* Winning trials are recorded in ``runs/<tag>/hpo/<model>_hpo.json`` and summarized in ``workflow_summary.json``.

Artifacts & Reproducibility
---------------------------

Each run creates a reproducible bundle:

.. code-block:: text

   runs/<run_tag>/
     models/                   # joblib adapter dumps
     scalers/                  # StandardScaler bundles
     predictions/<model>_<split>.csv
     predictions/<model>_val_uq.json
     metrics.json
     workflow_summary.json
     spec.yaml
     environment.txt           # pip freeze
     git.txt                   # git describe output
     hpo/<model>_hpo.json      # Optuna trials (if enabled)

Visualization
-------------

* ``surge.viz`` provides reusable plotting utilities (density scatter, profile bands, violin/SNR, etc.)
* ``examples/viz_theme_gallery.py`` shows the GT vs. prediction plots, per-mode scatter and violin/SNR panels.

Next Steps
----------

* Jump to :doc:`quickstart` for step-by-step installation and execution instructions
* Explore ``examples/viz_theme_gallery.py`` (visual)
* Inspect ``tests/test_workflow_new.py`` for lightweight examples of the new API







