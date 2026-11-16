.. image:: ../data/logos/surge_logo_os_expanded.png
   :align: center
   :alt: SURGE — Surrogate Unified Robust Generation Engine
   :width: 640px

.. |doi_surge| replace:: `10.11578/dc.20260422.5 <https://doi.org/10.11578/dc.20260422.5>`__

SURGE
=====

*Surrogate Unified Robust Generation Engine*

**Current status (v0.1.0):** the code is open source on
`GitHub <https://github.com/asvillar/SURGE>`__, registered in
`U.S. DOE OSTI / DOE Code <https://www.osti.gov/doecode/biblio/179819>`__
(``179819``, DOI |doi_surge|), with continuous integration on
``main`` and a `documented
<https://github.com/asvillar/SURGE/blob/main/README.md>`__ install +
quickstart (``uv``, ``.[torch,onnx,dev]``, including ``h5py`` for
HDF5 tests). The test suite: **52 passed, 1 skipped** in a full
dev environment (see :doc:`citation` for the formal cite).

**SURGE** is a scientific surrogate-modeling library: a single
**declarative pipeline** (``run_surrogate_workflow``,
``SurrogateWorkflowSpec``) that covers data ingest, train/val/test
splits, model adapters (scikit-learn, PyTorch MLP, optional GPflow), Optuna
HPO, **artifact lineage** (``runs/<tag>/``), optional ONNX export, and
diagnostic plots (``surge.viz``).

.. _surge-pipeline:

End-to-End Surrogate Modeling Pipeline
======================================

SURGE provides a fully automated AI/ML-powered pipeline for surrogate generation:

1. **Dataset Ingest** – ``SurrogateDataset`` auto-detects inputs/outputs, applies metadata overrides, drops NaNs, and summarizes stats/memory/missingness.
2. **Splitting & Scaling** – ``SurrogateEngine`` builds deterministic train/val/test splits and applies ``StandardScaler`` bundles (stored per run).
3. **Model Registry** – RFR (scikit-learn), Torch MLP with MC-Dropout, and GPflow GPR are registered adapters with unified ``fit/predict/predict_with_uncertainty`` APIs.
4. **Hyperparameter Optimization** – Optuna-driven sweeps (TPE or BoTorch) can be toggled per model via YAML or ``HPOConfig``; trials export to JSON.
5. **Training & Validation** – Metrics (R², RMSE, MAE, MAPE) and timings (train/inference per sample) are recorded for train/val/test.
6. **Artifacts** – Models, scalers, predictions, metrics, spec, environment snapshot, and git revision land under ``runs/<tag>/``.
7. **Visualization** – ``surge.viz`` exposes density scatter, violin/SNR, correlation heatmaps, and and per-mode scatter utilities.

This vertical "code → automation → surrogate inference" flow means you can move from raw simulation code to deployable, uncertainty-aware surrogate models with minimal boilerplate.

Main Applications
-----------------

SURGE has been designed to support a wide range of scientific and engineering use cases:

- **Real-time Control & Monitoring**  
  Replace expensive physics solvers with fast surrogates for in-shot decision making.

- **Design Optimization**  
  Embed surrogate models into optimization loops (e.g., Bayesian, genetic) for rapid parameter sweeps.

- **Uncertainty-Aware Prediction**  
  Leverage Gaussian Process Regressors to get both point estimates and confidence bounds.

- **Digital Twin Integration**  
  Seamlessly integrate surrogates into digital-twin frameworks for what-if analyses.

- **High-Performance Computing (HPC)**  
  Streamline training and inference workflows on clusters using standard ML libraries.

Core Functionalities
--------------------

- **Workflow-first API** – use ``run_surrogate_workflow`` with YAML specs (see ``examples/configs/``).  
- **Dataset analysis** – ``surge.dataset`` handles multi-format loads (CSV, HDF5, NetCDF, pickle) and enforces clean numeric frames before scaling.  
- **Optuna integration** – per-model search spaces defined inline, supporting categorical lists (e.g., hidden layer templates) and loguniform ranges.  
- **Artifact standardization** – consistent directory layout for reproducibility + notebooks/tests.  
- **Spec parity** – CLI runs and Python API calls share the same YAML specs.  

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   quickstart
   gallery
   BUILD_YOUR_OWN_SURROGATE
   CUSTOM_DATASET_TUTORIAL
   citation
   comparison
   api_reference/index

Quick Start
-----------

SURGE runs end-to-end on any tabular CSV. Here's a complete train →
evaluate → profile cycle on scikit-learn's 442-row diabetes dataset,
which runs in under 5 seconds on a laptop:

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

All artifacts (trained model, scaler, metrics, predictions, provenance)
land in ``runs/quickstart/``. See :doc:`quickstart` for a full tour and
the project README for a verbatim output transcript.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
