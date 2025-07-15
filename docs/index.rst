.. image:: ../surge_logo_panoramic.png
   :align: center
   :alt: SURGE Logo
   :width: 400px

SURGE
=====

*A Surrogate Unified Robust Generation Engine*

**SURGE** is a modular AI/ML framework for building fast, accurate, and uncertainty-aware 
surrogate models that emulate complex scientific simulations. Designed for flexibility and 
scientific rigor, it supports ensemble regressors, neural networks, and Gaussian processes, 
streamlining the development and deployment of surrogates for inference, optimization, and control.

.. _surge-pipeline:

End-to-End Surrogate Modeling Pipeline
======================================

SURGE provides a fully automated AI/ML-powered pipeline for surrogate generation:

- **Data Collection (Sampling)**  
  Integrate with any simulator or experiment to gather parameter–output pairs.

- **Model Selection**  
  Choose among ensemble regressors (RFR), neural networks (MLP), and Gaussian processes (GPR) via a unified API.

- **Hyperparameter Tuning**  
  Automate Bayesian or random search over model hyperparameters to find optimal architectures.

- **Training & Testing (Verification, Validation & UQ)**  
  Run configurable training loops with cross-validation, track mean squared error and R², and quantify uncertainty (via GPR).

- **Performance Evaluation**  
  Measure inference time, statistical metrics, and generate comprehensive reports.

- **Uncertainty Quantification**  
  Provide principled error bounds on predictions using built-in GPR support.

- **Data & Model I/O**  
  Save and load data and trained models in ONNX, HDF5, NPZ, PTH, or joblib formats.

- **Model Import & Inference**  
  Deploy surrogates in Python scripts, services, or embedded workflows for real-time or batch inference.

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

- **Unified API** for RFR, MLP, and GPR, with identical calls for training, evaluation, and prediction.  
- **Configurable Pipelines** via YAML/JSON, enabling reproducible workflows and easy parameter sweeps.  
- **Automated Cross-Validation** with built-in reporting of MSE, R², mean/std inference time.  
- **Hyperparameter Optimization** wrappers for both random and Bayesian search (Optuna, BoTorch, GPflow).  
- **Pluggable I/O** for common formats (ONNX, HDF5, NPZ, joblib), easing model exchange and deployment.  

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api_reference/index
   examples/index

🚀 Quick Start
--------------

Install SURGE:

.. code-block:: bash

   pip install surge-surrogate

You can see examples of SURGE in action in the :doc:`examples/index` section and get started with just a few lines of code:

.. code-block:: python

   from surge import SurrogateTrainer

   # train a GPR surrogate on your data
   trainer = SurrogateTrainer(model_type="gpr", optimizer="bayesian", n_trials=30)
   trainer.fit(X, y, test_size=0.2)
   results = trainer.cross_validate(n_splits=5)
   predictions = trainer.predict(X_new)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
