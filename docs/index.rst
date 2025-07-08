Welcome to SURGE Documentation
==============================

SURGE (Surrogate Models for Understanding and Rapid Generation of Experiments) is a Python library for surrogate modeling and machine learning applications in scientific computing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference/index
   examples/index
   contributing
   changelog

Features
--------

* **Machine Learning Models**: Support for various ML models including GPflow, PyTorch, and scikit-learn
* **Hyperparameter Optimization**: Integration with Optuna for Bayesian optimization
* **Preprocessing Utilities**: Data preprocessing and feature engineering tools
* **Metrics and Evaluation**: Comprehensive model evaluation metrics
* **Extensible Architecture**: Easy to extend with custom models and methods

Quick Start
-----------

Install SURGE:

.. code-block:: bash

   pip install surge-surrogate

Basic usage:

.. code-block:: python

   import surge
   from surge.models import RandomForestModel
   
   # Create and train a model
   model = RandomForestModel()
   model.fit(X_train, y_train)
   
   # Make predictions
   predictions = model.predict(X_test)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`on master file, created by
   sphinx-quickstart on Mon Jul  7 16:44:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SURGE's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
