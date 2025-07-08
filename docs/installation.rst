Installation
============

Requirements
------------

SURGE requires Python 3.8 or later and the following dependencies:

* numpy >= 1.19.0
* scipy >= 1.5.0
* scikit-learn >= 0.24.0
* matplotlib >= 3.3.0
* optuna >= 2.10.0 (for hyperparameter optimization)
* gpflow >= 2.5.0 (for Gaussian process models, optional)
* torch >= 1.9.0 (for PyTorch models, optional)

Install from PyPI
-----------------

The easiest way to install SURGE is using pip:

.. code-block:: bash

   pip install surge-surrogate

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/your-username/SURGE.git
   cd SURGE
   pip install -e .

Development Installation
------------------------

For development work:

.. code-block:: bash

   git clone https://github.com/your-username/SURGE.git
   cd SURGE
   pip install -e ".[dev]"
   
This installs SURGE in editable mode with development dependencies including:

* pytest (for testing)
* ruff (for linting and formatting)
* sphinx (for documentation)

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

   import surge
   print(surge.__version__)

Optional Dependencies
---------------------

For full functionality, install optional dependencies:

.. code-block:: bash

   # For Bayesian optimization with BoTorch
   pip install optuna-integration[botorch]
   
   # For Gaussian process models
   pip install gpflow tensorflow
   
   # For PyTorch models
   pip install torch torchvision
