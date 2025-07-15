Installation
============

🚀 Getting Started
------------------

From PyPI (when published)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install surge-surrogate

For Development
~~~~~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/your-username/SURGE.git
   cd SURGE
   pip install -e .

Or install with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Requirements
------------

SURGE requires Python 3.7+ and the following core dependencies:

* numpy >= 1.19.0
* scipy >= 1.5.0
* scikit-learn >= 0.24.0
* matplotlib >= 3.3.0
* pandas
* optuna >= 2.10.0 (for hyperparameter optimization)

Optional Dependencies
---------------------

For extended functionality, install optional dependencies:

* **Gaussian Process models**: ``pip install gpflow tensorflow``
* **PyTorch neural networks**: ``pip install torch torchvision``
* **Bayesian optimization with BoTorch**: ``pip install optuna-integration[botorch]``
* **Jupyter notebooks**: ``pip install jupyter``

Development Dependencies
------------------------

For development work, the following additional packages are included:

* pytest (for testing)
* ruff (for linting and formatting)
* sphinx (for documentation)
* mypy (for type checking)

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

   import surge
   from surge import SurrogateTrainer
   print("SURGE installation successful!")
