Examples
========

Runnable scripts live in the ``examples/`` folder of the repository. Run each
command from the repository root.

Getting started
---------------

``examples/quickstart.py``
   Train and evaluate a surrogate on a built-in scikit-learn dataset.
   ``python -m examples.quickstart --dataset diabetes``

``examples/run_workflow.py``
   Run any workflow spec YAML through the Python API.
   ``python examples/run_workflow.py --spec examples/configs/custom_dataset_tutorial.yaml``

``examples/custom_dataset_tutorial.py``
   Train on your own table (CSV, Parquet, pickle or HDF5); the code behind
   :doc:`../BUILD_YOUR_OWN_SURROGATE`.
   ``python examples/custom_dataset_tutorial.py``

``examples/qlknn_workflow.py``
   Residual MLP on the QLKNN transport data, with optional hyperparameter search.
   ``python examples/qlknn_workflow.py --epochs 20 --no-hpo``

Leaderboards
------------

Each script compares several models on a group of benchmarks; the same runs
are available from the CLI with ``surge bench``.

``examples/leaderboard_tabular.py``
   Tabular regression and classification.
   ``python examples/leaderboard_tabular.py``

``examples/leaderboard_pde.py``
   PDE and operator-learning benchmarks.
   ``python examples/leaderboard_pde.py --quick``

``examples/leaderboard_plasma.py``
   Plasma and fusion benchmarks.
   ``python examples/leaderboard_plasma.py``

``examples/leaderboard_vision.py``
   Image classification benchmarks.
   ``python examples/leaderboard_vision.py``

Custom models
-------------

``examples/custom_cnn_adapter_template.py``
   A small 1-D CNN adapter and its registration call; see
   :doc:`../ADDING_NEW_MODEL_ADAPTER`.
   ``python examples/custom_cnn_adapter_template.py``
