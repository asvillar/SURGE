Installation
============

Requirements
------------

* **Python:** 3.10 or 3.11 (see ``requires-python`` in the root ``pyproject.toml``).
* **Install name:** the PyPI distribution is **surge-ml**; the import name is **surge**.

From PyPI (after publish)
-------------------------

.. code-block:: bash

   python -m pip install "surge-ml[torch,onnx,dev]"

The ``dev`` extra pulls ``pytest``, ``ruff``, and ``h5py`` (HDF5 / M3DC1-related tests). For documentation builds, add the ``docs`` extra or install Sphinx dependencies (see the ``[docs]`` extra in ``pyproject.toml``).

From a clone (recommended for development)
-------------------------------------------

.. code-block:: bash

   git clone https://github.com/S-Villar/SURGE.git
   cd SURGE
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[torch,onnx,dev]"

Using `uv <https://github.com/astral-sh/uv>`__ is supported and faster; see
:doc:`index` and the
`README <https://github.com/S-Villar/SURGE/blob/main/README.md>`__ for a
copy-paste recipe.

Core and optional stack
-----------------------

SURGE’s base install includes NumPy, SciPy, scikit-learn, pandas, Parquet
support, PyYAML, Matplotlib, Optuna, and related tooling. Optional extras
include:

* **torch** — PyTorch (MLP adapter, etc.)
* **onnx** — export and ONNX Runtime checks
* **dev** — test and lint tools plus ``h5py``
* **docs** — Sphinx, Furo, MyST (for this site)

Verify
------

.. code-block:: python

   import surge
   from surge import SurrogateWorkflowSpec
   assert surge.__version__
   print("SURGE", surge.__version__, "— OK")

HPC / NERSC
-----------

See the walkthrough in the repository:
the project README's install section.
