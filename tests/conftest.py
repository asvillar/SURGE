"""
Pytest configuration for SURGE tests.

This file configures pytest to handle warnings appropriately and set default test behavior.
"""
import warnings

# Filter out common warnings that are expected but not critical
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*convergence.*",  # MLPRegressor convergence warnings
)

# Suppress DataConversionWarning from sklearn (we handle y shape conversion in code)
warnings.filterwarnings(
    "ignore",
    message=".*column-vector y was passed when a 1d array was expected.*",
    category=UserWarning,
    module="sklearn",
)

# Show deprecation warnings but don't fail tests on them
warnings.filterwarnings(
    "default",
    category=DeprecationWarning,
    module="sklearn|pandas|numpy",
)

# Ignore FutureWarnings from sklearn/pandas (these are about future API changes)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn|pandas",
)



def optional_backend(name: str):
    """Import an optional backend or SKIP the test.

    Unlike pytest.importorskip (ImportError only), this also skips when a
    native library fails to load (OSError / the package's own error type,
    e.g. lightgbm without libomp) — a broken optional dependency must never
    read as a SURGE test failure.
    """
    import importlib

    import pytest as _pytest

    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any import failure => skip
        _pytest.skip(f"{name} unavailable: {type(exc).__name__}: {exc}")
