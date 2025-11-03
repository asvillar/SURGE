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

