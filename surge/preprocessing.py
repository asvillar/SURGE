"""
Dataset analysis utilities used by the refactored SURGE stack.

This module supersedes the legacy `surge.preprocessing` helpers and adds:
  * Metadata-aware input/output overrides (YAML or dict)
  * Multi-profile grouping hints for scientific surrogate targets
  * Optional sampling knobs for large datasets
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROFILE_HINTS = (
    "gamma",
    "profile",
    "mode",
    "fourier",
    "harmonic",
    "spectrum",
    "psi",
    "rho",
)

OUTPUT_PREFIXES = (
    "y",
    "out",
    "output",
    "target",
    "gamma",
    "beta",
    "profile",
    "stability",
    "growth",
)

INPUT_PREFIXES = ("x", "inp", "input", "feature", "eq", "geom", "basis")

PATTERNS = [
    re.compile(r"^(?P<base>.+?)[\-_](?P<idx>\d+)$"),
    re.compile(r"^(?P<base>.+?)\[(?P<idx>\d+)\]$"),
    re.compile(r"^(?P<base>.+?)(?P<idx>\d+)$"),
]


def _lower_name(column: str) -> str:
    return column.lower()


def _group_repeating_columns(columns: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for col in columns:
        for pattern in PATTERNS:
            match = pattern.match(col)
            if match:
                base = match.group("base")
                groups[base].append(col)
                break
    return {base: sorted(cols) for base, cols in groups.items() if len(cols) > 1}


def _apply_prefix_hints(
    column: str, prefixes: Sequence[str]
) -> bool:
    lowered = _lower_name(column)
    return any(lowered.startswith(prefix) for prefix in prefixes)


def _metadata_list(metadata: Mapping[str, Any], key: str) -> List[str]:
    value = metadata.get(key, [])
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def analyze_dataset_structure(
    df: pd.DataFrame,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    hints: Optional[Mapping[str, Any]] = None,
    sample_size_for_stats: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze a tabular dataset to infer inputs, outputs, and multi-output groupings.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    metadata : dict, optional
        Structured metadata (from YAML or manual dict). Supports keys:
        - inputs: list of column names
        - outputs: list of column names
        - output_groups: mapping of group_name -> [columns]
        - profile_groups: mapping specific to scientific profile bundles
    hints : dict, optional
        Additional overrides (same structure as metadata) used at runtime
        without mutating the persisted metadata.
    sample_size_for_stats : int, optional
        If provided, use at most this many samples when computing dataset
        statistics (mean, std, etc.) to reduce memory pressure.
    """

    if df.empty:
        return {
            "input_variables": [],
            "output_variables": [],
            "output_groups": {},
            "profile_groups": {},
            "dataset_info": {"error": "Empty dataset"},
            "memory_usage_mb": 0,
            "data_types": {},
            "missing_values": {},
            "completeness_percent": 0.0,
            "metadata_overrides": {},
        }

    metadata = metadata or {}
    hints = hints or {}

    manual_inputs = set(_metadata_list(metadata, "inputs"))
    manual_outputs = set(_metadata_list(metadata, "outputs"))

    hint_inputs = set(_metadata_list(hints, "inputs"))
    hint_outputs = set(_metadata_list(hints, "outputs"))

    input_candidates = set(manual_inputs) | hint_inputs
    output_candidates = set(manual_outputs) | hint_outputs

    grouped = _group_repeating_columns(df.columns)

    output_groups: Dict[str, List[str]] = {}
    profile_groups: Dict[str, List[str]] = {}

    # Apply metadata-defined groups first
    for group_name, cols in metadata.get("output_groups", {}).items():
        output_groups[group_name] = [col for col in cols if col in df.columns]
        output_candidates.update(output_groups[group_name])

    for group_name, cols in metadata.get("profile_groups", {}).items():
        profile_groups[group_name] = [col for col in cols if col in df.columns]
        output_candidates.update(profile_groups[group_name])

    # Merge runtime hints
    for group_name, cols in hints.get("output_groups", {}).items():
        existing = output_groups.setdefault(group_name, [])
        for col in cols:
            if col in df.columns and col not in existing:
                existing.append(col)
                output_candidates.add(col)

    for group_name, cols in hints.get("profile_groups", {}).items():
        existing = profile_groups.setdefault(group_name, [])
        for col in cols:
            if col in df.columns and col not in existing:
                existing.append(col)
                output_candidates.add(col)

    # Pattern-based grouping
    for base, cols in grouped.items():
        output_groups.setdefault(base, [])
        for col in cols:
            if col not in output_groups[base]:
                output_groups[base].append(col)
                output_candidates.add(col)
        if any(base.lower().startswith(prefix) for prefix in PROFILE_HINTS):
            profile_groups.setdefault(base, output_groups[base])

    # Heuristic classification for remaining columns
    for col in df.columns:
        if col in input_candidates or col in output_candidates:
            continue
        if _apply_prefix_hints(col, OUTPUT_PREFIXES):
            output_candidates.add(col)
        elif _apply_prefix_hints(col, INPUT_PREFIXES):
            input_candidates.add(col)
        elif col in grouped:
            output_candidates.add(col)
        else:
            # Default to inputs when unsure
            input_candidates.add(col)

    # Ensure columns are classified exactly once
    overlap = input_candidates & output_candidates
    if overlap:
        for col in overlap:
            output_candidates.add(col)
            input_candidates.discard(col)

    input_variables = sorted(col for col in input_candidates if col in df.columns)
    output_variables = sorted(col for col in output_candidates if col in df.columns)

    dataset_info = {
        "shape": df.shape,
        "n_input_variables": len(input_variables),
        "n_output_variables": len(output_variables),
        "n_output_groups": len(output_groups),
        "largest_output_group": max(
            (len(cols) for cols in output_groups.values()), default=0
        ),
        "largest_output_group_name": max(
            output_groups.keys(), key=lambda k: len(output_groups[k]), default=None
        ),
    }

    # Memory usage + dtypes
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
    data_types = {str(dtype): int(count) for dtype, count in df.dtypes.value_counts().items()}

    missing_counts = df.isnull().sum()
    missing_values = {
        column: int(count) for column, count in missing_counts.items() if count > 0
    }
    total_cells = df.shape[0] * df.shape[1]
    completeness_percent = (
        (total_cells - missing_counts.sum()) / total_cells * 100 if total_cells else 0.0
    )

    # Optional statistics sample
    stats_sample = None
    if sample_size_for_stats and sample_size_for_stats < len(df):
        stats_sample = df.sample(n=sample_size_for_stats, random_state=42)

    return {
        "input_variables": input_variables,
        "output_variables": output_variables,
        "output_groups": {k: sorted(v) for k, v in output_groups.items()},
        "profile_groups": {k: sorted(v) for k, v in profile_groups.items()},
        "dataset_info": dataset_info,
        "memory_usage_mb": round(memory_usage_mb, 3),
        "data_types": data_types,
        "missing_values": missing_values,
        "completeness_percent": round(completeness_percent, 2),
        "metadata_overrides": {
            "inputs": sorted(manual_inputs),
            "outputs": sorted(manual_outputs),
        },
        "statistics_sample": stats_sample,
    }


def get_dataset_statistics(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience wrapper around `DataFrame.describe()` with optional sampling."""
    if df.empty:
        return pd.DataFrame()
    if columns:
        df = df[list(columns)]
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        return pd.DataFrame()
    if sample_size and sample_size < len(numeric_cols):
        numeric_cols = numeric_cols.sample(n=sample_size, random_state=42)
    stats = numeric_cols.describe().transpose()
    stats["variance"] = stats["std"] ** 2
    return stats


def print_dataset_analysis(analysis: Mapping[str, Any]) -> str:
    """
    Build a human-readable string summarizing the analysis results.

    Returned string can be printed or logged by callers; returning instead of
    printing makes the function easier to reuse in notebooks/tests.
    """

    info = analysis.get("dataset_info", {})
    if info.get("error"):
        return f"❌ {info['error']}"

    lines = []
    lines.append("📈 DATASET OVERVIEW")
    lines.append(f"Shape: {info.get('shape')}")
    lines.append(f"Memory usage: {analysis.get('memory_usage_mb')} MB")
    lines.append(f"Data types: {analysis.get('data_types')}")

    missing_values = analysis.get("missing_values", {})
    if missing_values:
        lines.append("⚠️ Missing values detected:")
        for col, count in missing_values.items():
            lines.append(f"  - {col}: {count}")
    else:
        lines.append("✅ No missing values detected.")

    lines.append("")
    lines.append("🔍 AUTOMATIC INPUT/OUTPUT DETECTION")
    lines.append(
        f"Inputs: {len(analysis.get('input_variables', []))} | "
        f"Outputs: {len(analysis.get('output_variables', []))}"
    )
    lines.append(
        f"Output groups: {len(analysis.get('output_groups', {}))} | "
        f"Profile groups: {len(analysis.get('profile_groups', {}))}"
    )

    for group_name, cols in analysis.get("output_groups", {}).items():
        preview = ", ".join(cols[:5])
        suffix = "..." if len(cols) > 5 else ""
        lines.append(f"  • {group_name}: {len(cols)} vars ({preview}{suffix})")

    return "\n".join(lines)

