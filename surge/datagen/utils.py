"""
Dataset Utilities for SURGE

This module provides common utilities for dataset operations including validation,
statistics, and format conversion.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def validate_dataset(
    df: pd.DataFrame,
    check_nan: bool = True,
    check_inf: bool = True,
    check_duplicates: bool = True,
    check_types: bool = True,
) -> Dict[str, Any]:
    """
    Validate a dataset for common issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate
    check_nan : bool, default=True
        Check for NaN values
    check_inf : bool, default=True
        Check for infinite values
    check_duplicates : bool, default=True
        Check for duplicate rows
    check_types : bool, default=True
        Check for mixed types in columns
        
    Returns
    -------
    dict
        Validation results with keys:
        - 'is_valid': bool, overall validation status
        - 'has_nan': bool, whether NaN values found
        - 'has_inf': bool, whether infinite values found
        - 'has_duplicates': bool, whether duplicate rows found
        - 'nan_columns': list, columns with NaN values
        - 'inf_columns': list, columns with infinite values
        - 'n_duplicates': int, number of duplicate rows
        - 'mixed_type_columns': list, columns with mixed types
        - 'warnings': list, warning messages
        - 'errors': list, error messages
    """
    results = {
        'is_valid': True,
        'has_nan': False,
        'has_inf': False,
        'has_duplicates': False,
        'nan_columns': [],
        'inf_columns': [],
        'n_duplicates': 0,
        'mixed_type_columns': [],
        'warnings': [],
        'errors': [],
    }
    
    if df.empty:
        results['errors'].append("Dataset is empty")
        results['is_valid'] = False
        return results
    
    # Check for NaN values
    if check_nan:
        nan_counts = df.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        if len(nan_columns) > 0:
            results['has_nan'] = True
            results['nan_columns'] = nan_columns.to_dict()
            results['warnings'].append(
                f"Found NaN values in {len(nan_columns)} columns: {list(nan_columns.index)}"
            )
    
    # Check for infinite values (only numeric columns)
    if check_inf:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_mask = np.isinf(df[numeric_cols]).any()
            inf_columns = inf_mask[inf_mask].index.tolist()
            if len(inf_columns) > 0:
                results['has_inf'] = True
                results['inf_columns'] = inf_columns
                results['warnings'].append(
                    f"Found infinite values in {len(inf_columns)} columns: {inf_columns}"
                )
    
    # Check for duplicate rows
    if check_duplicates:
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            results['has_duplicates'] = True
            results['n_duplicates'] = int(n_duplicates)
            results['warnings'].append(f"Found {n_duplicates} duplicate rows")
    
    # Check for mixed types (object columns that might contain mixed types)
    if check_types:
        object_cols = df.select_dtypes(include=['object']).columns
        mixed_type_cols = []
        for col in object_cols:
            # Check if column contains mixed types
            types = df[col].apply(type).unique()
            if len(types) > 1:
                mixed_type_cols.append(col)
        
        if len(mixed_type_cols) > 0:
            results['mixed_type_columns'] = mixed_type_cols
            results['warnings'].append(
                f"Found mixed types in {len(mixed_type_cols)} columns: {mixed_type_cols}"
            )
    
    # Overall validation status
    if results['errors']:
        results['is_valid'] = False
    
    return results


def get_dataset_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    include_stats: bool = True,
) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics for a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize
    columns : list of str, optional
        Specific columns to summarize. If None, summarizes all columns.
    include_stats : bool, default=True
        Whether to include detailed statistics (min/max/mean/std)
        
    Returns
    -------
    dict
        Summary dictionary with keys:
        - 'shape': tuple, (n_rows, n_cols)
        - 'memory_usage_mb': float, memory usage in MB
        - 'dtypes': dict, data type counts
        - 'numeric_columns': list, numeric column names
        - 'categorical_columns': list, categorical/object column names
        - 'statistics': pd.DataFrame, detailed statistics (if include_stats=True)
    """
    if columns is None:
        columns = df.columns.tolist()
    else:
        columns = [c for c in columns if c in df.columns]
    
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'dtypes': df[columns].dtypes.value_counts().to_dict(),
        'numeric_columns': df[columns].select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df[columns].select_dtypes(include=['object', 'category']).columns.tolist(),
    }
    
    if include_stats:
        numeric_cols = summary['numeric_columns']
        if len(numeric_cols) > 0:
            summary['statistics'] = df[numeric_cols].describe()
        else:
            summary['statistics'] = pd.DataFrame()
    
    return summary


def dataframe_to_pytorch_dataset(
    df: pd.DataFrame,
    input_columns: List[str],
    output_columns: List[str],
    device: Optional[str] = None,
) -> 'TensorDataset':
    """
    Convert pandas DataFrame to PyTorch TensorDataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    input_columns : list of str
        Column names for input features
    output_columns : list of str
        Column names for output targets
    device : str, optional
        Device to place tensors on ('cpu', 'cuda', etc.). If None, uses default.
        
    Returns
    -------
    torch.utils.data.TensorDataset
        PyTorch dataset ready for training
        
    Raises
    ------
    ImportError
        If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TensorDataset conversion. "
            "Install with: pip install torch"
        )
    
    # Extract input and output data
    X = df[input_columns].values.astype(np.float32)
    y = df[output_columns].values.astype(np.float32)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    # Move to device if specified
    if device is not None:
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    return dataset


def split_dataframe_by_size(
    df: pd.DataFrame,
    max_size_mb: float = 100.0,
    chunk_size: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Split a large DataFrame into smaller chunks based on memory size.
    
    Useful for processing very large datasets that don't fit in memory.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split
    max_size_mb : float, default=100.0
        Maximum size per chunk in MB
    chunk_size : int, optional
        Number of rows per chunk. If None, calculates based on max_size_mb.
        
    Returns
    -------
    list of pd.DataFrame
        List of DataFrame chunks
    """
    if chunk_size is None:
        # Estimate rows per MB
        sample_size = min(1000, len(df))
        sample_mb = df.head(sample_size).memory_usage(deep=True).sum() / (1024**2)
        rows_per_mb = sample_size / sample_mb if sample_mb > 0 else 1000
        chunk_size = int(max_size_mb * rows_per_mb)
    
    chunks = []
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)
    
    return chunks


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Parameters
    ----------
    validation_results : dict
        Results from validate_dataset()
    """
    print("=" * 70)
    print("📊 DATASET VALIDATION REPORT")
    print("=" * 70)
    
    if validation_results['is_valid']:
        print("✅ Dataset is VALID")
    else:
        print("❌ Dataset has ERRORS")
    
    print("\n" + "─" * 70)
    print("ISSUES FOUND")
    print("─" * 70)
    
    if validation_results['has_nan']:
        print(f"⚠️  NaN values found in {len(validation_results['nan_columns'])} columns")
        for col, count in list(validation_results['nan_columns'].items())[:5]:
            print(f"   - {col}: {count} NaN values")
        if len(validation_results['nan_columns']) > 5:
            print(f"   ... and {len(validation_results['nan_columns']) - 5} more columns")
    
    if validation_results['has_inf']:
        print(f"⚠️  Infinite values found in {len(validation_results['inf_columns'])} columns")
        for col in validation_results['inf_columns'][:5]:
            print(f"   - {col}")
        if len(validation_results['inf_columns']) > 5:
            print(f"   ... and {len(validation_results['inf_columns']) - 5} more columns")
    
    if validation_results['has_duplicates']:
        print(f"⚠️  {validation_results['n_duplicates']} duplicate rows found")
    
    if validation_results['mixed_type_columns']:
        print(f"⚠️  Mixed types found in {len(validation_results['mixed_type_columns'])} columns")
        for col in validation_results['mixed_type_columns'][:5]:
            print(f"   - {col}")
        if len(validation_results['mixed_type_columns']) > 5:
            print(f"   ... and {len(validation_results['mixed_type_columns']) - 5} more columns")
    
    if not any([
        validation_results['has_nan'],
        validation_results['has_inf'],
        validation_results['has_duplicates'],
        validation_results['mixed_type_columns'],
    ]):
        print("✅ No issues found")
    
    # Print warnings and errors
    if validation_results['warnings']:
        print("\n" + "─" * 70)
        print("WARNINGS")
        print("─" * 70)
        for warning in validation_results['warnings']:
            print(f"⚠️  {warning}")
    
    if validation_results['errors']:
        print("\n" + "─" * 70)
        print("ERRORS")
        print("─" * 70)
        for error in validation_results['errors']:
            print(f"❌ {error}")
    
    print("=" * 70)


__all__ = [
    'validate_dataset',
    'get_dataset_summary',
    'dataframe_to_pytorch_dataset',
    'split_dataframe_by_size',
    'print_validation_report',
    'TORCH_AVAILABLE',
]





