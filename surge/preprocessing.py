from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler  # Add StandardScaler import
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any

# Re-export StandardScaler for convenience
from sklearn.preprocessing import StandardScaler


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_cv_splits(X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(X, y))


def analyze_dataset_structure(
    df: pd.DataFrame, 
    memory_efficient: bool = True,
    max_correlation_features: int = 50,
    sample_size_for_stats: Optional[int] = None
) -> Dict[str, Any]:
    """
    Efficiently analyze dataset structure to automatically identify input/output variables
    based on column naming patterns and provide comprehensive dataset statistics.
    
    This function identifies output variables by detecting repeating patterns (e.g., var_1, var_2, ...)
    and treats non-repeating columns as input variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze
    memory_efficient : bool, default=True
        If True, uses memory-efficient operations for large datasets
    max_correlation_features : int, default=50
        Maximum number of features to include in correlation analysis
    sample_size_for_stats : int, optional
        Sample size for statistical analysis. If None, uses full dataset for small datasets
        or 10% sample for large datasets (>100k rows)
        
    Returns:
    --------
    Dict containing:
        - 'input_variables': List of identified input variable names
        - 'output_groups': Dict mapping group names to lists of output variables
        - 'output_variables': List of all output variables
        - 'dataset_info': Dict with basic dataset information
        - 'memory_usage_mb': Dataset memory usage in MB
        - 'data_types': Count of different data types
        - 'missing_values': Dict of columns with missing values
        - 'completeness_percent': Overall data completeness percentage
    """
    
    if df.empty:
        return {
            'input_variables': [],
            'output_groups': {},
            'output_variables': [],
            'dataset_info': {'error': 'Empty dataset'},
            'memory_usage_mb': 0,
            'data_types': {},
            'missing_values': {},
            'completeness_percent': 0
        }
    
    # Basic dataset information (memory efficient)
    n_rows, n_cols = df.shape
    
    # Memory usage calculation
    if memory_efficient and n_rows > 10000:
        # Sample for memory calculation on large datasets
        sample_df = df.sample(n=min(1000, n_rows), random_state=42)
        memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        total_memory_mb = (memory_per_row * n_rows) / (1024**2)
    else:
        total_memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    
    # Data types analysis
    data_types = df.dtypes.value_counts().to_dict()
    data_types = {str(k): v for k, v in data_types.items()}  # Convert dtypes to strings
    
    # Missing values analysis (memory efficient)
    if memory_efficient and n_rows > 50000:
        # For very large datasets, check missing values in chunks
        missing_counts = {}
        chunk_size = 10000
        for i in range(0, n_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_missing = chunk.isnull().sum()
            for col, count in chunk_missing.items():
                missing_counts[col] = missing_counts.get(col, 0) + count
    else:
        missing_counts = df.isnull().sum().to_dict()
    
    missing_values = {col: count for col, count in missing_counts.items() if count > 0}
    
    # Overall completeness
    total_cells = n_rows * n_cols
    missing_cells = sum(missing_counts.values())
    completeness_percent = ((total_cells - missing_cells) / total_cells) * 100
    
    # Automatic input/output variable detection
    column_groups = defaultdict(list)
    standalone_columns = []
    
    # Compile regex patterns once for efficiency
    pattern_underscore = re.compile(r'^(.+?)_(\d+)(_?)$')
    pattern_no_underscore = re.compile(r'^(.+?)(\d+)$')
    
    for col in df.columns:
        # Look for patterns like var_1, var_2, etc.
        match = pattern_underscore.match(col)
        if match:
            base_name = match.group(1)
            column_groups[base_name].append(col)
        else:
            # Check for patterns like var1, var2 (without underscore)
            match = pattern_no_underscore.match(col)
            if match:
                base_name = match.group(1)
                column_groups[base_name].append(col)
            else:
                standalone_columns.append(col)
    
    # Identify outputs (grouped columns) vs inputs (standalone columns)
    output_groups = {k: sorted(v) for k, v in column_groups.items() if len(v) > 1}
    
    # Collect all output variables
    output_variables = []
    for group_cols in output_groups.values():
        output_variables.extend(group_cols)
    
    # Input variables are standalone columns plus single-instance groups
    input_variables = standalone_columns.copy()
    for k, v in column_groups.items():
        if len(v) == 1:
            input_variables.extend(v)
    
    # Sort for consistency
    input_variables = sorted(input_variables)
    output_variables = sorted(output_variables)
    
    # Dataset info summary
    dataset_info = {
        'shape': (n_rows, n_cols),
        'n_input_variables': len(input_variables),
        'n_output_variables': len(output_variables),
        'n_output_groups': len(output_groups),
        'largest_output_group': max([len(v) for v in output_groups.values()]) if output_groups else 0,
        'largest_output_group_name': max(output_groups.keys(), 
                                       key=lambda k: len(output_groups[k])) if output_groups else None
    }
    
    return {
        'input_variables': input_variables,
        'output_groups': output_groups,
        'output_variables': output_variables,
        'dataset_info': dataset_info,
        'memory_usage_mb': round(total_memory_mb, 2),
        'data_types': data_types,
        'missing_values': missing_values,
        'completeness_percent': round(completeness_percent, 1)
    }


def get_dataset_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    memory_efficient: bool = True
) -> pd.DataFrame:
    """
    Get statistical summary for specified columns in a memory-efficient way.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        Columns to analyze. If None, analyzes all numeric columns
    sample_size : int, optional
        Sample size for large datasets. If None, auto-determines based on dataset size
    memory_efficient : bool, default=True
        Use memory-efficient operations for large datasets
        
    Returns:
    --------
    pd.DataFrame
        Statistical summary (describe() output)
    """
    
    if df.empty:
        return pd.DataFrame()
    
    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Determine if sampling is needed
    n_rows = len(df)
    if sample_size is None:
        if memory_efficient and n_rows > 100000:
            sample_size = min(10000, n_rows // 10)  # 10% sample, max 10k rows
        else:
            sample_size = n_rows  # Use full dataset
    
    # Get sample if needed
    if sample_size < n_rows:
        sample_df = df[columns].sample(n=sample_size, random_state=42)
        stats = sample_df.describe()
        # Add note that this is from a sample
        stats.loc['sample_size'] = sample_size
    else:
        stats = df[columns].describe()
        stats.loc['sample_size'] = n_rows
    
    return stats


def print_dataset_analysis(analysis_result: Dict, verbose: bool = True) -> None:
    """
    Pretty print the results from analyze_dataset_structure().
    
    Parameters:
    -----------
    analysis_result : Dict
        Result from analyze_dataset_structure()
    verbose : bool, default=True
        If True, prints detailed information
    """
    
    info = analysis_result['dataset_info']
    
    if 'error' in info:
        print(f"❌ {info['error']}")
        return
    
    print("=" * 50)
    print("📈 DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape: {info['shape']}")
    print(f"Memory usage: {analysis_result['memory_usage_mb']} MB")
    print(f"Data types: {analysis_result['data_types']}")
    
    if analysis_result['missing_values']:
        print(f"\n⚠️  Missing values found:")
        for col, count in analysis_result['missing_values'].items():
            print(f"   {col}: {count}")
    else:
        print("\n✅ No missing values found")
    
    print("\n" + "=" * 50)
    print("🔍 AUTOMATIC INPUT/OUTPUT DETECTION")
    print("=" * 50)
    
    output_groups = analysis_result['output_groups']
    input_vars = analysis_result['input_variables']
    
    print(f"🎯 **OUTPUT VARIABLE GROUPS** (repeating patterns): {len(output_groups)}")
    for base_name, cols in output_groups.items():
        print(f"   📊 {base_name}: {len(cols)} variables")
        if verbose:
            display_cols = cols[:5] + ['...'] if len(cols) > 5 else cols
            print(f"      └── {display_cols}")
    
    print(f"\n📥 **INPUT VARIABLES** (non-repeating): {len(input_vars)}")
    if verbose:
        display_inputs = input_vars[:10] + ['...'] if len(input_vars) > 10 else input_vars
        print(f"   Variables: {display_inputs}")
    
    print(f"\n📊 **SUMMARY:**")
    print(f"   ✅ Total columns: {info['shape'][1]}")
    print(f"   📥 Input variables: {info['n_input_variables']}")
    print(f"   📤 Output variables: {info['n_output_variables']}")
    print(f"   🏷️  Output groups: {info['n_output_groups']}")
    print(f"   📊 Data completeness: {analysis_result['completeness_percent']}%")
    
    if info['largest_output_group_name']:
        print(f"   🎯 Largest output group: '{info['largest_output_group_name']}' "
              f"({info['largest_output_group']} variables)")
