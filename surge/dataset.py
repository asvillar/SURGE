"""
SurrogateDataset - Generalized dataset loading and auto-detection.

This module provides the SurrogateDataset class for loading datasets from various
formats and automatically detecting input/output variables using pattern analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .preprocessing import (
    analyze_dataset_structure,
    get_dataset_statistics,
    print_dataset_analysis,
)


class SurrogateDataset:
    """
    General-purpose dataset loader with automatic input/output detection.
    
    This class handles loading datasets from various file formats (CSV, pickle,
    parquet, Excel) and automatically identifies input and output variables based
    on column naming patterns (e.g., var_1, var_2, ...).
    
    Uses the preprocessing.analyze_dataset_structure() function for generalized
    pattern-based detection, avoiding hardcoded naming conventions.
    
    Examples
    --------
    >>> dataset = SurrogateDataset()
    >>> dataset.load_from_file('data.pkl')
    >>> print(dataset.input_columns)
    >>> print(dataset.output_columns)
    
    >>> # Or with manual specification
    >>> dataset.load_from_file('data.csv', input_cols=['x1', 'x2'], output_cols=['y1', 'y2'])
    """
    
    def __init__(self):
        """Initialize SurrogateDataset."""
        self.df: Optional[pd.DataFrame] = None
        self.input_columns: List[str] = []
        self.output_columns: List[str] = []
        self.output_groups: Dict[str, List[str]] = {}
        self.analysis: Optional[Dict[str, Any]] = None
        self.file_path: Optional[Path] = None
        
    def load_from_file(
        self,
        file_path: Union[str, Path],
        input_cols: Optional[List[str]] = None,
        output_cols: Optional[List[str]] = None,
        auto_detect: bool = True,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from file with optional auto-detection.
        
        Supports CSV, pickle (.pkl, .pickle), parquet (.parquet, .pq), and Excel (.xlsx, .xls).
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        input_cols : list of str, optional
            Manual specification of input column names. If None and auto_detect=True,
            will attempt automatic detection.
        output_cols : list of str, optional
            Manual specification of output column names. If None and auto_detect=True,
            will attempt automatic detection.
        auto_detect : bool, default=True
            If True, automatically detect inputs/outputs using pattern analysis.
            If False, requires manual specification of input_cols and output_cols.
        **kwargs
            Additional arguments passed to pandas read function.
            
        Returns
        -------
        tuple
            (input_columns, output_columns) lists
            
        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If auto_detect=False and columns not specified
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        
        # Load data based on file extension
        print(f"📂 Loading dataset from: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            self.df = pd.read_csv(file_path, **kwargs)
        elif suffix in ['.pkl', '.pickle']:
            self.df = pd.read_pickle(file_path)
        elif suffix in ['.parquet', '.pq']:
            self.df = pd.read_parquet(file_path, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .pkl, .parquet, .xlsx")
        
        print(f"✅ Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} columns")
        
        # Auto-detect or use manual specification
        if auto_detect and (input_cols is None or output_cols is None):
            print("\n🔍 Auto-detecting input/output columns...")
            self._auto_detect_columns()
        elif input_cols is not None and output_cols is not None:
            print("\n📋 Using manually specified columns...")
            self.input_columns = input_cols
            self.output_columns = output_cols
            # Still run analysis for metadata
            self.analysis = analyze_dataset_structure(self.df)
            self.output_groups = self.analysis.get('output_groups', {})
        else:
            raise ValueError(
                "Must provide both input_cols and output_cols when auto_detect=False, "
                "or set auto_detect=True for automatic detection"
            )
        
        return self.input_columns, self.output_columns
    
    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        input_cols: Optional[List[str]] = None,
        output_cols: Optional[List[str]] = None,
        auto_detect: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from pandas DataFrame with optional auto-detection.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        input_cols : list of str, optional
            Manual specification of input column names
        output_cols : list of str, optional
            Manual specification of output column names
        auto_detect : bool, default=True
            If True, automatically detect inputs/outputs
            
        Returns
        -------
        tuple
            (input_columns, output_columns) lists
        """
        self.df = df
        self.file_path = None  # No file path for in-memory data
        
        if auto_detect and (input_cols is None or output_cols is None):
            print("🔍 Auto-detecting input/output columns...")
            self._auto_detect_columns()
        elif input_cols is not None and output_cols is not None:
            print("📋 Using manually specified columns...")
            self.input_columns = input_cols
            self.output_columns = output_cols
            self.analysis = analyze_dataset_structure(self.df)
            self.output_groups = self.analysis.get('output_groups', {})
        else:
            raise ValueError(
                "Must provide both input_cols and output_cols when auto_detect=False, "
                "or set auto_detect=True for automatic detection"
            )
        
        return self.input_columns, self.output_columns
    
    def _auto_detect_columns(self) -> None:
        """
        Automatically detect input and output columns using pattern analysis.
        
        Uses analyze_dataset_structure() from preprocessing module, which detects
        repeating patterns (e.g., var_1, var_2, ...) as outputs and standalone
        columns as inputs. This is a generalized approach that works with any
        naming convention, not hardcoded to specific prefixes.
        """
        # Perform comprehensive dataset structure analysis
        self.analysis = analyze_dataset_structure(self.df)
        
        # Print analysis results
        print("\n📊 Dataset Analysis Results:")
        print_dataset_analysis(self.analysis)
        
        # Extract detected columns
        self.input_columns = self.analysis['input_variables']
        self.output_columns = self.analysis['output_variables']
        self.output_groups = self.analysis['output_groups']
        
        print(f"\n✅ Auto-detection complete:")
        print(f"   📥 Input columns: {len(self.input_columns)}")
        print(f"   📤 Output columns: {len(self.output_columns)}")
        print(f"   🏷️  Output groups: {len(self.output_groups)}")
        
        if self.input_columns:
            sample_inputs = self.input_columns[:5]
            print(f"   Sample inputs: {sample_inputs}{'...' if len(self.input_columns) > 5 else ''}")
        if self.output_columns:
            sample_outputs = self.output_columns[:5]
            print(f"   Sample outputs: {sample_outputs}{'...' if len(self.output_columns) > 5 else ''}")
    
    def get_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get statistical summary for the dataset.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to analyze. If None, analyzes all numeric columns.
            
        Returns
        -------
        pd.DataFrame
            Statistical summary (describe() output)
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_from_file() or load_from_dataframe() first.")
        
        print("\n📈 Dataset Statistics:")
        stats = get_dataset_statistics(self.df, columns=columns)
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        if self.df is None:
            return "SurrogateDataset(not loaded)"
        
        return (
            f"SurrogateDataset("
            f"shape={self.df.shape}, "
            f"inputs={len(self.input_columns)}, "
            f"outputs={len(self.output_columns)})"
        )

