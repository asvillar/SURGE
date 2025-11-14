
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from .metrics import mean_squared_error, r2_score
from .model import BaseModelAdapter, MODEL_REGISTRY, FNNEnsemble, GPFLOW_AVAILABLE, PYTORCH_AVAILABLE

# Import visualization functions
try:
    from .visualization import (
        plot_gt_vs_prediction,
        plot_regression_comparison,
        plot_multi_output_comparison,
        plot_model_comparison,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Optional imports for advanced functionality
try:
    import optuna
    from optuna.integration import BoTorchSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model type in the registry."""
    registry_key: Optional[str] = None
    display_name: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)


class SurrogateEngine:
    """
    Unified SURGE surrogate modeling engine.
    
    This class provides a comprehensive interface for surrogate modeling workflows,
    including automated data loading with pattern-based detection, model training,
    evaluation, visualization, and optimization.
    
    Supports both auto-detection and manual specification of inputs/outputs,
    and works with various model types through the model registry.
    
    Examples
    --------
    >>> # Auto-detection from file
    >>> engine = SurrogateEngine()
    >>> engine.load_from_file('data.pkl', auto_detect=True)
    >>> engine.train_test_split()
    >>> engine.standardize_data()
    >>> engine.init_model('random_forest')
    >>> engine.train(0)
    
    >>> # Manual specification
    >>> engine = SurrogateEngine()
    >>> engine.load_df_dataset(df, input_cols, output_cols)
    >>> # ... rest same
    """

    def __init__(self, n_features=None, n_outputs=None, dir_path=None):
        """
        Initialize the SurrogateEngine.
        
        Parameters:
        -----------
        n_features : int, optional
            Number of input features (can be set later via load_data)
        n_outputs : int, optional
            Number of output targets (can be set later via load_data)
        dir_path : str, optional
            Directory path for saving outputs
        """
        print("🚀 Initializing SURGE SurrogateEngine")
        self.F = n_features
        self.T = n_outputs
        self.models: List[BaseModelAdapter] = []
        self.model_types: List[ModelSpec] = []
        self.dir_path = dir_path or ''

        # Model performance tracking
        self.model_performance: List[Dict[str, Any]] = []
        self.ensembles: List[FNNEnsemble] = []
        
        # Model predictions storage (per model index)
        # Format: {model_index: {'train': y_pred_train, 'test': y_pred_test}}
        self.model_predictions: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Global predictions (for backward compatibility, stores last model's predictions)
        self.y_pred_train_val: Optional[np.ndarray] = None
        self.y_pred_test: Optional[np.ndarray] = None
        self.MSE_train_val: Optional[float] = None
        self.R2_train_val: Optional[float] = None
        self.MSE: Optional[float] = None
        self.R2: Optional[float] = None

        # Baseline performance tracking
        self.baseline_r2: Optional[float] = None

        # Model type mapping (legacy integer-based, for backward compatibility)
        self.MODEL_TYPES: Dict[int, ModelSpec] = {
            0: ModelSpec('sklearn.random_forest', 'RandomForestRegressor'),
            1: ModelSpec('sklearn.mlp', 'MLPRegressor'),
            2: ModelSpec('pytorch.mlp', 'PyTorchMLP'),
            3: ModelSpec('gpflow.gpr', 'GPflowGPR'),
        }
        self.MODEL_EXPORT_PREFIX: Dict[str, str] = {
            'sklearn.random_forest': 'RFR',
            'sklearn.mlp': 'MLP',
            'pytorch.mlp': 'PyTorchMLP',
            'gpflow.gpr': 'GPR',
            'gpflow.multi_kernel_gpr': 'GPRMulti',
        }
        
        # Track if model registry keys are used
        self._uses_registry = False

        if n_features is not None and n_outputs is not None:
            print(f"📊 Configured for {n_features} features → {n_outputs} outputs")
        else:
            print("📊 Features and outputs will be set when loading data")

    # ------------------------------------------------------------------
    # Internal helpers for model registry integration
    # ------------------------------------------------------------------

    def _resolve_model_spec(self, model_type: int) -> ModelSpec:
        spec = self.MODEL_TYPES.get(model_type)
        if spec is None:
            available = ", ".join(f"{idx}:{spec.display_name}" for idx, spec in self.MODEL_TYPES.items())
            raise ValueError(f"Unknown model_type {model_type}. Available: {available}")
        return spec

    def _build_performance_stub(self, model_name: str) -> Dict[str, Optional[float]]:
        return {
            'model_type': model_name,
            'R2_train_val': None,
            'R2': None,
            'MSE_train_val': None,
            'MSE': None,
            'training_time': None,
            'average_inference_time_train': None,
            'average_inference_time_test': None,
        }

    def _get_model_spec(self, model_index: int) -> ModelSpec:
        if model_index >= len(self.model_types):
            raise ValueError(f"Model index {model_index} not available")
        return self.model_types[model_index]

    def _get_model_adapter(self, model_index: int) -> BaseModelAdapter:
        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} not available")
        return self.models[model_index]

    def _to_numpy(self, data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, 'values'):
            return data.values
        return np.asarray(data)

    def _ensure_2d(self, data: Any) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _select_training_data(self, adapter: BaseModelAdapter) -> Tuple[np.ndarray, np.ndarray]:
        if adapter.uses_internal_preprocessing:
            X = self._to_numpy(self.X_train_val)
            y = self._to_numpy(self.y_train_val)
        else:
            X = self.x_train_val_sc
            y = self.y_train_val_sc
        
        # sklearn models prefer 1D y for single output (prevents DataConversionWarning)
        # Reshape y to 1D if it's a column vector and we have single output
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        
        return X, y

    def _select_prediction_inputs(self, adapter: BaseModelAdapter, dataset: str) -> np.ndarray:
        if adapter.uses_internal_preprocessing:
            if dataset == 'train':
                return self._to_numpy(self.X_train_val)
            return self._to_numpy(self.X_test)
        if dataset == 'train':
            return self.x_train_val_sc
        return self.x_test_sc

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
        Uses SurrogateDataset for generalized pattern-based detection.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        input_cols : list of str, optional
            Manual specification of input column names
        output_cols : list of str, optional
            Manual specification of output column names
        auto_detect : bool, default=True
            If True, automatically detect inputs/outputs using pattern analysis
        **kwargs
            Additional arguments passed to pandas read function
            
        Returns
        -------
        tuple
            (input_columns, output_columns) lists
            
        Examples
        --------
        >>> engine = SurrogateEngine()
        >>> input_cols, output_cols = engine.load_from_file('data.pkl', auto_detect=True)
        """
        from .dataset import SurrogateDataset
        
        dataset = SurrogateDataset()
        input_cols, output_cols = dataset.load_from_file(
            file_path, input_cols, output_cols, auto_detect, **kwargs
        )
        
        # Load into engine
        self.load_from_dataset(dataset)
        
        return input_cols, output_cols
    
    def load_from_dataset(self, dataset):
        """
        Load dataset from SurrogateDataset instance.
        
        Parameters
        ----------
        dataset : SurrogateDataset
            SurrogateDataset instance with loaded data
            
        Examples
        --------
        >>> from surge import SurrogateDataset, SurrogateEngine
        >>> dataset = SurrogateDataset()
        >>> dataset.load_from_file('data.pkl')
        >>> engine = SurrogateEngine()
        >>> engine.load_from_dataset(dataset)
        """
        if dataset.df is None:
            raise ValueError("Dataset not loaded. Call dataset.load_from_file() first.")
        
        if not dataset.input_columns or not dataset.output_columns:
            raise ValueError("Dataset missing input/output columns. Ensure auto_detect=True or specify manually.")
        
        # Load using existing load_df_dataset method
        self.load_df_dataset(
            dataset.df,
            dataset.input_columns,
            dataset.output_columns
        )
        
        # Store reference to dataset for access to analysis
        self._dataset = dataset
    
    def load_df_dataset(self, df, input_feature_names, output_feature_names):
        """
        Load dataset from pandas DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing features and targets
        input_feature_names : list
            List of input feature column names
        output_feature_names : list
            List of output target column names
        """
        print("📥 Loading dataset...")

        self.df = df
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names

        # Determine main output prefix
        common_prefix = os.path.commonprefix(output_feature_names)
        print(f"📋 Common output prefix: {common_prefix}")
        self.main_output = common_prefix

        # Extract features and targets
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]

        # Set F and T if not provided in constructor
        if self.F is None:
            self.F = len(input_feature_names)
        if self.T is None:
            self.T = len(output_feature_names)

        print(f"✅ Dataset loaded: {self.X.shape[0]} samples")
        print(f"   📥 Input features: {len(input_feature_names)} (F={self.F})")
        print(f"   📤 Output targets: {len(output_feature_names)} (T={self.T})")

    def train_test_split(
        self,
        test_split: float = 0.2,
        random_state: Optional[int] = None
    ):
        """
        Split data into train+validation and test sets.
        
        Parameters:
        -----------
        test_split : float
            Fraction of data to use for testing (default: 0.2)
        """
        print(f"\n🔄 Splitting into test and training data (test_split={test_split})")

        # Split into train+val and test
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=test_split, random_state=random_state, shuffle=True
        )

        print(f"X train + val shape: {self.X_train_val.shape}")
        print(f"y train + val shape: {self.y_train_val.shape}")
        print(f"X test shape: {self.X_test.shape}")
        print(f"y test shape: {self.y_test.shape}")

    def standardize_data(self):
        """Standardize the dataset using training set statistics."""
        print("\n🔧 Standardizing data")

        # Standardize inputs
        self.x_scaler = SKStandardScaler()
        self.x_train_val_sc = self.x_scaler.fit_transform(self.X_train_val)
        self.x_test_sc = self.x_scaler.transform(self.X_test)

        # Standardize outputs
        self.y_scaler = SKStandardScaler()
        self.y_train_val_sc = self.y_scaler.fit_transform(self.y_train_val)
        self.y_test_sc = self.y_scaler.transform(self.y_test)

        print("✅ x_train_sc, x_test_sc, y_train_sc, y_test_sc generated")
        print("\n--- Statistics for x_scaler [all features] ---")
        print("means: ", self.x_scaler.mean_)
        if self.x_scaler.var_ is not None:
            print("standard deviation: ", np.sqrt(self.x_scaler.var_))

        # Show statistics for most varying output
        if self.y_scaler.mean_ is not None and self.y_scaler.var_ is not None:
            idx_max_ytrain_mean = np.argmax(self.y_scaler.mean_)
            print("\n--- Statistics for y_scaler [on axis, max_value] ---")
            print("means: ", self.y_scaler.mean_[[0, idx_max_ytrain_mean]])
            print("standard deviation: ", np.sqrt(self.y_scaler.var_[[0, idx_max_ytrain_mean]]))

    def create_fnn_ensemble(self, n_members: int = 5, **model_kwargs: Any) -> FNNEnsemble:
        """Build a feed-forward neural network ensemble for UQ."""
        ensemble_models = []
        for member_idx in range(n_members):
            params = dict(model_kwargs)
            params.setdefault('random_state', 42 + member_idx)
            adapter = MODEL_REGISTRY.create('pytorch.mlp', **params)
            ensemble_models.append(adapter)
        ensemble = FNNEnsemble(ensemble_models)
        self.ensembles.append(ensemble)
        return ensemble

    def predict_with_ensemble(self, ensemble_index: int, X: Any) -> Dict[str, Any]:
        """Return mean/std predictions for an ensemble."""
        ensemble = self.ensembles[ensemble_index]
        result = ensemble.predict(X)
        return {
            'mean': result.mean,
            'std': result.std,
            'members': result.members,
        }

    def init_model(
        self,
        model_type: Union[int, str],
        **model_kwargs: Any
    ):
        """
        Initialize a model of the specified type.
        
        Supports both legacy integer-based model types and registry keys.
        
        Parameters
        ----------
        model_type : int or str
            Model type:
            - int: Legacy model type (0=RFR, 1=MLP, 2=PyTorchMLP, 3=GPflowGPR)
            - str: Model registry key (e.g., 'random_forest', 'sklearn.random_forest', 'pytorch.mlp_model')
        **model_kwargs
            Additional keyword arguments for model initialization.
        
        Examples
        --------
        >>> # Using legacy integer-based model type
        >>> trainer.init_model(0, n_estimators=100, max_depth=10)
        >>> 
        >>> # Using model registry key
        >>> trainer.init_model('random_forest', n_estimators=100, max_depth=10)
        >>> trainer.init_model('pytorch.mlp_model', hidden_layers=[50, 25])
        """
        # Handle string-based model types (registry keys)
        if isinstance(model_type, str):
            # Use model registry directly
            adapter = MODEL_REGISTRY.create(model_type, **model_kwargs)
            self.models.append(adapter)
            
            # Create spec for registry-based model
            registry_model = MODEL_REGISTRY.get(model_type)
            spec = ModelSpec(
                registry_key=model_type,
                display_name=adapter.__class__.__name__.replace('Adapter', '').replace('Model', ''),
                default_params=model_kwargs
            )
            self.model_types.append(spec)
            self._uses_registry = True
            
            model_index = len(self.models) - 1
            print(f"\n🎯 Initializing Model {model_index}: {spec.display_name} (from registry)")
        else:
            # Legacy integer-based model type
            spec = self._resolve_model_spec(model_type)
            print(f"\n🎯 Initializing Model {model_type}: {spec.display_name}")

            params = dict(spec.default_params)
            params.update(model_kwargs)

            adapter = MODEL_REGISTRY.create(spec.registry_key, **params)
            self.models.append(adapter)
            self.model_types.append(spec)
        
        performance = self._build_performance_stub(spec.display_name)
        self.model_performance.append(performance)
        print(f"✅ {spec.display_name} initialized")

    def train(self, model_index: int):
        """Train the specified model."""
        adapter = self._get_model_adapter(model_index)
        spec = self._get_model_spec(model_index)

        print(f'\n🎯 Training Model {model_index}: {spec.display_name}')

        start_time = time.time()
        X_train, y_train = self._select_training_data(adapter)
        adapter.fit(X_train, y_train)
        training_time = time.time() - start_time

        self.model_performance[model_index]['training_time'] = training_time
        print(f"⏱️ Elapsed time: {training_time:.2f} seconds")

    def _train_gpr(self, model_index):
        """Train Gaussian Process Regressor"""
        print("🔄 GPR training - to be implemented")

    def predict_output(self, model_index: int):
        """Generate predictions and calculate performance metrics."""
        adapter = self._get_model_adapter(model_index)
        spec = self._get_model_spec(model_index)
        print(f'\n🎯 Predicting outputs - Model {model_index} ({spec.display_name})')

        # Training set predictions
        print('\n--- Training Set Results ---')
        X_train = self._select_prediction_inputs(adapter, 'train')
        start_time = time.time()
        y_pred_train = adapter.predict(X_train)
        t_pred_train_val = time.time() - start_time

        y_pred_train = self._ensure_2d(y_pred_train)
        if adapter.handles_output_scaling:
            y_pred_train_inv = y_pred_train
        else:
            y_pred_train_inv = self.y_scaler.inverse_transform(y_pred_train)

        avg_inference_time_train = t_pred_train_val / X_train.shape[0]
        print(f' t_I(avg) = {avg_inference_time_train:.6f} seconds per sample')

        y_true_train = self._to_numpy(self.y_train_val)
        train_mse = mean_squared_error(y_true_train, y_pred_train_inv)
        train_r2 = r2_score(y_true_train, y_pred_train_inv)
        
        # Store globally for backward compatibility
        self.MSE_train_val = train_mse
        self.R2_train_val = train_r2

        print(f' MSE_train_val = {train_mse:.6f}')
        print(f' R2_train_val = {train_r2:.4f}')

        # Test set predictions
        print('\n--- Testing Set Results ---')
        X_test = self._select_prediction_inputs(adapter, 'test')
        start_time = time.time()
        y_pred_test = adapter.predict(X_test)
        t_pred_test = time.time() - start_time

        y_pred_test = self._ensure_2d(y_pred_test)
        if adapter.handles_output_scaling:
            y_pred_test_inv = y_pred_test
        else:
            y_pred_test_inv = self.y_scaler.inverse_transform(y_pred_test)

        # Store predictions per model
        self.model_predictions[model_index] = {
            'train': y_pred_train_inv,
            'test': y_pred_test_inv
        }
        
        # Store globally for backward compatibility
        self.y_pred_train_val = y_pred_train_inv
        self.y_pred_test = y_pred_test_inv

        avg_inference_time_test = t_pred_test / X_test.shape[0]
        print(f' t_I(avg) = {avg_inference_time_test:.6f} seconds per sample')

        y_true_test = self._to_numpy(self.y_test)
        test_mse = mean_squared_error(y_true_test, y_pred_test_inv)
        test_r2 = r2_score(y_true_test, y_pred_test_inv)
        
        # Store globally for backward compatibility
        self.MSE = test_mse
        self.R2 = test_r2

        print(f' MSE = {test_mse:.6f}')
        print(f' R2 = {test_r2:.4f}')

        self.model_performance[model_index].update({
            'R2_train_val': train_r2,
            'R2': test_r2,
            'MSE_train_val': train_mse,
            'MSE': test_mse,
            'average_inference_time_train': avg_inference_time_train,
            'average_inference_time_test': avg_inference_time_test
        })

    def plot_regression_results(
        self,
        model_index: int = 0,
        output_index: Optional[int] = None,
        dataset: str = 'both',
        bins: int = 100,
        cmap: str = 'plasma_r',
        save_path: Optional[str] = None,
        dpi: int = 150,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        units: Optional[str] = None,
        **kwargs
    ):
        """
        Plot Ground Truth vs Prediction with density heatmap.
        
        Creates publication-quality plots showing model performance using
        density scatter plots with histograms in 'plasma' colormap.
        
        Parameters
        ----------
        model_index : int, default=0
            Index of the model to plot.
        output_index : int, optional
            Index of output to plot (for multi-output models).
            If None and multi-output, plots first output.
        dataset : str, default='both'
            Which dataset to plot: 'train', 'test', or 'both'.
            If 'both', creates side-by-side comparison.
        bins : int, default=50
            Number of bins for the 2D histogram.
        cmap : str, default='plasma_r'
            Colormap for the density heatmap. Defaults to inverse plasma ('plasma_r').
            Use 'plasma' for original direction, or any other matplotlib colormap.
        save_path : str, optional
            Path to save the figure. If None, figure is displayed.
        dpi : int, default=150
            Resolution for saved figures.
        **kwargs
            Additional arguments passed to plot_gt_vs_prediction.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : matplotlib.axes.Axes or np.ndarray
            The axes object(s).
        results : dict
            Dictionary with R² scores.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "Visualization functions not available. "
                "Ensure matplotlib is installed: pip install matplotlib"
            )
        
        # Check if predictions exist
        if not hasattr(self, 'y_pred_train_val') or not hasattr(self, 'y_pred_test'):
            raise ValueError(
                "Predictions not available. Call predict_output() first."
            )
        
        adapter = self._get_model_adapter(model_index)
        spec = self._get_model_spec(model_index)
        model_name = spec.display_name
        
        # Get true and predicted values
        y_true_train = self._to_numpy(self.y_train_val)
        y_pred_train = self.y_pred_train_val
        y_true_test = self._to_numpy(self.y_test)
        y_pred_test = self.y_pred_test
        
        # Handle multi-output
        if output_index is not None:
            if y_true_train.ndim > 1:
                y_true_train = y_true_train[:, output_index]
                y_pred_train = y_pred_train[:, output_index]
                y_true_test = y_true_test[:, output_index]
                y_pred_test = y_pred_test[:, output_index]
            else:
                print("⚠️ Single output model, ignoring output_index")
        
        # Get output name and units from dataset if available
        output_name = ""
        if hasattr(self, 'output_feature_names') and self.output_feature_names:
            if output_index is not None and output_index < len(self.output_feature_names):
                output_name = self.output_feature_names[output_index]
            elif y_true_train.ndim == 1:
                output_name = self.output_feature_names[0] if self.output_feature_names else ""
        
        # Set default labels if not provided - use dynamic names from dataset
        if xlabel is None:
            xlabel = 'Ground Truth'
        if ylabel is None:
            ylabel = f'Prediction ({model_name})' if model_name else 'Prediction'
        
        # Default to arbitrary units if not provided
        # User can pass units parameter to override (e.g., "a.u.")
        if units is None:
            units = 'a.u.'  # Default to arbitrary units
        
        # Plot based on dataset selection
        if dataset == 'both':
            # Side-by-side comparison
            fig, axes, results = plot_regression_comparison(
                y_true_train, y_pred_train,
                y_true_test, y_pred_test,
                model_name=model_name,
                output_name=output_name,
                bins=bins,
                cmap=cmap,
                save_path=save_path,
                dpi=dpi,
                xlabel=xlabel,
                ylabel=ylabel,
                units=units,
            )
            return fig, axes, results
        
        elif dataset == 'train':
            title = f"Training Set - {model_name}"
            if output_name:
                title = f"{output_name} - Training Set"
            
            fig, ax, r2 = plot_gt_vs_prediction(
                y_true_train, y_pred_train,
                dataset_name="Training Set",
                title=title,
                bins=bins,
                cmap=cmap,
                save_path=save_path,
                dpi=dpi,
                xlabel=xlabel,
                ylabel=ylabel,
                units=units,
                **kwargs
            )
            results = {'r2_train': r2}
            return fig, ax, results
        
        elif dataset == 'test':
            title = f"Test Set - {model_name}"
            if output_name:
                title = f"{output_name} - Test Set"
            
            fig, ax, r2 = plot_gt_vs_prediction(
                y_true_test, y_pred_test,
                dataset_name="Test Set",
                title=title,
                bins=bins,
                cmap=cmap,
                save_path=save_path,
                dpi=dpi,
                xlabel=xlabel,
                ylabel=ylabel,
                units=units,
                **kwargs
            )
            results = {'r2_test': r2}
            return fig, ax, results
        
        else:
            raise ValueError(f"Invalid dataset: {dataset}. Must be 'train', 'test', or 'both'")

    def plot_all_outputs(
        self,
        model_index: int = 0,
        max_outputs: int = 4,
        bins: int = 100,
        cmap: str = 'plasma_r',
        save_path: Optional[str] = None,
        dpi: int = 150,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        units: Optional[str] = None,
    ):
        """
        Plot Ground Truth vs Prediction for all outputs (multi-output models).
        
        Creates a grid of plots showing performance for each output variable.
        
        Parameters
        ----------
        model_index : int, default=0
            Index of the model to plot.
        max_outputs : int, default=4
            Maximum number of outputs to plot.
        bins : int, default=50
            Number of bins for the 2D histogram.
        cmap : str, default='plasma_r'
            Colormap for the density heatmap. Defaults to inverse plasma ('plasma_r').
            Use 'plasma' for original direction, or any other matplotlib colormap.
        save_path : str, optional
            Path to save the figure.
        dpi : int, default=150
            Resolution for saved figures.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : np.ndarray
            Array of axes objects.
        results : dict
            Dictionary with R² scores for each output.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "Visualization functions not available. "
                "Ensure matplotlib is installed: pip install matplotlib"
            )
        
        # Check if predictions exist
        if not hasattr(self, 'y_pred_train_val') or not hasattr(self, 'y_pred_test'):
            raise ValueError(
                "Predictions not available. Call predict_output() first."
            )
        
        adapter = self._get_model_adapter(model_index)
        spec = self._get_model_spec(model_index)
        model_name = spec.display_name
        
        # Get true and predicted values
        y_true_train = self._to_numpy(self.y_train_val)
        y_pred_train = self.y_pred_train_val
        y_true_test = self._to_numpy(self.y_test)
        y_pred_test = self.y_pred_test
        
        # Get output names from dataset
        output_names = []
        if hasattr(self, 'output_feature_names') and self.output_feature_names:
            output_names = self.output_feature_names
        else:
            n_outputs = y_true_train.shape[1] if y_true_train.ndim > 1 else 1
            output_names = [f"Output {i+1}" for i in range(n_outputs)]
        
        # Set default labels if not provided
        if xlabel is None:
            xlabel = 'Ground Truth'
        if ylabel is None:
            ylabel = f'Prediction ({model_name})' if model_name else 'Prediction'
        
        # Default to arbitrary units if not provided
        if units is None:
            units = 'a.u.'  # Default to arbitrary units
        
        # Create multi-output comparison plot
        fig, axes, results = plot_multi_output_comparison(
            y_true_train, y_pred_train,
            y_true_test, y_pred_test,
            output_names=output_names,
            model_name=model_name,
            bins=bins,
            cmap=cmap,
            save_path=save_path,
            dpi=dpi,
            max_outputs=max_outputs,
            xlabel=xlabel,
            ylabel=ylabel,
            units=units,
        )
        
        return fig, axes, results

    def compare_models(
        self,
        model_indices: List[int],
        output_index: Optional[int] = None,
        dataset: str = 'test',
        bins: int = 100,
        cmap: str = 'plasma_r',
        save_path: Optional[str] = None,
        dpi: int = 150,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        units: Optional[str] = None,
    ):
        """
        Compare multiple models side-by-side.
        
        Creates comparison plots showing Ground Truth vs Prediction for multiple models,
        allowing direct visual comparison of model performance.
        
        Parameters
        ----------
        model_indices : List[int]
            List of model indices to compare (e.g., [0, 1, 2]).
        output_index : int, optional
            Index of output to plot. If None, uses first output.
        dataset : str, default='test'
            Dataset to use for comparison ('train' or 'test').
        bins : int, default=100
            Number of bins for the 2D histogram.
        cmap : str, default='plasma_r'
            Colormap for the density heatmap.
        save_path : str, optional
            Path to save the figure.
        dpi : int, default=150
            Resolution for saved figures.
        xlabel : str, optional
            X-axis label. Defaults to "Ground Truth [units]".
        ylabel : str, optional
            Y-axis label. Defaults to "Prediction [units]".
        units : str, optional
            Units for the output variable. Defaults to "a.u." if None.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : np.ndarray
            Array of axes objects.
        results : dict
            Dictionary with R² scores for each model.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if not model_indices:
            raise ValueError("model_indices cannot be empty")
        
        # Validate model indices
        for idx in model_indices:
            if idx >= len(self.models):
                raise ValueError(f"Model index {idx} out of range. Available: 0-{len(self.models)-1}")
            if idx not in self.model_predictions:
                raise ValueError(f"Model {idx} has not been trained. Call predict_output({idx}) first.")
        
        # Get predictions for each model
        predictions = {}
        model_names = []
        
        for idx in model_indices:
            spec = self._get_model_spec(idx)
            model_name = spec.display_name
            
            # Get predictions
            preds = self.model_predictions[idx][dataset]
            
            # Extract output if needed
            if output_index is not None:
                if preds.ndim > 1:
                    preds = preds[:, output_index]
                else:
                    preds = preds
            
            predictions[model_name] = preds
            model_names.append(model_name)
        
        # Get ground truth
        if dataset == 'train':
            y_true = self._to_numpy(self.y_train_val)
        else:
            y_true = self._to_numpy(self.y_test)
        
        # Extract output if needed
        if output_index is not None:
            if y_true.ndim > 1:
                y_true = y_true[:, output_index]
            else:
                y_true = y_true
        
        # Get output name
        output_name = ""
        if hasattr(self, 'output_feature_names') and self.output_feature_names:
            if output_index is not None and output_index < len(self.output_feature_names):
                output_name = self.output_feature_names[output_index]
        
        # Set default labels
        if xlabel is None:
            xlabel = 'Ground Truth'
        if units is None:
            units = 'a.u.'
        
        # Create comparison plot
        fig, axes, results = plot_model_comparison(
            y_true=y_true,
            predictions=predictions,
            dataset_name=f"{dataset.capitalize()} Set",
            output_name=output_name,
            bins=bins,
            cmap=cmap,
            save_path=save_path,
            dpi=dpi,
            xlabel=xlabel,
            ylabel=ylabel,
            units=units,
        )
        
        return fig, axes, results

    def plot_profiles(
        self,
        rho: np.ndarray,
        case_indices: List[int],
        case_labels: Optional[List[str]] = None,
        model_indices: Optional[List[int]] = None,
        output_indices: Optional[List[int]] = None,
        dataset: str = 'test',
        input_params_list: Optional[List[dict]] = None,
        xlabel: str = "ρ [-]",
        units: str = "",
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        Plot profile comparisons for specific cases (similar to paper figure).
        
        Creates a grid of plots showing profiles (y vs ρ) comparing ground truth
        with predictions from multiple models for selected cases.
        
        Parameters
        ----------
        rho : np.ndarray
            Radial coordinate (spatial coordinate). Shape: (n_points,)
            This should match the length of each profile.
        case_indices : List[int]
            List of sample indices to plot (e.g., [3676, 8197, 12027])
        case_labels : List[str], optional
            Labels for each case (e.g., ["Good", "Average", "Poor"]).
            If None, uses case indices.
        model_indices : List[int], optional
            List of model indices to compare. If None, uses all trained models.
        output_indices : List[int], optional
            List of output indices to plot. If None, uses all outputs.
        dataset : str, default='test'
            Dataset to use ('train' or 'test').
        input_params_list : List[dict], optional
            List of input parameter dictionaries, one per case.
            Each dict can contain keys like 'ne0', 'Te0', 'Nφ', 'α', etc.
        xlabel : str, default="ρ [-]"
            X-axis label.
        units : str, default=""
            Units for outputs (e.g., "a.u."). Defaults to "a.u." if None.
        figsize : tuple, optional
            Figure size. Auto-calculated if None.
        save_path : str, optional
            Path to save the figure.
        dpi : int, default=150
            Resolution for saved figures.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : np.ndarray
            Array of axes objects.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        # Validate case indices
        if dataset == 'train':
            n_samples = len(self.y_train_val) if hasattr(self, 'y_train_val') else 0
        else:
            n_samples = len(self.y_test) if hasattr(self, 'y_test') else 0
        
        for case_idx in case_indices:
            if case_idx >= n_samples:
                raise ValueError(f"Case index {case_idx} out of range. Available: 0-{n_samples-1}")
        
        # Get model indices
        if model_indices is None:
            model_indices = [idx for idx in range(len(self.models)) if idx in self.model_predictions]
        if not model_indices:
            raise ValueError("No models available. Train models first.")
        
        # Get output indices
        if output_indices is None:
            output_indices = list(range(self.T))
        
        # Get case labels
        if case_labels is None:
            case_labels = [f"Case {idx}" for idx in case_indices]
        
        # Get output names
        if hasattr(self, 'output_feature_names') and self.output_feature_names:
            output_names = [self.output_feature_names[i] for i in output_indices]
        else:
            output_names = [f"Output {i+1}" for i in output_indices]
        
        # Get ground truth profiles for selected cases
        if dataset == 'train':
            y_true = self._to_numpy(self.y_train_val)
        else:
            y_true = self._to_numpy(self.y_test)
        
        # Extract profiles for selected cases and outputs
        # Shape: (n_cases, n_outputs, n_points)
        # Assuming outputs are profiles (vectors), need to reshape
        y_true_profiles = []
        for case_idx in case_indices:
            case_true = y_true[case_idx]
            # If outputs are vectors, they should already be shaped correctly
            # Otherwise, we treat each output as a scalar (single point)
            if case_true.ndim == 1:
                # Multiple scalar outputs - convert to profiles if needed
                # For now, assume we need to reshape or repeat
                # This depends on data structure - may need adjustment
                case_profiles = []
                for out_idx in output_indices:
                    if len(case_true) > out_idx:
                        # Treat as scalar - create constant profile
                        profile = np.full_like(rho, case_true[out_idx])
                    else:
                        profile = np.full_like(rho, case_true[out_idx] if out_idx < len(case_true) else 0)
                    case_profiles.append(profile)
                y_true_profiles.append(np.array(case_profiles))
            else:
                # Already a profile
                case_profiles = [case_true[i] for i in output_indices]
                y_true_profiles.append(np.array(case_profiles))
        
        y_true_profiles = np.array(y_true_profiles)
        
        # Get predictions for each model
        y_pred_profiles = {}
        mse_values_list = []
        
        for model_idx in model_indices:
            spec = self._get_model_spec(model_idx)
            model_name = spec.display_name
            
            # Get predictions
            preds = self.model_predictions[model_idx][dataset]
            
            # Extract profiles for selected cases and outputs
            model_preds = []
            case_mse = {}
            
            for case_idx in case_indices:
                case_pred = preds[case_idx]
                if case_pred.ndim == 0:
                    # Scalar - create constant profile
                    case_profiles = [np.full_like(rho, case_pred) for _ in output_indices]
                elif case_pred.ndim == 1:
                    # Vector - treat as scalar outputs
                    case_profiles = []
                    for out_idx in output_indices:
                        if len(case_pred) > out_idx:
                            profile = np.full_like(rho, case_pred[out_idx])
                        else:
                            profile = np.full_like(rho, case_pred[out_idx] if out_idx < len(case_pred) else 0)
                        case_profiles.append(profile)
                else:
                    # Already profiles
                    case_profiles = [case_pred[i] for i in output_indices]
                
                model_preds.append(np.array(case_profiles))
                
                # Calculate MSE for this case
                case_true = y_true_profiles[case_indices.index(case_idx)]
                mse = mean_squared_error(case_true.flatten(), np.array(case_profiles).flatten())
                case_mse[model_name] = mse
            
            y_pred_profiles[model_name] = np.array(model_preds)
            mse_values_list.append(case_mse)
        
        # Combine MSE values across models
        combined_mse_list = []
        for case_idx in range(len(case_indices)):
            combined_mse = {}
            for mse_dict in mse_values_list:
                combined_mse.update(mse_dict)
            combined_mse_list.append(combined_mse)
        
        # Default units
        if not units:
            units = 'a.u.'
        
        # Create profile grid plot
        fig, axes = plot_profile_grid(
            rho=rho,
            y_true_profiles=y_true_profiles,
            y_pred_profiles=y_pred_profiles,
            case_indices=case_indices,
            case_labels=case_labels,
            output_names=output_names,
            input_params_list=input_params_list,
            mse_values_list=combined_mse_list if combined_mse_list else None,
            units=units,
            xlabel=xlabel,
            ylabel_prefix="",
            figsize=figsize,
            save_path=save_path,
            dpi=dpi,
        )
        
        return fig, axes

    def list_available_models(self) -> Dict[str, Any]:
        """
        List all available models from the registry.
        
        Returns
        -------
        dict
            Dictionary mapping model keys to their adapter class names.
        
        Examples
        --------
        >>> trainer.list_available_models()
        {'random_forest': 'RandomForestModel', 'mlp': 'MLPModel', ...}
        """
        return MODEL_REGISTRY.list_models()
    
    def get_model_summary(self, model_index: int) -> Dict[str, Any]:
        """
        Get a summary of a trained model including performance metrics.
        
        Parameters
        ----------
        model_index : int
            Index of the model to summarize.
        
        Returns
        -------
        dict
            Dictionary containing model information and performance metrics.
        
        Examples
        --------
        >>> summary = trainer.get_model_summary(0)
        >>> print(f"R² Score: {summary['performance']['R2']:.4f}")
        """
        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} not available")
        
        spec = self._get_model_spec(model_index)
        perf = self.model_performance[model_index]
        
        return {
            'model_index': model_index,
            'model_name': spec.display_name,
            'registry_key': spec.registry_key,
            'performance': perf,
            'has_predictions': model_index in self.model_predictions,
        }
    
    def compare_all_models(
        self,
        output_index: Optional[int] = None,
        dataset: str = 'test',
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Compare all trained models side-by-side.
        
        Convenience method that compares all models that have been trained.
        
        Parameters
        ----------
        output_index : int, optional
            Index of output to compare. If None and T=1, uses 0.
        dataset : str, default='test'
            Dataset to use ('train' or 'test').
        save_path : str, optional
            Path to save comparison plot.
        **kwargs
            Additional arguments passed to compare_models().
        
        Returns
        -------
        fig, axes
            Figure and axes objects from the comparison plot.
        """
        # Get all model indices that have predictions
        model_indices = [idx for idx in range(len(self.models)) if idx in self.model_predictions]
        
        if not model_indices:
            raise ValueError("No trained models available. Train models first.")
        
        return self.compare_models(
            model_indices=model_indices,
            output_index=output_index,
            dataset=dataset,
            save_path=save_path,
            **kwargs
        )

    def optimize_solver(self, n_iter=50, model_idx=0, search_spaces=None):
        """
        Optimize hyperparameters using Bayesian Search.
        
        Parameters:
        -----------
        n_iter : int
            Number of optimization iterations
        model_idx : int
            Index of model to optimize
        search_spaces : dict, optional
            Custom search space for hyperparameters
        """
        if not SKOPT_AVAILABLE:
            print("⚠️ scikit-optimize not available. Please install scikit-optimize.")
            return

        adapter = self._get_model_adapter(model_idx)
        spec = self._get_model_spec(model_idx)

        if spec.registry_key == 'sklearn.random_forest':
            print('🔍 Optimizing Random Forest Regressor Model')
            print('- Method: Bayesian Search Cross Validated')

            estimator = adapter.get_estimator()

            if search_spaces is None:
                search_spaces = {
                    'n_estimators': (50, 500),
                    'max_depth': (5, 50),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': [1.0, 'log2', 'sqrt'],
                }

            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=search_spaces,
                n_iter=n_iter,
                scoring='neg_mean_squared_error',
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=2
            )

            print(f'🚀 Running optimization for n_iter = {n_iter}')
            opt.fit(self.x_train_val_sc, self.y_train_val_sc)

            print("🎯 Best hyperparameters:", opt.best_params_)  # type: ignore
            adapter.update_estimator(opt.best_estimator_)  # type: ignore

            # Generate predictions with optimized model
            self.predict_output(model_idx)

        else:
            print(f"⚠️ Optimization not implemented for model type {spec.display_name}")

    def cross_validate(self, model_idx=0, cv_folds=5):
        """
        Perform k-fold cross-validation with metrics computed on original scale.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to cross-validate
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Cross-validation results with metrics on original scale
        """
        import time

        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error, r2_score

        print(f'\n🔄 Performing {cv_folds}-fold cross-validation')
        print('   Computing MSE and R² on original (unstandardized) scale...')

        adapter = self._get_model_adapter(model_idx)
        spec = self._get_model_spec(model_idx)

        if not adapter.supports_sklearn_interface:
            raise ValueError(f"Cross-validation currently requires sklearn-compatible models. '{spec.display_name}' is not supported.")

        base_model = adapter.get_estimator()

        # Initialize KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Storage for results
        fold_mse = []
        fold_r2 = []
        fold_train_time = []
        fold_inference_time = []

        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.x_train_val_sc)):
            print(f'   Fold {fold_idx + 1}/{cv_folds}...', end=' ')

            # Split data for this fold
            X_fold_train = self.x_train_val_sc[train_idx]
            y_fold_train = self.y_train_val_sc[train_idx]
            X_fold_val = self.x_train_val_sc[val_idx]
            y_fold_val = self.y_train_val_sc[val_idx]

            # Clone model for this fold
            fold_model = clone(base_model)

            # Train model and measure training time
            start_time = time.time()
            fold_model.fit(X_fold_train, y_fold_train)
            train_time = time.time() - start_time

            # Make predictions and measure inference time
            start_time = time.time()
            y_pred_sc = fold_model.predict(X_fold_val)
            inference_time = (time.time() - start_time) / len(val_idx)  # Per sample

            # Ensure 2D shapes for sklearn inverse_transform
            if y_pred_sc.ndim == 1:
                y_pred_sc = y_pred_sc.reshape(-1, 1)
            if y_fold_val.ndim == 1:
                y_fold_val = y_fold_val.reshape(-1, 1)

            # Rescale predictions and targets back to original scale
            y_pred_orig = self.y_scaler.inverse_transform(y_pred_sc)
            y_true_orig = self.y_scaler.inverse_transform(y_fold_val)

            # Compute metrics on original scale
            mse = mean_squared_error(y_true_orig, y_pred_orig)
            r2 = r2_score(y_true_orig, y_pred_orig)

            # Store results
            fold_mse.append(mse)
            fold_r2.append(r2)
            fold_train_time.append(train_time)
            fold_inference_time.append(inference_time)

            print(f'MSE={mse:.6f}, R²={r2:.4f}')

        # Convert to numpy arrays
        fold_mse = np.array(fold_mse)
        fold_r2 = np.array(fold_r2)
        fold_train_time = np.array(fold_train_time)
        fold_inference_time = np.array(fold_inference_time)

        # Print summary statistics
        print('\n📊 Cross-validation results (original scale):')
        print(f'   MSE:  {fold_mse.mean():.6f} ± {fold_mse.std():.6f}')
        print(f'   R²:   {fold_r2.mean():.4f} ± {fold_r2.std():.4f}')
        print(f'   Training time:   {fold_train_time.mean():.3f} ± {fold_train_time.std():.3f} sec/fold')
        print(f'   Inference time:  {fold_inference_time.mean()*1000:.2f} ± {fold_inference_time.std()*1000:.2f} ms/sample')

        # Create results dictionary
        results = {
            'mse_folds': fold_mse,
            'r2_folds': fold_r2,
            'train_time_folds': fold_train_time,
            'inference_time_folds': fold_inference_time,
            'mse_mean': fold_mse.mean(),
            'mse_std': fold_mse.std(),
            'r2_mean': fold_r2.mean(),
            'r2_std': fold_r2.std(),
            'train_time_mean': fold_train_time.mean(),
            'train_time_std': fold_train_time.std(),
            'inference_time_mean': fold_inference_time.mean(),
            'inference_time_std': fold_inference_time.std()
        }

        return results

    def save_prediction(self, output_fmt='pickle', output_dir=None, model_idx=0):
        """
        Save predictions to file.
        
        Parameters:
        -----------
        output_fmt : str
            Output format ('pickle')
        output_dir : str
            Output directory path
        model_idx : int
            Model index to save predictions for
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        spec = self._get_model_spec(model_idx)
        model_type_str = self.MODEL_EXPORT_PREFIX.get(spec.registry_key, spec.display_name.replace(' ', ''))

        if output_fmt == 'pickle':
            # Create prediction dataframe
            columns_out = [f'{model_type_str}_{col}' for col in self.output_feature_names]

            # Initialize prediction array
            y_pred_full = np.zeros(self.y.shape)

            # Create dataframe with predictions
            df_y_pred = pd.DataFrame(data=y_pred_full, columns=columns_out, dtype='float64')
            df_complete = pd.concat([self.df, df_y_pred], axis=1)

            # Fill in predictions
            df_complete.loc[self.X_train_val.index, columns_out] = self.y_pred_train_val
            df_complete.loc[self.X_test.index, columns_out] = self.y_pred_test

            # Determine output path
            if self.dir_path:
                if output_dir:
                    path_out_folder = os.path.join(self.dir_path, output_dir)
                else:
                    path_out_folder = os.path.join(self.dir_path, 'output_prediction')
            else:
                if output_dir:
                    path_out_folder = output_dir
                else:
                    path_out_folder = os.path.join(os.getenv('HOME', '.'), 'Downloads', 'Model_prediction')

        if not os.path.exists(path_out_folder):
            os.makedirs(path_out_folder)

        path_out_file = os.path.join(path_out_folder, f'{self.main_output}{model_type_str}.pkl')
        print(f"💾 Saving prediction to: {path_out_file}")

        df_complete.to_pickle(path_out_file)
        print('✅ Prediction saved')

    def save_results(
        self,
        model_index: int = 0,
        cv_results: Optional[Dict[str, Any]] = None,
        filepath: Optional[str] = None,
        include_dataset_info: bool = True,
    ) -> str:
        """
        Save model results to JSON file.
        
        Saves comprehensive results including:
        - Model type
        - Cross-validation results (if provided)
        - Training metrics
        - Testing metrics
        - Dataset information (optional)
        
        Parameters
        ----------
        model_index : int, default=0
            Index of the model to save results for.
        cv_results : dict, optional
            Cross-validation results from cross_validate(). If None, skipped.
        filepath : str, optional
            Path to save JSON file. If None, uses dir_path or current directory.
        include_dataset_info : bool, default=True
            Whether to include dataset information (n_samples, n_features, etc.).
        
        Returns
        -------
        str
            Path to saved file.
        
        Examples
        --------
        >>> trainer = MLTrainer(n_features=5, n_outputs=2)
        >>> # ... train model ...
        >>> cv_results = trainer.cross_validate(model_idx=0)
        >>> trainer.predict_output(0)
        >>> trainer.save_results(0, cv_results=cv_results, filepath='results.json')
        """
        import json
        
        spec = self._get_model_spec(model_index)
        model_type = spec.display_name
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Build results dictionary
        results = {
            'model_type': model_type,
        }
        
        # Add cross-validation results if provided
        if cv_results is not None:
            results['cross_validation'] = convert_numpy(cv_results)
        
        # Add training metrics
        if hasattr(self, 'R2_train_val') and self.R2_train_val is not None:
            training_info = {
                'r2_train': float(self.R2_train_val),
                'mse_train': float(self.MSE_train_val),
            }
            if self.model_performance and model_index < len(self.model_performance):
                perf = self.model_performance[model_index]
                if 'training_time' in perf and perf['training_time'] is not None:
                    training_info['training_time'] = float(perf['training_time'])
                if 'average_inference_time_train' in perf:
                    training_info['inference_time_train'] = float(perf['average_inference_time_train'])
            results['training'] = training_info
        
        # Add testing metrics
        if hasattr(self, 'R2') and self.R2 is not None:
            testing_info = {
                'r2_test': float(self.R2),
                'mse_test': float(self.MSE),
            }
            if self.model_performance and model_index < len(self.model_performance):
                perf = self.model_performance[model_index]
                if 'average_inference_time_test' in perf:
                    testing_info['inference_time_test'] = float(perf['average_inference_time_test'])
            results['testing'] = testing_info
        
        # Add dataset information if requested
        if include_dataset_info:
            dataset_info = {}
            if hasattr(self, 'X') and self.X is not None:
                dataset_info['n_samples'] = int(len(self.X))
            if self.F is not None:
                dataset_info['n_features'] = int(self.F)
            if self.T is not None:
                dataset_info['n_outputs'] = int(self.T)
            if hasattr(self, 'X_train_val') and self.X_train_val is not None:
                dataset_info['train_size'] = int(len(self.X_train_val))
            if hasattr(self, 'X_test') and self.X_test is not None:
                dataset_info['test_size'] = int(len(self.X_test))
            if dataset_info:
                results['dataset_info'] = dataset_info
        
        # Determine output path
        if filepath is None:
            if self.dir_path:
                output_dir = Path(self.dir_path)
            else:
                output_dir = Path.cwd()
            # Generate filename based on model type
            model_type_str = self.MODEL_EXPORT_PREFIX.get(
                spec.registry_key, 
                model_type.replace(' ', '_')
            )
            filepath = output_dir / f'{model_type_str}_results.json'
        else:
            filepath = Path(filepath)
        
        # Ensure output directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
        
        return str(filepath)

    def optimize_with_optuna(self, model_idx=0, n_trials=100, objective_type='mse', sampler_type='default'):
        """
        Optimize hyperparameters using Optuna with different samplers.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to optimize
        n_trials : int
            Number of Optuna trials
        objective_type : str
            Optimization objective ('mse', 'r2')
        sampler_type : str
            Optuna sampler type ('default', 'tpe', 'botorch')
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Please install optuna.")
            return

        import optuna
        print(f"🔍 Optuna imported successfully: {optuna.__version__}")
        adapter = self._get_model_adapter(model_idx)
        spec = self._get_model_spec(model_idx)
        
        # Initialize sampler based on type
        sampler = None
        if sampler_type == 'tpe':
            try:
                sampler = optuna.samplers.TPESampler(seed=42)
                print(f"🔬 Using TPE Sampler")
            except Exception as e:
                print(f"⚠️ TPE sampler initialization failed: {e}")
                sampler = None
        elif sampler_type == 'botorch':
            try:
                from optuna.integration import BoTorchSampler
                sampler = BoTorchSampler()
                print(f"🔬 Using BoTorch Sampler")
            except ImportError:
                print("⚠️ BoTorch not available. Install with: pip install botorch")
                sampler = None
            except Exception as e:
                print(f"⚠️ BoTorch sampler initialization failed: {e}")
                sampler = None
        else:
            print(f"🔬 Using Default Optuna Sampler")
            sampler = None

        if spec.registry_key == 'sklearn.random_forest':
            print(f'🔍 Optimizing Random Forest with Optuna ({n_trials} trials)')

            def objective_rf(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 5, 50)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])

                # Create model with suggested parameters
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, self.x_train_val_sc, self.y_train_val_sc,
                    cv=3, scoring='neg_mean_squared_error', n_jobs=-1
                )

                return -cv_scores.mean()  # Return positive MSE

            # Create study and optimize
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective_rf, n_trials=n_trials)

            print("🎯 Best hyperparameters:", study.best_params)
            print(f"🎯 Best MSE: {study.best_value:.6f}")

            # Update model with best parameters
            best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
            adapter.update_estimator(best_model)
            self.model_performance[model_idx] = self._build_performance_stub(spec.display_name)

            # Train with best parameters
            self.train(model_idx)
            self.predict_output(model_idx)

        elif spec.registry_key == 'pytorch.mlp':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Please install torch.")

            print(f'🔍 Optimizing PyTorch MLP with Optuna ({n_trials} trials)')

            from .pytorch_models import PyTorchMLPModel  # Local import to avoid optional dependency issues

            def objective_mlp(trial):
                num_layers = trial.suggest_int("num_layers", 1, 5)
                hidden_size = trial.suggest_int("hidden_size", 32, 256)
                dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_int("batch_size", 16, 128)
                activation_fn = trial.suggest_categorical("activation_fn", ["relu", "tanh", "leaky_relu"])

                hidden_layers = [hidden_size] * num_layers
                candidate = PyTorchMLPModel(
                    hidden_layers=hidden_layers,
                    dropout_rate=dropout_rate,
                    activation_fn=activation_fn,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    n_epochs=80,
                )

                candidate.fit(self.X_train_val, self.y_train_val)
                preds = candidate.predict(self.X_train_val)
                mse = mean_squared_error(self._to_numpy(self.y_train_val), self._ensure_2d(preds))
                return mse

            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective_mlp, n_trials=n_trials)

            print("🎯 Best hyperparameters:", study.best_params)
            print(f"🎯 Best MSE: {study.best_value:.6f}")

            best_params = study.best_params
            hidden_layers = [best_params['hidden_size']] * best_params['num_layers']
            activation_fn = best_params['activation_fn']

            tuned_adapter = MODEL_REGISTRY.create(
                spec.registry_key,
                hidden_layers=hidden_layers,
                dropout_rate=best_params['dropout_rate'],
                activation_fn=activation_fn,
                learning_rate=best_params['learning_rate'],
                batch_size=best_params['batch_size'],
                n_epochs=200,
            )

            self.models[model_idx] = tuned_adapter
            self.best_mlp_params = best_params
            self.model_performance[model_idx] = self._build_performance_stub(spec.display_name)

            self.train(model_idx)
            self.predict_output(model_idx)

        else:
            print(f"⚠️ Optuna optimization not implemented for model type {spec.display_name}")

    def print_model_summary(self, model_index=None):
        """
        Display comprehensive performance summary for models.
        
        This is a convenience method for printing. Use get_model_summary() to get a dict.
        
        Parameters:
        -----------
        model_index : int, optional
            Index of specific model to summarize. If None, summarizes all models.
        """
        if model_index is not None:
            # Summary for specific model
            if model_index >= len(self.model_performance):
                raise ValueError(f"Model index {model_index} not available")

            perf = self.model_performance[model_index]
            print(f"\n🏆 MODEL {model_index} PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Model Type: {perf['model_type']}")
            print(f"Training R²: {perf['R2_train_val']:.4f}")
            print(f"Test R²: {perf['R2']:.4f}")
            print(f"Training MSE: {perf['MSE_train_val']:.6f}")
            print(f"Test MSE: {perf['MSE']:.6f}")
            print(f"Training Time: {perf['training_time']:.2f} seconds")
            print(f"Avg Inference Time (Train): {perf['average_inference_time_train']:.6f} s/sample")
            print(f"Avg Inference Time (Test): {perf['average_inference_time_test']:.6f} s/sample")
            print("=" * 60)
        else:
            # Summary for all models
            print("\n🏆 ALL MODELS PERFORMANCE SUMMARY")
            print("=" * 80)

            if not self.model_performance:
                print("No models trained yet.")
                return

            for i, perf in enumerate(self.model_performance):
                print(f"Model {i} ({perf['model_type']}):")
                print(f"  Training R²: {perf['R2_train_val']:.4f} | Test R²: {perf['R2']:.4f}")
                print(f"  Training MSE: {perf['MSE_train_val']:.6f} | Test MSE: {perf['MSE']:.6f}")
                print(f"  Training Time: {perf['training_time']:.2f}s | Inference: {perf['average_inference_time_test']:.6f}s/sample")
                print("-" * 60)

            print("=" * 80)

    def export_model_onnx(self, model_idx=0, output_path='model.onnx'):
        """
        Export PyTorch model to ONNX format for deployment.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to export
        output_path : str
            Path to save ONNX model
        """
        if not PYTORCH_AVAILABLE:
            print("⚠️ PyTorch not available for ONNX export")
            return

        adapter = self._get_model_adapter(model_idx)
        spec = self._get_model_spec(model_idx)

        if spec.registry_key == 'pytorch.mlp':
            try:
                import torch.onnx

                estimator = adapter.get_estimator()
                if estimator is None:
                    raise ValueError("PyTorch model not initialized")

                if hasattr(estimator, 'eval'):
                    export_model = estimator
                elif hasattr(estimator, 'model') and hasattr(estimator.model, 'eval'):
                    export_model = estimator.model  # type: ignore[attr-defined]
                else:
                    raise TypeError("Estimator does not expose an eval() method for ONNX export")

                export_model.eval()

                # Create dummy input with correct shape
                if self.F is None:
                    raise ValueError("Number of features (F) not set. Call load_data() first.")
                dummy_input = torch.randn(1, self.F, dtype=torch.float32)

                # Export to ONNX
                torch.onnx.export(
                    export_model,
                    (dummy_input,),
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )

                print(f"✅ Model exported to ONNX: {output_path}")

            except ImportError:
                print("⚠️ ONNX export requires torch.onnx")
            except Exception as e:
                print(f"❌ ONNX export failed: {e}")
        else:
            print("⚠️ ONNX export only available for PyTorch models")

    def run_comprehensive_cv(self, model_idx=0, cv_folds=5, n_repeats=3):
        """
        Run comprehensive cross-validation with multiple repeats.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to cross-validate
        cv_folds : int
            Number of CV folds
        n_repeats : int
            Number of repeated CV runs
        """
        from sklearn.model_selection import RepeatedKFold

        print(f'\n🔄 Comprehensive CV: {n_repeats} × {cv_folds}-fold cross-validation')

        adapter = self._get_model_adapter(model_idx)
        spec = self._get_model_spec(model_idx)

        if not adapter.supports_sklearn_interface:
            raise ValueError(f"Comprehensive CV currently requires sklearn-compatible models. '{spec.display_name}' is not supported.")

        model = adapter.get_estimator()

        # Setup repeated k-fold
        rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(
            model, self.x_train_val_sc, self.y_train_val_sc,
            cv=rkf, scoring='neg_mean_squared_error', n_jobs=-1
        )

        # Convert to positive MSE
        cv_mse = -cv_scores

        print(f'📊 Comprehensive CV results ({len(cv_scores)} total folds):')
        print(f'   Mean MSE: {cv_mse.mean():.6f} ± {cv_mse.std():.6f}')
        print(f'   Min MSE: {cv_mse.min():.6f}')
        print(f'   Max MSE: {cv_mse.max():.6f}')

        return {
            'cv_scores': cv_mse,
            'mean_mse': cv_mse.mean(),
            'std_mse': cv_mse.std(),
            'min_mse': cv_mse.min(),
            'max_mse': cv_mse.max()
        }

    def tune(self, model_index=0, method='bayesian', n_trials=20, search_space=None, 
             monitor_resources=True, plot_resources=True, plot_results=True):
        """
        Memory-efficient hyperparameter tuning with various optimization methods and resource monitoring.
        
        Parameters:
        -----------
        model_index : int
            Index of the model to tune
        method : str
            Tuning method: 'random_mem_eff', 'bayesian_skopt', 'optuna_botorch', 'optuna_tpe'
        n_trials : int
            Number of optimization trials
        search_space : dict, optional
            Custom search space for hyperparameters
        monitor_resources : bool, default=True
            Whether to monitor system resources (RAM, CPU) during tuning
        plot_resources : bool, default=True
            Whether to plot resource usage after tuning completes
        plot_results : bool, default=True
            Whether to plot optimization results showing R² evolution
            
        Returns:
        --------
        dict : Tuning results with best parameters, history, and resource monitoring data
        """

        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} not available")

        spec = self._get_model_spec(model_index)
        if spec.registry_key != 'sklearn.random_forest':  # Only Random Forest for now
            raise ValueError("Hyperparameter tuning currently only supports Random Forest models")

        print(f"🔍 Hyperparameter Tuning: {method.upper()}")
        print(f"Model: RandomForestRegressor | Trials: {n_trials}")
        if monitor_resources:
            print("📊 Resource monitoring: ENABLED")
        print("=" * 60)

        # Check if baseline performance has already been calculated
        if not hasattr(self, 'baseline_r2') or self.baseline_r2 is None:
            print("📊 Calculating baseline performance...")
            baseline_model = RandomForestRegressor(
                n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, random_state=42, n_jobs=-1
            )
            baseline_model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_baseline_sc = baseline_model.predict(self.x_test_sc)
            y_pred_baseline = self.y_scaler.inverse_transform(y_pred_baseline_sc)
            self.baseline_r2 = r2_score(self.y_test, y_pred_baseline)
            print(f"📊 Baseline R²: {self.baseline_r2:.4f}")
        else:
            print(f"📊 Using cached baseline R²: {self.baseline_r2:.4f}")
        print("=" * 60)

        # Initialize resource monitor if requested
        resource_monitor = None
        if monitor_resources:
            try:
                from .utils import ResourceMonitor
                resource_monitor = ResourceMonitor()
                initial_ram, initial_cpu = resource_monitor.update()
                print(f"📊 Initial Resources: RAM: {initial_ram:.1f} MB | CPU: {initial_cpu:.1f}%")
                print("=" * 60)
            except ImportError as e:
                print(f"⚠️ Resource monitoring disabled: {e}")
                monitor_resources = False

        # Default search space for Random Forest
        # max_depth is fixed to None (unlimited) for all optimization methods
        if search_space is None:
            search_space = {
                'n_estimators': (50, 500),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', 1.0]
                # max_depth is excluded from optimization and fixed to None
            }

        # Record start time for total tuning time
        start_time = time.time()

        # Execute the appropriate tuning method
        if method == 'random_mem_eff':
            result = self._tune_random_memory_efficient(model_index, n_trials, search_space, resource_monitor)
        elif method == 'bayesian_skopt':
            result = self._tune_bayesian_skopt(model_index, n_trials, search_space, resource_monitor)
        elif method == 'optuna_botorch':
            result = self._tune_optuna_botorch(model_index, n_trials, search_space, resource_monitor)
        elif method == 'optuna_tpe':
            result = self._tune_optuna_tpe(model_index, n_trials, search_space, resource_monitor)
        else:
            raise ValueError(f"Unknown tuning method: {method}")

        # Calculate total tuning time
        total_tuning_time = time.time() - start_time
        result['total_tuning_time'] = total_tuning_time
        result['baseline_r2'] = self.baseline_r2

        print(f"\n⏱️ Total tuning time: {total_tuning_time:.2f} seconds")
        print(f"📊 Baseline vs Best: {self.baseline_r2:.4f} → {result['best_r2']:.4f} "
              f"(Δ = {result['best_r2'] - self.baseline_r2:+.4f})")

        # Add resource monitoring results and plotting
        if resource_monitor is not None:
            resource_summary = resource_monitor.get_summary()
            result['resource_summary'] = resource_summary
            result['resource_monitor'] = resource_monitor
            
            # Only print if we have valid dictionary data (not the "No data collected" string)
            if isinstance(resource_summary, dict):
                print(f"\n📊 Resource Summary:")
                print(f"   Peak RAM Usage: {resource_summary['peak_ram_mb']:.1f} MB")
                print(f"   Average RAM Usage: {resource_summary['avg_ram_mb']:.1f} MB")
                print(f"   Average CPU Usage: {resource_summary['avg_cpu_percent']:.1f}%")
                print(f"   Total Duration: {resource_summary['duration_sec']:.1f} seconds")
            else:
                print(f"\n📊 Resource Summary: {resource_summary}")
            
            if plot_resources:
                resource_monitor.plot_resources(title=f"Resource Usage - {method.upper()} Optimization")

        # Plot optimization results if requested
        if plot_results and 'r2_evolution' in result:
            self._plot_optimization_results(result, self.baseline_r2)

        return result

    def _tune_random_memory_efficient(self, model_index, n_trials, search_space, resource_monitor=None):
        """Random search with memory cleanup and resource monitoring"""
        import gc

        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        # Initialize tracking
        results = {
            'method': 'random_mem_eff',
            'iterations': [],
            'r2_scores': [],
            'r2_evolution': [],  # Track R² evolution over iterations
            'best_r2': -np.inf,
            'best_params': None,
            'hyperparams_history': []
        }

        np.random.seed(42)
        for i in range(n_trials):
            # Monitor resources before training
            if resource_monitor:
                pre_ram, pre_cpu = resource_monitor.update()
            
            # Sample random hyperparameters with proper types
            params = {
                'n_estimators': int(np.random.randint(search_space['n_estimators'][0], search_space['n_estimators'][1] + 1)),
                'max_depth': None,  # Fixed to None (unlimited depth)
                'min_samples_split': int(np.random.randint(search_space['min_samples_split'][0], search_space['min_samples_split'][1] + 1)),
                'min_samples_leaf': int(np.random.randint(search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1] + 1)),
                'max_features': search_space['max_features'][np.random.randint(0, len(search_space['max_features']))],
                'random_state': 42,
                'n_jobs': -1
            }

            # Train and evaluate
            model = RandomForestRegressor(**params)
            model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_sc = model.predict(self.x_test_sc)
            y_pred = self.y_scaler.inverse_transform(y_pred_sc)
            r2 = r2_score(self.y_test, y_pred)

            # Monitor resources after training
            if resource_monitor:
                post_ram, post_cpu = resource_monitor.update()

            # Track results
            results['iterations'].append(i + 1)
            results['r2_scores'].append(r2)
            results['r2_evolution'].append(max(results['r2_scores']))  # Best R² so far
            results['hyperparams_history'].append(params.copy())

            if r2 > results['best_r2']:
                results['best_r2'] = r2
                results['best_params'] = params.copy()

            # Enhanced progress report with resource info
            if resource_monitor:
                print(f"Trial {i+1:2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f} | "
                      f"RAM: {pre_ram:.0f}→{post_ram:.0f}MB")
            else:
                print(f"Trial {i+1:2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f}")

            # Memory cleanup
            del model
            gc.collect()
            
            # Monitor resources after cleanup
            if resource_monitor:
                resource_monitor.update()

        return results

    def _tune_bayesian_skopt(self, model_index, n_trials, search_space, resource_monitor=None):
        """Bayesian optimization using scikit-optimize with resource monitoring"""
        try:
            import gc

            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
            from skopt import BayesSearchCV
        except ImportError:
            raise ImportError("scikit-optimize not available. Install with: pip install scikit-optimize")

        # Monitor resources at start
        if resource_monitor:
            resource_monitor.update()

        # Convert search space for skopt (max_depth excluded from optimization)
        skopt_space = {}
        for key, value in search_space.items():
            if key == 'max_features':
                skopt_space[key] = value  # Categorical
            elif isinstance(value, tuple):
                skopt_space[key] = value  # Continuous/Integer ranges
            else:
                skopt_space[key] = value  # Categorical

        # Initialize base model with max_depth=None (fixed, not optimized)
        base_model = RandomForestRegressor(max_depth=None, random_state=42, n_jobs=-1)

        # Bayesian search
        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=skopt_space,
            n_iter=n_trials,
            scoring='r2',
            cv=1,
            n_jobs=-1,  
            random_state=42,
            verbose=2
        )

        print("🚀 Running Bayesian optimization with scikit-optimize...")
        
        # Monitor during optimization
        if resource_monitor:
            resource_monitor.update()
            
        opt.fit(self.x_train_val_sc, self.y_train_val_sc)
        
        # Monitor after optimization
        if resource_monitor:
            resource_monitor.update()

        # Evaluate best model on test set
        y_pred_sc = opt.predict(self.x_test_sc)
        y_pred = self.y_scaler.inverse_transform(y_pred_sc)
        test_r2 = r2_score(self.y_test, y_pred)

        # Extract iteration data from cv_results_ for consistent format
        cv_results = getattr(opt, 'cv_results_', {})
        iterations = list(range(1, len(cv_results.get('mean_test_score', [])) + 1)) if cv_results else []
        r2_scores = list(cv_results.get('mean_test_score', [])) if cv_results else []
        
        # Calculate r2_evolution (best R² so far at each iteration)
        r2_evolution = []
        if r2_scores:
            best_so_far = -np.inf
            for score in r2_scores:
                if score > best_so_far:
                    best_so_far = score
                r2_evolution.append(best_so_far)

        results = {
            'method': 'bayesian_skopt',
            'iterations': iterations,
            'r2_scores': r2_scores,
            'r2_evolution': r2_evolution,
            'best_r2': test_r2,
            'best_params': dict(getattr(opt, 'best_params_', {})),
            'cv_best_score': getattr(opt, 'best_score_', None),
            'optimization_results': cv_results
        }

        cv_score = getattr(opt, 'best_score_', "N/A")
        print(f"✅ Best CV Score: {cv_score}, Test R²: {test_r2:.4f}")

        return results

    def _tune_optuna_botorch(self, model_index, n_trials, search_space, resource_monitor=None):
        """Optuna optimization with BoTorch sampler and resource monitoring"""
        try:
            import gc

            import numpy as np
            import optuna
            from optuna.integration import BoTorchSampler
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
        except ImportError:
            raise ImportError("Optuna/BoTorch not available. Install with: pip install optuna botorch")

        # Monitor resources at start
        if resource_monitor:
            resource_monitor.update()

        # Track results
        results = {
            'method': 'optuna_botorch',
            'iterations': [],
            'r2_scores': [],
            'r2_evolution': [],
            'best_r2': -np.inf,
            'best_params': None
        }

        def objective(trial):
            # Monitor resources before training
            if resource_monitor:
                pre_ram, pre_cpu = resource_monitor.update()
                
            # Suggest hyperparameters (max_depth fixed to None)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', search_space['n_estimators'][0], search_space['n_estimators'][1]),
                'max_depth': None,  # Fixed to None (unlimited depth)
                'min_samples_split': trial.suggest_int('min_samples_split', search_space['min_samples_split'][0], search_space['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical('max_features', search_space['max_features']),
                'random_state': 42,
                'n_jobs': -1
            }

            # Train and evaluate
            model = RandomForestRegressor(**params)
            model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_sc = model.predict(self.x_test_sc)
            y_pred = self.y_scaler.inverse_transform(y_pred_sc)
            r2 = r2_score(self.y_test, y_pred)

            # Monitor resources after training
            if resource_monitor:
                post_ram, post_cpu = resource_monitor.update()

            # Track progress
            results['iterations'].append(len(results['iterations']) + 1)
            results['r2_scores'].append(r2)
            results['r2_evolution'].append(max(results['r2_scores']))  # Best R² so far
            if r2 > results['best_r2']:
                results['best_r2'] = r2
                results['best_params'] = params.copy()

            # Enhanced progress report with resource info
            if resource_monitor:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f} | "
                      f"RAM: {pre_ram:.0f}→{post_ram:.0f}MB")
            else:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f}")

            # Memory cleanup
            del model
            gc.collect()
            
            # Monitor resources after cleanup
            if resource_monitor:
                resource_monitor.update()

            return r2

        # Create study with BoTorch sampler
        sampler = BoTorchSampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)

        print("🚀 Running Optuna optimization with BoTorch sampler...")
        study.optimize(objective, n_trials=n_trials)

        results['study'] = study
        results['best_r2'] = study.best_value
        results['best_params'] = study.best_params

        print(f"✅ Best R²: {study.best_value:.4f}")

        return results

    def _tune_optuna_tpe(self, model_index, n_trials, search_space, resource_monitor=None):
        """Optuna optimization with TPE sampler and resource monitoring"""
        try:
            import gc

            import numpy as np
            import optuna
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
        except ImportError:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        # Monitor resources at start
        if resource_monitor:
            resource_monitor.update()

        # Track results
        results = {
            'method': 'optuna_tpe',
            'iterations': [],
            'r2_scores': [],
            'r2_evolution': [],
            'best_r2': -np.inf,
            'best_params': None
        }

        def objective(trial):
            # Monitor resources before training
            if resource_monitor:
                pre_ram, pre_cpu = resource_monitor.update()
                
            # Suggest hyperparameters (max_depth fixed to None)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', search_space['n_estimators'][0], search_space['n_estimators'][1]),
                'max_depth': None,  # Fixed to None (unlimited depth)
                'min_samples_split': trial.suggest_int('min_samples_split', search_space['min_samples_split'][0], search_space['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical('max_features', search_space['max_features']),
                'random_state': 42,
                'n_jobs': -1
            }

            # Train and evaluate
            model = RandomForestRegressor(**params)
            model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_sc = model.predict(self.x_test_sc)
            y_pred = self.y_scaler.inverse_transform(y_pred_sc)
            r2 = r2_score(self.y_test, y_pred)

            # Monitor resources after training
            if resource_monitor:
                post_ram, post_cpu = resource_monitor.update()

            # Track progress
            results['iterations'].append(len(results['iterations']) + 1)
            results['r2_scores'].append(r2)
            results['r2_evolution'].append(max(results['r2_scores']))  # Best R² so far
            if r2 > results['best_r2']:
                results['best_r2'] = r2
                results['best_params'] = params.copy()

            # Enhanced progress report with resource info
            if resource_monitor:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f} | "
                      f"RAM: {pre_ram:.0f}→{post_ram:.0f}MB")
            else:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f}")

            # Memory cleanup
            del model
            gc.collect()
            
            # Monitor resources after cleanup
            if resource_monitor:
                resource_monitor.update()

            return r2

        # Create study with TPE sampler (default)
        study = optuna.create_study(direction='maximize')

        print("🚀 Running Optuna optimization with TPE sampler...")
        study.optimize(objective, n_trials=n_trials)

        results['study'] = study
        results['best_r2'] = study.best_value
        results['best_params'] = study.best_params

        print(f"✅ Best R²: {study.best_value:.4f}")

        return results

    def _plot_optimization_results(self, result, baseline_r2):
        """
        Plot optimization results showing R² evolution over iterations.
        
        Parameters:
        -----------
        result : dict
            Tuning results dictionary with r2_evolution, iterations, etc.
        baseline_r2 : float
            Baseline R² score for reference
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            iterations = result.get('iterations', [])
            r2_evolution = result.get('r2_evolution', [])
            method = result.get('method', 'optimization')
            
            if not iterations or not r2_evolution:
                print("⚠️ No iteration data available for plotting")
                return
                
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, r2_evolution, 'b-', linewidth=2, label=f'{method.upper()} - Best R² Evolution')
            plt.axhline(y=baseline_r2, color='r', linestyle='--', linewidth=2, label=f'Baseline R² = {baseline_r2:.4f}')
            
            # Add final improvement annotation
            final_r2 = r2_evolution[-1] if r2_evolution else baseline_r2
            improvement = final_r2 - baseline_r2
            plt.annotate(f'Final: {final_r2:.4f}\nΔ = {improvement:+.4f}', 
                        xy=(len(iterations), final_r2), 
                        xytext=(len(iterations) * 0.7, final_r2 + 0.01),
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black'))
            
            plt.xlabel('Iteration')
            plt.ylabel('R² Score')
            plt.title(f'Hyperparameter Optimization Results - {method.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            print(f"📊 Optimization plot displayed for {method.upper()}")
            
        except ImportError:
            print("⚠️ Matplotlib not available for plotting. Install with: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ Error creating plot: {e}")
