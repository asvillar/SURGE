"""
GPflow-based model implementations for SURGE package.
Integrates GPflow (TensorFlow) Gaussian Process models with the unified SURGE interface.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import gpflow as gpf
    from gpflow.mean_functions import Constant
    from gpflow.utilities import positive, print_summary
    GPFLOW_AVAILABLE = True

    # Set GPflow configuration
    gpf.config.set_default_float(np.float64)
    gpf.config.set_default_summary_fmt("notebook")

except Exception:
    GPFLOW_AVAILABLE = False
    gpf = None
    print("Warning: GPflow not available or incompatible. Install with: pip install gpflow")


class GPflowGPRModel:
    """
    SURGE-compatible wrapper for GPflow Gaussian Process Regression models.
    """

    def __init__(self, kernel_type="matern32", lengthscales=None, variance=1.0,
                 noise_variance=0.1, mean_function=None, optimize=True,
                 maxiter=1000, add_linear=True, linear_variance=None, 
                 kernel_combinations=None, **kwargs):

        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow is required for GPflowGPRModel. Install with: pip install gpflow")

        self.kernel_type = kernel_type.lower() if isinstance(kernel_type, str) else kernel_type
        self.lengthscales = lengthscales
        self.variance = variance
        self.noise_variance = noise_variance
        self.mean_function = mean_function
        self.optimize = optimize
        self.maxiter = maxiter
        # Handle add_linear: support bool, string "true"/"false", etc.
        if isinstance(add_linear, bool):
            self.add_linear = add_linear
        elif isinstance(add_linear, str):
            self.add_linear = add_linear.lower() in ("true", "1", "yes", "y")
        else:
            self.add_linear = bool(add_linear)
        self.linear_variance = linear_variance if linear_variance is not None else 0.1
        # kernel_combinations: list of kernel types to combine (e.g., ["rbf", "matern32"])
        # If provided, will create additive combinations instead of using single kernel_type
        # Handle "null" strings from YAML/categorical choices
        if kernel_combinations == "null" or kernel_combinations == "None":
            self.kernel_combinations = None
        else:
            self.kernel_combinations = kernel_combinations

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _create_kernel(self, input_dim):
        """
        Create GPflow kernel based on specifications.
        Supports single kernels, kernel combinations, and optional Linear kernel.
        """
        # Set default lengthscales if not provided
        if self.lengthscales is None:
            lengthscales = np.ones(input_dim) * 0.75
        else:
            lengthscales = np.array(self.lengthscales)
            if lengthscales.ndim == 0:
                lengthscales = np.ones(input_dim) * float(lengthscales)
            elif lengthscales.shape[0] != input_dim:
                lengthscales = np.ones(input_dim) * lengthscales[0]
        
        # Helper function to create a single kernel
        def _create_single_kernel(k_type, ls, var):
            k_type_lower = k_type.lower() if isinstance(k_type, str) else str(k_type).lower()
            if k_type_lower == "matern32":
                return gpf.kernels.Matern32(lengthscales=ls, variance=var)  # type: ignore
            elif k_type_lower == "matern52" or k_type_lower == "matern53":
                return gpf.kernels.Matern52(lengthscales=ls, variance=var)  # type: ignore
            elif k_type_lower == "matern12":
                return gpf.kernels.Matern12(lengthscales=ls, variance=var)  # type: ignore
            elif k_type_lower == "rbf" or k_type_lower == "squared_exponential":
                return gpf.kernels.SquaredExponential(lengthscales=ls, variance=var)  # type: ignore
            elif k_type_lower == "exponential":
                return gpf.kernels.Exponential(lengthscales=ls, variance=var)  # type: ignore
            elif k_type_lower == "rational_quadratic":
                return gpf.kernels.RationalQuadratic(lengthscales=ls, variance=var, alpha=1.0)  # type: ignore
            elif k_type_lower == "linear":
                return gpf.kernels.Linear(variance=var)  # type: ignore
            else:
                raise ValueError(f"Unknown kernel type: {k_type}")

        # Handle kernel combinations
        # Convert string "null" to None if needed
        kernel_combo = self.kernel_combinations
        if kernel_combo == "null" or kernel_combo == "None":
            kernel_combo = None
        
        if kernel_combo is not None and isinstance(kernel_combo, (list, tuple)) and len(kernel_combo) > 0:
            # Create additive combination of multiple kernels
            kernels_to_combine = []
            for k_type in kernel_combo:
                if k_type.lower() == "linear":
                    # Linear kernel doesn't use lengthscales, uses its own variance
                    kernels_to_combine.append(_create_single_kernel(k_type, lengthscales, self.linear_variance))
                else:
                    kernels_to_combine.append(_create_single_kernel(k_type, lengthscales, self.variance))
            
            # Combine all kernels by addition
            combined_kernel = kernels_to_combine[0]
            for k in kernels_to_combine[1:]:
                combined_kernel = combined_kernel + k
        else:
            # Single kernel mode
            combined_kernel = _create_single_kernel(self.kernel_type, lengthscales, self.variance)
            
            # Optionally add Linear kernel (if add_linear is True and not using combinations)
            if self.add_linear:
                linear_kernel = gpf.kernels.Linear(variance=self.linear_variance)  # type: ignore
                combined_kernel = combined_kernel + linear_kernel

        return combined_kernel

    def fit(self, X, y):
        """
        Fit the GPflow Gaussian Process model.
        """
        # Convert to numpy if needed
        if hasattr(X, 'values'):  # pandas DataFrame
            X = X.values
        if hasattr(y, 'values'):  # pandas Series/DataFrame
            y = y.values

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Create kernel
        kernel = self._create_kernel(X_scaled.shape[1])

        # Set mean function
        if self.mean_function is None:
            mean_function = Constant()
        else:
            mean_function = self.mean_function

        # Create training data tuple for GPflow
        train_data = (X_scaled, y_scaled)

        # Create GPR model
        self.model = gpf.models.GPR(
            data=train_data,
            kernel=kernel,
            mean_function=mean_function,
            noise_variance=self.noise_variance
        )

        # Optimize hyperparameters if requested
        if self.optimize:
            opt = gpf.optimizers.Scipy()
            opt.minimize(
                self.model.training_loss,
                variables=self.model.trainable_variables,  # type: ignore
                options={"maxiter": self.maxiter}
            )

        self.is_fitted = True
        return self

    def predict(self, X, return_std=False):
        """
        Make predictions with the fitted GP model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        assert self.model is not None, "Model should be fitted"

        # Convert to numpy if needed
        if hasattr(X, 'values'):  # pandas DataFrame
            X = X.values

        # Scale the input
        X_scaled = self.scaler_X.transform(X)

        # Make predictions
        mean_scaled, var_scaled = self.model.predict_f(X_scaled)

        # Convert to numpy
        mean_scaled = mean_scaled.numpy()  # type: ignore
        var_scaled = var_scaled.numpy()  # type: ignore

        # Inverse transform predictions
        mean_pred = self.scaler_y.inverse_transform(mean_scaled)

        # Handle variance scaling (approximate)
        std_scaled = np.sqrt(var_scaled)
        assert self.scaler_y.scale_ is not None, "Scaler should be fitted"
        y_scale = self.scaler_y.scale_.reshape(1, -1)
        std_pred = std_scaled * y_scale

        # Return 1D arrays if output is 1D
        if mean_pred.shape[1] == 1:
            mean_pred = mean_pred.ravel()
            std_pred = std_pred.ravel()

        if return_std:
            return mean_pred, std_pred
        else:
            return mean_pred

    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty quantification.
        Returns both mean and standard deviation.
        """
        return self.predict(X, return_std=True)

    def sample_posterior(self, X, num_samples=10):
        """
        Sample from the posterior distribution.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        assert self.model is not None, "Model should be fitted"

        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values

        # Scale the input
        X_scaled = self.scaler_X.transform(X)

        # Sample from posterior
        samples_scaled = self.model.predict_f_samples(X_scaled, num_samples)
        samples_scaled = samples_scaled.numpy()  # type: ignore

        # Inverse transform samples
        samples = []
        for i in range(num_samples):
            sample = self.scaler_y.inverse_transform(samples_scaled[i])
            if sample.shape[1] == 1:
                sample = sample.ravel()
            samples.append(sample)

        return np.array(samples)

    def log_marginal_likelihood(self):
        """
        Compute the log marginal likelihood of the model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing likelihood")
        
        assert self.model is not None, "Model should be fitted"
        return self.model.log_marginal_likelihood().numpy()  # type: ignore


class GPflowMultiKernelGPR:
    """
    Advanced GPflow GP model that can test multiple kernel types and select the best one.
    Based on a kernel-comparison pattern used in earlier project prototypes.
    """

    def __init__(self, kernel_types=None, lengthscales_range=(0.25, 2.0),
                 variance_range=(0.1, 2.0), optimize=True, maxiter=1000, **kwargs):

        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow is required for GPflowMultiKernelGPR")

        if kernel_types is None:
            kernel_types = ["matern32", "matern52", "rbf", "exponential"]

        self.kernel_types = kernel_types
        self.lengthscales_range = lengthscales_range
        self.variance_range = variance_range
        self.optimize = optimize
        self.maxiter = maxiter

        self.best_model = None
        self.best_kernel_info = None
        self.all_models = []
        self.model_results = []

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _create_kernel_variants(self, input_dim):
        """
        Create multiple kernel variants for comparison.
        """
        kernels = []
        kernel_info = []

        # Generate lengthscale variants
        lengthscales_values = np.arange(
            self.lengthscales_range[0],
            self.lengthscales_range[1] + 0.25,
            0.25
        )
        variance_values = np.arange(
            self.variance_range[0],
            self.variance_range[1] + 0.1,
            0.5
        )

        for kernel_type in self.kernel_types:
            for lengthscale in lengthscales_values:
                for variance in variance_values:

                    lengthscales = np.ones(input_dim) * lengthscale

                    # Create base kernel
                    if kernel_type == "matern32":
                        base_kernel = gpf.kernels.Matern32(lengthscales=lengthscales, variance=variance)
                    elif kernel_type == "matern52":
                        base_kernel = gpf.kernels.Matern52(lengthscales=lengthscales, variance=variance)
                    elif kernel_type == "rbf":
                        base_kernel = gpf.kernels.SquaredExponential(lengthscales=lengthscales, variance=variance)
                    elif kernel_type == "exponential":
                        base_kernel = gpf.kernels.Exponential(lengthscales=lengthscales, variance=variance)
                    else:
                        continue

                    # Add linear component
                    linear_kernel = gpf.kernels.Linear(variance=0.1)  # type: ignore
                    combined_kernel = base_kernel + linear_kernel

                    kernels.append(combined_kernel)
                    kernel_info.append({
                        'type': kernel_type,
                        'lengthscales': lengthscales,
                        'variance': variance,
                        'name': f"{kernel_type} + Linear"
                    })

        return kernels, kernel_info

    def fit(self, X, y, validation_split=0.2):
        """
        Fit multiple GP models and select the best one based on validation performance.
        """
        # Data preprocessing
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, test_size=validation_split, random_state=42
            )
        else:
            X_train, y_train = X_scaled, y_scaled
            X_val, y_val = X_scaled, y_scaled

        # Create kernel variants
        kernels, kernel_info = self._create_kernel_variants(X_scaled.shape[1])

        print(f"Testing {len(kernels)} kernel configurations...")

        best_score = -np.inf

        for i, (kernel, info) in enumerate(zip(kernels, kernel_info)):
            try:
                # Create and train model
                train_data = (X_train, y_train)
                model = gpf.models.GPR(
                    data=train_data,
                    kernel=kernel,
                    mean_function=Constant()
                )

                # Optimize if requested
                if self.optimize:
                    opt = gpf.optimizers.Scipy()
                    opt.minimize(
                        model.training_loss,
                        variables=model.trainable_variables,  # type: ignore
                        options={"maxiter": self.maxiter}
                    )

                # Evaluate on validation set
                mean_pred, _ = model.predict_f(X_val)
                mse = np.mean((y_val - mean_pred.numpy()) ** 2)  # type: ignore
                log_likelihood = model.log_marginal_likelihood().numpy()  # type: ignore

                # Store results
                result = {
                    'kernel_info': info,
                    'mse': mse,
                    'log_likelihood': log_likelihood,
                    'score': log_likelihood  # Use log likelihood as selection criterion
                }

                self.all_models.append(model)
                self.model_results.append(result)

                # Check if this is the best model
                if result['score'] > best_score:
                    best_score = result['score']
                    self.best_model = model
                    self.best_kernel_info = info

                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(kernels)} kernel configurations")

            except Exception as e:
                print(f"Failed to fit kernel {info['name']}: {e}")
                continue

        if self.best_model is None:
            raise RuntimeError("No models were successfully fitted")

        assert self.best_kernel_info is not None, "Best kernel info should be set"
        print(f"Best kernel: {self.best_kernel_info['name']}")
        print(f"Best score (log likelihood): {best_score:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X, return_std=False):
        """
        Make predictions using the best fitted model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        assert self.best_model is not None, "Best model should be fitted"

        # Convert and scale input
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler_X.transform(X)

        # Make predictions
        mean_scaled, var_scaled = self.best_model.predict_f(X_scaled)
        mean_scaled = mean_scaled.numpy()  # type: ignore
        var_scaled = var_scaled.numpy()  # type: ignore

        # Inverse transform
        mean_pred = self.scaler_y.inverse_transform(mean_scaled)

        if mean_pred.shape[1] == 1:
            mean_pred = mean_pred.ravel()

        if return_std:
            std_scaled = np.sqrt(var_scaled)
            assert self.scaler_y.scale_ is not None, "Scaler should be fitted"
            y_scale = self.scaler_y.scale_.reshape(1, -1)
            std_pred = std_scaled * y_scale
            if std_pred.shape[1] == 1:
                std_pred = std_pred.ravel()
            return mean_pred, std_pred

        return mean_pred

    def get_best_kernel_info(self):
        """
        Return information about the best kernel found.
        """
        return self.best_kernel_info

    def get_all_results(self):
        """
        Return results for all tested kernels.
        """
        return self.model_results
