Quick Start Guide
================

This guide will get you up and running with SURGE in just a few minutes.

Basic Example
-------------

Here's a simple example demonstrating hyperparameter optimization for a RandomForest model:

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score
   import optuna
   
   # Generate sample data
   np.random.seed(42)
   X = np.random.randn(1000, 5)
   y = 2*X[:, 0] + X[:, 1]**2 + 0.5*X[:, 2] + 0.1*np.random.randn(1000)
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Define objective function for optimization
   def objective(trial):
       params = {
           'n_estimators': trial.suggest_int('n_estimators', 50, 200),
           'max_depth': trial.suggest_int('max_depth', 3, 15),
           'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
           'random_state': 42
       }
       
       model = RandomForestRegressor(**params)
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       
       return r2_score(y_test, predictions)
   
   # Run optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   
   print(f"Best R² score: {study.best_value:.4f}")
   print(f"Best parameters: {study.best_params}")

Using SURGE Models
------------------

SURGE provides pre-built model classes for common use cases:

.. code-block:: python

   from surge.models import RandomForestModel
   from surge.preprocessing import StandardScaler
   from surge.metrics import evaluate_model
   
   # Preprocess data
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Create and train model
   model = RandomForestModel(
       n_estimators=100,
       max_depth=10,
       random_state=42
   )
   model.fit(X_scaled, y_train)
   
   # Make predictions
   predictions = model.predict(X_test_scaled)
   
   # Evaluate performance
   metrics = evaluate_model(y_test, predictions)
   print(f"Model performance: {metrics}")

Hyperparameter Optimization with BoTorch
-----------------------------------------

For advanced Bayesian optimization using BoTorch:

.. code-block:: python

   from optuna.integration import BoTorchSampler
   
   # Create study with BoTorch sampler
   sampler = BoTorchSampler()
   study = optuna.create_study(direction='maximize', sampler=sampler)
   study.optimize(objective, n_trials=50)
   
   # Analyze convergence
   trial_values = [trial.value for trial in study.trials]
   
   import matplotlib.pyplot as plt
   plt.plot(trial_values)
   plt.xlabel('Trial')
   plt.ylabel('R² Score')
   plt.title('Optimization Convergence')
   plt.show()

Next Steps
----------

* Read the :doc:`user_guide/index` for detailed explanations
* Check out the :doc:`examples/index` for more complex use cases
* Explore the :doc:`api_reference/index` for complete API documentation
