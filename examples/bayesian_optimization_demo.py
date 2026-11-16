"""
Bayesian Optimization for RandomForestRegressor using Optuna and BoTorchSampler
Demonstrates convergence of R² score over 100 trials with visualization

This script demonstrates:
1. RandomForestRegressor hyperparameter optimization using Optuna
2. BoTorchSampler for Bayesian optimization
3. 100 trials showing convergence
4. Visualization of how maximum R² increases with iterations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_regression
import optuna
from optuna.integration import BoTorchSampler
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')

def generate_regression_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data for optimization demo."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y

def objective(trial, X_train, y_train):
    """Objective function for Optuna optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
    return cv_scores.mean()

def run_optimization(X_train, y_train, sampler_name="BoTorch", n_trials=100):
    """Run optimization with specified sampler."""
    if sampler_name == "BoTorch":
        sampler = BoTorchSampler()
    else:
        sampler = TPESampler(seed=42)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"RandomForest_{sampler_name}"
    )
    
    def objective_wrapper(trial):
        return objective(trial, X_train, y_train)
    
    print(f"Running {sampler_name} optimization with {n_trials} trials...")
    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True)
    
    return study

def analyze_convergence(study, title="Bayesian Optimization Convergence"):
    """Analyze and visualize convergence of the optimization."""
    # Extract trial values
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    trial_numbers = list(range(1, len(trial_values) + 1))
    
    # Calculate running maximum (best score so far)
    running_max = []
    current_max = -np.inf
    for value in trial_values:
        if value > current_max:
            current_max = value
        running_max.append(current_max)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Running maximum R² score
    ax1.plot(trial_numbers, running_max, 'b-', linewidth=2, label='Best R² Score')
    ax1.scatter(trial_numbers, trial_values, alpha=0.3, s=30, color='gray', label='All Trials')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'{title}: Best Score Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight final best score
    final_best = running_max[-1]
    ax1.axhline(y=final_best, color='red', linestyle='--', alpha=0.7, 
                label=f'Final Best: {final_best:.4f}')
    ax1.legend()
    
    # Plot 2: Improvement rate over trials
    improvements = np.diff(running_max)
    improvement_trials = [i+2 for i, imp in enumerate(improvements) if imp > 0]
    improvement_values = [improvements[i] for i, imp in enumerate(improvements) if imp > 0]
    
    ax2.bar(improvement_trials, improvement_values, alpha=0.7, color='green')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Improvement in R² Score')
    ax2.set_title('Improvements Found During Optimization')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, running_max, trial_values

def print_convergence_summary(study, running_max, trial_values):
    """Print detailed convergence analysis."""
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    final_best = running_max[-1]
    initial_score = trial_values[0]
    total_improvement = final_best - initial_score
    
    # Find when 90% of improvement was achieved
    target_score = initial_score + 0.9 * total_improvement
    convergence_trial = next((i+1 for i, score in enumerate(running_max) if score >= target_score), len(running_max))
    
    # Count significant improvements (> 0.001)
    improvements = np.diff(running_max)
    significant_improvements = sum(1 for imp in improvements if imp > 0.001)
    
    print(f"📊 Best R² Score: {final_best:.6f}")
    print(f"📊 Initial Score: {initial_score:.6f}")
    print(f"📊 Total Improvement: {total_improvement:.6f}")
    print(f"📊 90% Convergence at Trial: {convergence_trial}")
    print(f"📊 Significant Improvements: {significant_improvements}")
    print(f"📊 Final 10-trial Stability: {np.std(running_max[-10:]):.6f}")
    
    # Best parameters
    print(f"\n🎯 Best Parameters:")
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"   {param}: {value}")

def compare_samplers(X_train, y_train, n_trials=100):
    """Compare TPE and BoTorch samplers."""
    print("🚀 BAYESIAN OPTIMIZATION COMPARISON")
    print("="*60)
    
    # Run both optimizations
    tpe_study = run_optimization(X_train, y_train, "TPE", n_trials)
    botorch_study = run_optimization(X_train, y_train, "BoTorch", n_trials)
    
    # Analyze convergence for both
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # TPE Analysis
    tpe_values = [trial.value for trial in tpe_study.trials if trial.value is not None]
    tpe_running_max = []
    current_max = -np.inf
    for value in tpe_values:
        if value > current_max:
            current_max = value
        tpe_running_max.append(current_max)
    
    # BoTorch Analysis
    botorch_values = [trial.value for trial in botorch_study.trials if trial.value is not None]
    botorch_running_max = []
    current_max = -np.inf
    for value in botorch_values:
        if value > current_max:
            current_max = value
        botorch_running_max.append(current_max)
    
    trial_numbers = list(range(1, n_trials + 1))
    
    # Plot comparisons
    axes[0, 0].plot(trial_numbers, tpe_running_max, 'b-', linewidth=2, label='TPE Best Score')
    axes[0, 0].scatter(trial_numbers, tpe_values, alpha=0.3, s=20, color='blue')
    axes[0, 0].set_title('TPE Sampler Convergence')
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(trial_numbers, botorch_running_max, 'r-', linewidth=2, label='BoTorch Best Score')
    axes[0, 1].scatter(trial_numbers, botorch_values, alpha=0.3, s=20, color='red')
    axes[0, 1].set_title('BoTorch Sampler Convergence')
    axes[0, 1].set_xlabel('Trial Number')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Direct comparison
    axes[1, 0].plot(trial_numbers, tpe_running_max, 'b-', linewidth=2, label=f'TPE (Best: {tpe_running_max[-1]:.4f})')
    axes[1, 0].plot(trial_numbers, botorch_running_max, 'r-', linewidth=2, label=f'BoTorch (Best: {botorch_running_max[-1]:.4f})')
    axes[1, 0].set_title('TPE vs BoTorch Comparison')
    axes[1, 0].set_xlabel('Trial Number')
    axes[1, 0].set_ylabel('Best R² Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Improvement rate comparison
    tpe_improvements = np.diff(tpe_running_max)
    botorch_improvements = np.diff(botorch_running_max)
    
    tpe_imp_count = [sum(1 for imp in tpe_improvements[:i] if imp > 0) for i in range(1, len(tpe_improvements)+1)]
    botorch_imp_count = [sum(1 for imp in botorch_improvements[:i] if imp > 0) for i in range(1, len(botorch_improvements)+1)]
    
    axes[1, 1].plot(trial_numbers[1:], tpe_imp_count, 'b-', linewidth=2, label='TPE Improvements')
    axes[1, 1].plot(trial_numbers[1:], botorch_imp_count, 'r-', linewidth=2, label='BoTorch Improvements')
    axes[1, 1].set_title('Cumulative Improvements')
    axes[1, 1].set_xlabel('Trial Number')
    axes[1, 1].set_ylabel('Number of Improvements')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summaries
    print("\n" + "="*30 + " TPE SUMMARY " + "="*30)
    print_convergence_summary(tpe_study, tpe_running_max, tpe_values)
    
    print("\n" + "="*30 + " BOTORCH SUMMARY " + "="*30)
    print_convergence_summary(botorch_study, botorch_running_max, botorch_values)
    
    return tpe_study, botorch_study

def main():
    """Main demonstration function."""
    print("🎯 BAYESIAN OPTIMIZATION DEMO FOR RANDOM FOREST")
    print("=" * 60)
    
    # Generate data
    print("📊 Generating synthetic regression data...")
    X, y = generate_regression_data(n_samples=1000, n_features=10, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Single BoTorch demonstration
    print("\n🔬 Running BoTorch Bayesian Optimization...")
    study = run_optimization(X_train, y_train, "BoTorch", n_trials=100)
    
    # Analyze convergence
    fig, running_max, trial_values = analyze_convergence(study, "BoTorch Bayesian Optimization")
    plt.savefig('botorch_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print_convergence_summary(study, running_max, trial_values)
    
    # Compare samplers
    print("\n🔄 Comparing TPE vs BoTorch samplers...")
    tpe_study, botorch_study = compare_samplers(X_train, y_train, n_trials=100)
    
    print("\n🎉 Demo completed! Check the generated plots for visualization.")
    
    return study

if __name__ == "__main__":
    study = main()
