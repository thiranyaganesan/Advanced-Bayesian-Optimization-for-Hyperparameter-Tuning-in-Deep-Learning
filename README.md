# Advanced Bayesian Optimization for Hyperparameter Tuning (Code Only)

This repository contains code to run hyperparameter tuning for a Keras CNN on CIFAR-10
using **Optuna** (Bayesian-like optimization) and an example using **scikit-optimize**.

**Files included**
- `data.py` - CIFAR-10 loader + preprocessing.
- `model.py` - Keras model factory that builds a CNN from hyperparameters.
- `optuna_objective.py` - Optuna objective function and study runner.
- `skopt_objective.py` - Example using scikit-optimize (optional dependency).
- `run_optuna.py` - Simple CLI wrapper to run Optuna tuning and save results.
- `requirements.txt` - Python package suggestions.

**Quick start (recommended)**
1. Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Optuna tuning (default 20 trials, change `n_trials` in run_optuna.py):
   ```bash
   python run_optuna.py
   ```
3. The best hyperparameters will be saved to `best_params_optuna.json` and the study to `optuna_study.db`.

**Notes**
- Code is intentionally compact and commented. It is meant for educational use and can be extended
  to larger search spaces, different datasets, or more epochs.
- If you prefer `scikit-optimize`, see `skopt_objective.py` for an example; install `scikit-optimize` first.
