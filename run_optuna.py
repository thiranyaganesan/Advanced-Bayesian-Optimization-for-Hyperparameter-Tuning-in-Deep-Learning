import argparse
from optuna_objective import run_optuna

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optuna Bayesian-style hyperparameter tuning.')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--use_sqlite', action='store_true', help='Save study to sqlite file (optuna_study.db)')
    args = parser.parse_args()

    storage = None
    if args.use_sqlite:
        storage = 'sqlite:///optuna_study.db'
    run_optuna(n_trials=args.n_trials, storage=storage)
