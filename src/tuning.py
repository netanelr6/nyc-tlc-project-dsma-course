"""
NYC TLC -- Hyperparameter Tuning via W&B Sweeps (Trip Fare Prediction)
======================================================================
Provides configuration and functions for hyperparameter tuning using W&B Sweeps
and retraining the best model on the full dataset.


Two-phase tuning strategy that mirrors the lecture theory:

  Phase 1 — Random Search (model family selection)
    Wide parameter grid, both Random Forest and Gradient Boosting.
    Purpose: quickly identify which model family fits this dataset.
    W&B sweep method: "random"

  Phase 2 — Grid Search (fine-tuning the winner)
    Narrow, exhaustive grid around the best region found in Phase 1.
    Purpose: rigorously squeeze out the last performance from the winner.
    W&B sweep method: "grid"

Adding a new model family
-------------------------
1. Add a MODEL_TYPE entry to RANDOM_SEARCH_CONFIG["parameters"]["model_type"].
2. Add its hyperparameters to RANDOM_SEARCH_CONFIG["parameters"].
3. Add a matching entry to GRID_SEARCH_CONFIGS.
4. Handle it in _build_model_from_config().
"""

import wandb
import joblib
import json
import numpy as np
import pandas as pd
from pathlib                     import Path
from sklearn.ensemble            import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model        import LinearRegression, Ridge
from sklearn.tree                import DecisionTreeRegressor
from sklearn.neural_network      import MLPRegressor
from sklearn.svm                 import LinearSVR
from sklearn.model_selection     import cross_val_score, cross_validate
from sklearn.base                import clone


try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


from src.models       import save_model, CANDIDATE_MODELS, run_live_timer
from src.experiment_tracking import configure_wandb
import threading
import sys


def get_stratified_tuning_sample(X, y, sample_size=100000, random_state=42):
    """
    Select a representative subsample of the training data stratified by
    pickup_hour and day_of_week to preserve temporal and traffic patterns.
    """
    if sample_size is None or len(X) <= sample_size:
        return X, y
    
    stratify_cols = []
    for col in X.columns:
        if "hour" in col or "day_of_week" in col:
            if X[col].nunique() <= 24:
                stratify_cols.append(col)
                
    if not stratify_cols:
        print("  [Warning] No temporal features found for stratification. Falling back to simple random sampling.")
        X_sample = X.sample(n=sample_size, random_state=random_state)
        y_sample = y.loc[X_sample.index]
        return X_sample, y_sample

    frac = sample_size / len(X)
    try:
        sampled_indices = X.groupby(stratify_cols, group_keys=False).apply(
            lambda g: g.sample(n=max(1, int(np.round(len(g) * frac))), random_state=random_state)
        ).index
        
        if len(sampled_indices) > sample_size:
            sampled_indices = pd.Index(sampled_indices).to_series().sample(n=sample_size, random_state=random_state)
        elif len(sampled_indices) < sample_size:
            extra_size = sample_size - len(sampled_indices)
            remaining = X.index.difference(sampled_indices)
            extra_indices = remaining.to_series().sample(n=extra_size, random_state=random_state)
            sampled_indices = sampled_indices.union(extra_indices)
            
        X_sample = X.loc[sampled_indices]
        y_sample = y.loc[sampled_indices]
        return X_sample, y_sample
    except Exception as e:
        print(f"  [Warning] Stratified sampling failed ({e}). Falling back to simple random sampling.")
        X_sample = X.sample(n=sample_size, random_state=random_state)
        y_sample = y.loc[X_sample.index]
        return X_sample, y_sample



# ── Supported models list ─────────────────────────────────────────────────────

supported_models = ["random_forest", "gradient_boosting", "linear_regression", "ridge", "neural_network", "svm", "decision_tree"]
if _XGBOOST_AVAILABLE:
    supported_models.append("xgboost")


# ── Phase 1: Random search config ─────────────────────────────────────────────
#
# Sweeps across all supported model families in a single sweep run.
# Parameters that are not supported by a model family are ignored by its builder.

RANDOM_SEARCH_CONFIG = {
    "method": "random",
    "metric": {"name": "mae", "goal": "minimize"},
    "parameters": {
        "model_type":       {"values": supported_models},
        "n_estimators":     {"values": [50, 100, 150, 200]},
        "max_depth":        {"values": [5, 10, 15, 20]},
        "min_samples_leaf": {"values": [10, 20, 50, 100]},
        "learning_rate":    {"values": [0.01, 0.05, 0.1, 0.2]},
        "max_features":     {"values": ["sqrt", "log2"]},
        "alpha":            {"values": [0.01, 0.1, 1.0, 10.0]},
        "fit_intercept":    {"values": [True, False]},
        
        # Neural Network (MLPRegressor) parameters
        "nn_epochs":        {"values": [5, 10]},
        "nn_activation":    {"values": ["relu"]},
        "nn_hidden_layers": {"values": [[32], [50], [32, 16]]},
        "nn_learning_rate": {"values": [0.001, 0.01]},

        # SVM (LinearSVR) parameters
        "svm_c":            {"values": [0.1, 1.0, 10.0]},
        "svm_epsilon":      {"values": [0.0, 0.1, 0.2]},
    },
}


# ── Phase 2: Grid search configs (one per model family) ───────────────────────
#
# Exhaustive search over a tight grid centred on the region the random
# search identified as promising.  Run only the winning family here.
# Grid sizes are kept small (≤ 12 combinations) for classroom runtime.

GRID_SEARCH_CONFIGS = {
    "random_forest": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "random_forest"},
            "n_estimators":     {"values": [100, 200]},
            "max_depth":        {"values": [10, 15]},
            "min_samples_leaf": {"values": [50, 100]},
        },
    },
    "gradient_boosting": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "gradient_boosting"},
            "n_estimators":     {"values": [100, 200]},
            "max_depth":        {"values": [5, 8]},
            "learning_rate":    {"values": [0.05, 0.1]},
            "min_samples_leaf": {"values": [50]},
        },
    },
    "linear_regression": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "linear_regression"},
            "fit_intercept":    {"values": [True, False]},
        },
    },
    "ridge": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "ridge"},
            "alpha":            {"values": [0.1, 1.0, 10.0]},
        },
    },
    "neural_network": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "neural_network"},
            "nn_epochs":        {"values": [10]},
            "nn_activation":    {"values": ["relu"]},
            "nn_hidden_layers": {"values": [[32]]},
        },
    },
    "svm": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "svm"},
            "svm_c":            {"values": [0.1, 1.0]},
            "svm_epsilon":      {"values": [0.1]},
        },
    },
    "decision_tree": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "decision_tree"},
            "max_depth":        {"values": [10, 15]},
            "min_samples_leaf": {"values": [50, 100]},
        },
    },
}

if _XGBOOST_AVAILABLE:
    GRID_SEARCH_CONFIGS["xgboost"] = {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "xgboost"},
            "n_estimators":     {"values": [100, 200]},
            "max_depth":        {"values": [5, 8]},
            "learning_rate":    {"values": [0.05, 0.1]},
        },
    }

# ── Model builder ─────────────────────────────────────────────────────────────

def _build_model_from_config(cfg):
    """
    Instantiate an unfitted sklearn model from a W&B run config dict.
    Unused parameters (e.g. learning_rate for RF) are silently ignored.
    """
    model_type = cfg.get("model_type", "random_forest")

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators     = int(cfg.get("n_estimators", 100)),
            max_depth        = cfg.get("max_depth", None),
            min_samples_leaf = int(cfg.get("min_samples_leaf", 50)),
            max_features     = cfg.get("max_features", "sqrt"),
            max_samples      = 0.01,
            n_jobs           = -1,
            random_state     = 42,
        )
    elif model_type == "decision_tree":
        return DecisionTreeRegressor(
            max_depth        = cfg.get("max_depth", 10),
            min_samples_leaf = int(cfg.get("min_samples_leaf", 50)),
            max_features     = cfg.get("max_features", "sqrt"),
            random_state     = 42,
        )
    elif model_type == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter         = int(cfg.get("n_estimators", 100)),
            max_depth        = int(cfg.get("max_depth", 5)),
            learning_rate    = float(cfg.get("learning_rate", 0.1)),
            min_samples_leaf = int(cfg.get("min_samples_leaf", 50)),
            random_state     = 42,
        )
    elif model_type == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost is selected but not installed in the environment.")
        return XGBRegressor(
            n_estimators     = int(cfg.get("n_estimators", 100)),
            max_depth        = int(cfg.get("max_depth", 5)),
            learning_rate    = float(cfg.get("learning_rate", 0.1)),
            n_jobs           = -1,
            random_state     = 42,
        )
    elif model_type == "linear_regression":
        return LinearRegression(
            fit_intercept    = bool(cfg.get("fit_intercept", True))
        )
    elif model_type == "ridge":
        return Ridge(
            alpha            = float(cfg.get("alpha", 1.0)),
            random_state     = 42,
        )
    elif model_type == "neural_network":
        hidden_layers = tuple(cfg.get("nn_hidden_layers", [32]))
        return MLPRegressor(
            hidden_layer_sizes = hidden_layers,
            activation         = cfg.get("nn_activation", "relu"),
            max_iter           = int(cfg.get("nn_epochs", 5)),
            learning_rate_init = float(cfg.get("nn_learning_rate", 0.001)),
            solver             = "adam",
            batch_size         = 16384,
            early_stopping     = True,
            random_state       = 42,
        )
    elif model_type == "svm":
        return LinearSVR(
            C            = float(cfg.get("svm_c", 1.0)),
            epsilon      = float(cfg.get("svm_epsilon", 0.1)),
            loss         = "squared_epsilon_insensitive",
            dual         = False,
            random_state = 42,
            max_iter     = 2000,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


# ── Sweep training closure ────────────────────────────────────────────────────

def neg_mape_scorer(estimator, X, y):
    """
    Custom scorer to compute negative Mean Absolute Percentage Error (MAPE).
    Excludes values where y < 1.0 (following the compute_metrics logic).
    Returns negative value since sklearn maximizes scorers.
    """
    y_pred = estimator.predict(X)
    y_true = np.asarray(y, dtype=float)
    mask = y_true >= 1.0
    if not np.any(mask):
        return 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return -float(mape)


def _make_train_fn(X_train, y_train, nn_limit=100000, svm_limit=100000):
    """
    Return a zero-argument callable suitable for wandb.agent().

    The closure captures X_train / y_train so the agent can call it
    without arguments.  Each invocation:
      1. Reads hyperparameters from wandb.config
      2. Runs 3-fold CV and computes mean MAE, RMSE and MAPE using cross_validate
      3. Logs mae, rmse and mape to W&B
    """
    def train_fn():
        with wandb.init() as run:
            cfg     = run.config
            model   = _build_model_from_config(cfg)

            model_type = cfg.get("model_type", "random_forest")
            if model_type == "neural_network" and nn_limit is not None and len(X_train) > nn_limit:
                X_tr = X_train.sample(n=nn_limit, random_state=42)
                y_tr = y_train.loc[X_tr.index]
            elif model_type == "svm" and svm_limit is not None and len(X_train) > svm_limit:
                X_tr = X_train.sample(n=svm_limit, random_state=42)
                y_tr = y_train.loc[X_tr.index]
            else:
                X_tr = X_train
                y_tr = y_train

            scores = cross_validate(
                model, X_tr, y_tr,
                cv      = 3,
                scoring = {
                    "mae": "neg_mean_absolute_error",
                    "mse": "neg_mean_squared_error",
                    "mape": neg_mape_scorer
                },
                n_jobs  = None,
            )
            mae  = float(-scores["test_mae"].mean())
            rmse = float((-scores["test_mse"].mean()) ** 0.5)
            mape = float(-scores["test_mape"].mean())

            run.log({"mae": mae, "rmse": rmse, "mape": mape})

    return train_fn

import random
import itertools

def _generate_combinations(parameters):
    """Generate all parameter combinations from a sweep config parameter dict."""
    keys = list(parameters.keys())
    grid_lists = []
    for k in keys:
        v = parameters[k]
        if "values" in v:
            grid_lists.append(v["values"])
        elif "value" in v:
            grid_lists.append([v["value"]])
        else:
            grid_lists.append([])
    
    combinations = []
    for comb in itertools.product(*grid_lists):
        combinations.append(dict(zip(keys, comb)))
    return combinations


# ── Public API ────────────────────────────────────────────────────────────────

def run_wandb_sweep(X_train, y_train, sweep_config: dict,
                    project: str, entity: str = None, n_runs: int = 15,
                    nn_limit=100000, svm_limit=100000):
    """
    Register a W&B sweep, run `n_runs` trials, and return the best config.
    If W&B is offline or not logged in, falls back to local tuning using cross-validation.

    Args:
        X_train      : training feature DataFrame
        y_train      : training target Series
        sweep_config : RANDOM_SEARCH_CONFIG or GRID_SEARCH_CONFIGS[model]
        project      : W&B project name
        n_runs       : number of trials to run (ignored for grid sweeps,
                       which always run all combinations)
        nn_limit     : int | None  subsample limit for Neural Network training (or None)
        svm_limit    : int | None  subsample limit for SVM training (or None)

    Returns:
        sweep_id  (str)  : W&B sweep ID — "local-sweep" if run offline
        best_config (dict): hyperparameter dict of the best trial
        best_mae    (float): CV MAE of the best trial
    """
    import os
    configure_wandb()
    is_logged_in = False
    if os.environ.get("WANDB_MODE") != "disabled":
        try:
            is_logged_in = wandb.login(anonymous="never", relogin=False)
        except Exception:
            is_logged_in = False

    if is_logged_in:
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        train_fn = _make_train_fn(X_train, y_train, nn_limit=nn_limit, svm_limit=svm_limit)
        wandb.agent(sweep_id, function=train_fn, count=n_runs)

        api  = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"sweep": sweep_id})
        completed = [r for r in runs if "mae" in r.summary]

        if not completed:
            raise RuntimeError("Sweep produced no results -- check W&B connection.")

        best = min(completed, key=lambda r: r.summary["mae"])
        print(f"\n  \033[92m[Tuning] Sweep {sweep_id} complete!\033[0m")
        print(f"    - Best Run Name: \033[96m{best.name}\033[0m")
        print(f"    - Best Run ID:   \033[96m{best.id}\033[0m")
        print(f"    - Best Run MAE:  ${best.summary.get('mae', 0.0):.4f}")
        if "mape" in best.summary:
            print(f"    - Best Run MAPE: {best.summary.get('mape', 0.0):.2f}%")
        
        best_config_with_metadata = best.config.copy()
        best_config_with_metadata["run_name"] = best.name
        best_config_with_metadata["run_id"] = best.id
        best_config_with_metadata["sweep_id"] = sweep_id
        
        return sweep_id, best_config_with_metadata, best.summary["mae"]
    else:
        print("\033[93m[Tuning] W&B is not logged in or offline. Running local hyperparameter search...\033[0m")
        combinations = _generate_combinations(sweep_config["parameters"])
        
        if sweep_config.get("method") == "grid":
            selected_combs = combinations
        else:
            random.seed(42)
            if len(combinations) <= n_runs:
                selected_combs = combinations
            else:
                selected_combs = random.sample(combinations, n_runs)
                
        best_config = None
        best_mae = float("inf")
        
        import time
        print(f"  Running local search over {len(selected_combs)} configurations...")
        for i, comb in enumerate(selected_combs, 1):
            model = _build_model_from_config(comb)
            try:
                t0 = time.time()
                is_tty = sys.stdout.isatty()
                if is_tty:
                    stop_event = threading.Event()
                    timer_thread = threading.Thread(
                        target=run_live_timer,
                        args=(stop_event, f"    Trial {i}/{len(selected_combs)} ({comb.get('model_type')})")
                    )
                    timer_thread.daemon = True
                    timer_thread.start()
                else:
                    print(f"    Trial {i}/{len(selected_combs)}: {comb} ...")

                model_type = comb.get("model_type")
                if model_type == "neural_network" and nn_limit is not None and len(X_train) > nn_limit:
                    X_tr = X_train.sample(n=nn_limit, random_state=42)
                    y_tr = y_train.loc[X_tr.index]
                elif model_type == "svm" and svm_limit is not None and len(X_train) > svm_limit:
                    X_tr = X_train.sample(n=svm_limit, random_state=42)
                    y_tr = y_train.loc[X_tr.index]
                else:
                    X_tr = X_train
                    y_tr = y_train

                scores = cross_validate(
                    model, X_tr, y_tr,
                    cv=3,
                    scoring={
                        "mae": "neg_mean_absolute_error",
                        "mape": neg_mape_scorer
                    },
                    n_jobs=None
                )
                mae = float(-scores["test_mae"].mean())
                mape = float(-scores["test_mape"].mean())
                duration = time.time() - t0

                if is_tty:
                    stop_event.set()
                    timer_thread.join()
                    sys.stdout.write(f"\r\033[33m    [Duration] Trial {i}/{len(selected_combs)} finished in {duration:.2f}s -> MAE: ${mae:.2f}, MAPE: {mape:.2f}%\033[0m\n")
                    print(f"      Config: {comb}")
                    sys.stdout.flush()
                else:
                    print(f"\033[33m    [Duration] Trial {i}/{len(selected_combs)} finished in {duration:.2f}s -> MAE: ${mae:.2f}, MAPE: {mape:.2f}%\033[0m")

                if mae < best_mae:
                    best_mae = mae
                    best_config = comb
            except Exception as e:
                print(f"    Trial {i}/{len(selected_combs)} failed: {e}")
                
        print(f"  Best local config: {best_config}")
        print(f"  Best local MAE: ${best_mae:.2f}")
        return "local-sweep", best_config, best_mae


def retrain_best_model(best_config: dict, X_train, y_train,
                       model_dir: str = "models/winning_model",
                       nn_limit=100000, svm_limit=100000):
    """
    Build the winning model from its config, retrain on the full training
    set, and save it to disk.

    This is separate from the sweep because the sweep only runs CV —
    it never trains on the full dataset.  This step does the final fit.

    Returns:
        fitted model object
    """
    model = _build_model_from_config(best_config)
    model_type = best_config.get("model_type", "random_forest")

    import time
    t0 = time.time()
    is_tty = sys.stdout.isatty()
    if is_tty:
        stop_event = threading.Event()
        timer_thread = threading.Thread(
            target=run_live_timer,
            args=(stop_event, f"  Retraining best {model_type} on full training set")
        )
        timer_thread.daemon = True
        timer_thread.start()
    else:
        print(f"  Retraining best {model_type} on full training set...")

    if model_type == "neural_network" and nn_limit is not None and len(X_train) > nn_limit:
        print(f"  [Info] Subsampling training data to {nn_limit:,} samples specifically for {model_type} to prevent CPU slowdown.")
        X_tr = X_train.sample(n=nn_limit, random_state=42)
        y_tr = y_train.loc[X_tr.index]
    elif model_type == "svm" and svm_limit is not None and len(X_train) > svm_limit:
        print(f"  [Info] Subsampling training data to {svm_limit:,} samples specifically for {model_type} to prevent CPU slowdown.")
        X_tr = X_train.sample(n=svm_limit, random_state=42)
        y_tr = y_train.loc[X_tr.index]
    else:
        X_tr = X_train
        y_tr = y_train

    model.fit(X_tr, y_tr)
    duration = time.time() - t0

    if is_tty:
        stop_event.set()
        timer_thread.join()
        sys.stdout.write(f"\r\033[33m  [Duration] Finished retraining best {model_type} in {duration:.2f}s\033[0m\n")
        sys.stdout.flush()
    else:
        print(f"\033[33m  [Duration] Finished retraining best {model_type} in {duration:.2f}s\033[0m")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_model(model, f"tuned_{model_type}", model_dir)
    print(f"  Tuned model saved -> {save_path}")

    config_path = Path(model_dir) / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"  Best config saved -> {config_path}")

    return model
