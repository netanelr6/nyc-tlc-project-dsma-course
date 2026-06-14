"""
NYC TLC Yellow Taxi -- Model Creation & Training
=================================================

Defines a registry of candidate regression models for the ETA prediction task.
Each model is trained on the feature DataFrame produced by features.py and
saved to disk so that evaluation.py can load and compare them independently.

Adding a new model
------------------
1. Import the class at the top of this file.
2. Add an entry to CANDIDATE_MODELS with a descriptive string key.
That's it — train_all_models() will pick it up automatically.
"""

import joblib
from pathlib import Path
import threading
import sys
import time

def run_live_timer(stop_event, message):
    if not sys.stdout.isatty():
        return
    if len(message) > 60:
        message = message[:57] + "..."
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r  {message} ... [Elapsed: {elapsed:.1f}s]")
        sys.stdout.flush()
        time.sleep(0.5)

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, HistGradientBoostingRegressor  # noqa: F401
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.svm             import LinearSVR
from sklearn.base            import clone

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ── Candidate model registry ──────────────────────────────────────────────────
#
# This dictionary drives the entire training loop.
# Keys become the model's file name on disk (e.g. "random_forest.pkl").
#
# Hyperparameter note:
#   The values here are sensible starting points.  Lecture 3 will introduce
#   systematic hyperparameter tuning — treat these as the baseline to beat.

CANDIDATE_MODELS = {
    "linear_regression": LinearRegression(),

    "decision_tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=1000,
        max_features="sqrt",
        random_state=42,
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        min_samples_leaf=100,
        max_samples=0.01,
        n_jobs=-1,
        random_state=42,
    ),

    "gradient_boosting": HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=50,
        random_state=42,
    ),

    "neural_network": MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="relu",
        max_iter=5,
        batch_size=16384,
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42,
    ),

    "svm": LinearSVR(
        C=1.0,
        epsilon=0.1,
        loss="squared_epsilon_insensitive",
        dual=False,
        random_state=42,
        max_iter=2000,
    ),
}




# ── Training functions ────────────────────────────────────────────────────────

def get_candidate_models():
    """
    Return fresh (unfitted) copies of every model in the registry.
    Using clone() ensures a clean slate even if this is called multiple times.
    """
    return {name: clone(model) for name, model in CANDIDATE_MODELS.items()}


def train_all_models(X_train, y_train, model_dir, nn_limit=100000, svm_limit=100000):
    """
    Train every candidate model and save each to disk.

    Args:
        X_train   : pd.DataFrame of training features
        y_train   : pd.Series   of training labels (trip_duration_minutes)
        model_dir : str | Path  directory where .pkl files will be written
        nn_limit  : int | None  subsample limit for Neural Network training (or None)
        svm_limit : int | None  subsample limit for SVM training (or None)

    Returns:
        dict mapping model name → fitted model object
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    models  = get_candidate_models()
    trained = {}

    import time
    for name, model in models.items():
        t0 = time.time()
        is_tty = sys.stdout.isatty()
        if is_tty:
            stop_event = threading.Event()
            timer_thread = threading.Thread(target=run_live_timer, args=(stop_event, f"Training {name}"))
            timer_thread.daemon = True
            timer_thread.start()
        else:
            print(f"  Training {name} ...")

        # Subsample neural network and SVM training data to prevent CPU slowdown on large datasets
        if name == "neural_network" and nn_limit is not None and len(X_train) > nn_limit:
            print(f"  [Info] Subsampling training data to {nn_limit:,} samples specifically for {name} to prevent CPU slowdown.")
            X_tr = X_train.sample(n=nn_limit, random_state=42)
            y_tr = y_train.loc[X_tr.index]
        elif name == "svm" and svm_limit is not None and len(X_train) > svm_limit:
            print(f"  [Info] Subsampling training data to {svm_limit:,} samples specifically for {name} to prevent CPU slowdown.")
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
            sys.stdout.write(f"\r\033[33m  [Duration] Finished training {name} in {duration:.2f}s\033[0m\n")
            sys.stdout.flush()
        else:
            print(f"\033[33m  [Duration] Finished training {name} in {duration:.2f}s\033[0m")

        save_model(model, name, model_dir)
        trained[name] = model

    return trained


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_model(model, name, model_dir):
    """Serialise a fitted model to <model_dir>/<name>.pkl."""
    path = Path(model_dir) / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name, model_dir):
    """Load a previously saved model by name."""
    path = Path(model_dir) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    return joblib.load(path)


def load_all_models(model_dir):
    """
    Load every .pkl file in model_dir and return a dict of
    { model_name: fitted_model } (skips scaler.pkl).
    """
    model_dir = Path(model_dir)
    models    = {}
    for pkl_file in sorted(model_dir.glob("*.pkl")):
        if pkl_file.name == "scaler.pkl":
            continue
        models[pkl_file.stem] = joblib.load(pkl_file)
    if not models:
        raise FileNotFoundError(f"No model .pkl files found in {model_dir}")
    return models
