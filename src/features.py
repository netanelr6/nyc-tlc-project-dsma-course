"""
NYC TLC Yellow Taxi -- Feature Engineering
===========================================

This module is organised into four sections:

  1. LEAKAGE REFERENCE      -- columns that cannot be used as features and why
  2. FEATURE CREATION       -- functions that build new columns from existing data
  3. FEATURE TRANSFORMATION -- functions that reshape existing columns
  4. PIPELINE ORCHESTRATION -- the plug-and-play pipeline entry point

Plug-and-play design
--------------------
FEATURE_CREATION_STEPS and FEATURE_TRANSFORMATION_STEPS are plain Python lists
of functions. To add a step: append the function. To remove one: comment it out.
The pipeline will execute them in order and return a clean feature DataFrame.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib


# ── Constants ─────────────────────────────────────────────────────────────────

# TARGET_COL = "trip_duration_minutes"
TARGET_COL = "total_fare_amount"

# Features to scale (fit on train, apply to both train and test)
# SCALE_FEATURES = ["trip_distance", "passenger_count"]
SCALE_FEATURES = ["trip_distance"]
# SCALE_FEATURES = [
#     "trip_distance",
#     "pickup_hour",
#     "day_of_week",
#     "is_weekend",
#     "time_of_day_bucket",
#     "distance_x_time_of_day",
#     "pickup_zone_x_hour",
#     "distance_x_rush_hour",
#     "pickup_hour_sin",
#     "pickup_hour_cos"
# ]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LEAKAGE REFERENCE
# ═════════════════════════════════════════════════════════════════════════════
#
# Column                   | Why it leaks
# -------------------------|---------------------------------------------------
# tpep_dropoff_datetime    | Recorded at trip end — IS the source of our target
# trip_duration_minutes    | IS the target variable
#
# Note on trip_distance: Strictly, the metered trip_distance is also post-trip.
# In production this would come from a routing API at pickup time. We treat it
# here as a proxy for "estimated distance" for teaching purposes.
# ─────────────────────────────────────────────────────────────────────────────

LEAKY_COLUMNS = [
    "tpep_dropoff_datetime",
    "total_amount",
    "tip_amount",
    "fare_amount",
    "payment_type",
    "RatecodeID",
]

# Raw datetime column consumed by feature creation; dropped after extraction
_DATETIME_COLS_TO_DROP = ["tpep_pickup_datetime"]


"""
instrections for the feature :

a.Input – PULocationID, DOLocationID, trip_distance, Time of day, Day of week
b.Output (target variable) – total_fare_amount (total_amount – tip_amount)
c.The challenge: unlike the trip_duration prediction problem, 
  total_amount prediction is highly sensitive to all the values that are added up for the fare. 
  This problem will require a significant amount of feature engineering to get high accuracy and low error rates.
"""

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE CREATION
# ═════════════════════════════════════════════════════════════════════════════

# ── Target variable ───────────────────────────────────────────────────────────

def _add_trip_duration_minutes(df):
    """Compute the target: elapsed trip time in minutes."""
    delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df[TARGET_COL] = delta.dt.total_seconds() / 60
    return df


def _add_total_fare_amount(df):
    """Compute the target: total fare amount (total_amount – tip_amount)."""
    df[TARGET_COL] = df["total_amount"] - df["tip_amount"]
    return df 


# ── Temporal features ─────────────────────────────────────────────────────────

def _add_pickup_hour(df):
    """Hour of day (0–23) extracted from pickup datetime."""
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    return df


def _add_day_of_week(df):
    """Day of week: 0 = Monday, 6 = Sunday."""
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    return df


def _add_is_weekend(df):
    """Binary flag: 1 if Saturday or Sunday, else 0."""
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_time_of_day_bucket(df):
    """
    Domain-driven bucketing of the day into three traffic time zones.

    Bucket values
    -------------
    0 = overnight  (00:00 – 05:59)  -- low demand, fast roads
    1 = off-peak   (06:00 – 23:59, excluding rush windows)
    2 = rush hour  (07:00 – 09:59 and 16:00 – 19:59)

    Also creates is_rush_hour (binary) as a convenience column used
    by the interaction feature steps below.
    """
    hour = df["pickup_hour"]
    is_morning_rush = hour.between(7, 9)
    is_evening_rush = hour.between(16, 19)
    is_overnight    = hour < 6

    df["is_rush_hour"] = (is_morning_rush | is_evening_rush).astype(int)

    df["time_of_day_bucket"] = 1  # default: off-peak
    df.loc[is_overnight, "time_of_day_bucket"] = 0
    df.loc[is_morning_rush | is_evening_rush, "time_of_day_bucket"] = 2
    return df


# ── Domain-driven features ────────────────────────────────────────────────────

# ── Interaction features ──────────────────────────────────────────────────────

def _add_distance_x_time_of_day(df):
    """
    Interaction: trip_distance × time_of_day_bucket.

    Intuition: a 5-mile trip at rush hour takes much longer than a 5-mile trip
    at midnight. This feature lets the model capture that multiplicative effect
    without needing a deep tree to discover it.
    """
    df["distance_x_time_of_day"] = df["trip_distance"] * df["time_of_day_bucket"]
    return df


def _add_pickup_zone_x_hour(df):
    """
    Interaction: PULocationID × pickup_hour.

    Intuition: Zone 161 (Midtown) at 08:00 behaves very differently from
    Zone 161 at 14:00. Multiplying encodes that joint signal as a single
    numeric feature.
    """
    df["pickup_zone_x_hour"] = df["PULocationID"] * df["pickup_hour"]
    return df


def _add_distance_x_rush_hour(df):
    """
    Interaction: trip_distance × is_rush_hour.

    Intuition: rush-hour congestion penalises longer trips
    disproportionately — this feature captures that non-linear relationship
    without requiring the model to learn it implicitly.
    """
    df["distance_x_rush_hour"] = df["trip_distance"] * df["is_rush_hour"]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE TRANSFORMATION
# ═════════════════════════════════════════════════════════════════════════════

def _add_cyclical_hour_encoding(df):
    """
    Cyclical encoding of pickup_hour using sine and cosine transforms.

    Why: hour 23 and hour 0 are adjacent in time but numerically far apart.
    Treating hour as a raw integer breaks that neighbourhood relationship.
    Sin/cos maps the 24-hour cycle onto a circle so midnight wraps back to
    midnight correctly.

      pickup_hour_sin = sin(2π × hour / 24)
      pickup_hour_cos = cos(2π × hour / 24)
    """
    cycle = 2 * np.pi * df["pickup_hour"] / 24
    df["pickup_hour_sin"] = np.sin(cycle)
    df["pickup_hour_cos"] = np.cos(cycle)
    return df



# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PIPELINE ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

# ── Plug-and-play step registries ─────────────────────────────────────────────
# Each entry is a function with signature:  fn(df: pd.DataFrame) -> pd.DataFrame
# Comment out any step to remove it from the pipeline.

FEATURE_CREATION_STEPS = [
    _add_pickup_hour,              # 1. extract hour (needed by later steps)
    _add_day_of_week,              # 2. extract day of week
    _add_is_weekend,               # 3. weekend flag   (needs day_of_week)
    _add_time_of_day_bucket,       # 4. rush/off-peak/overnight + is_rush_hour (needs pickup_hour)
    # _add_distance_x_time_of_day,   # 5. interaction    (needs trip_distance, time_of_day_bucket)
    # _add_pickup_zone_x_hour,       # 6. interaction    (needs PULocationID, pickup_hour)
    # _add_distance_x_rush_hour,     # 7. interaction    (needs trip_distance, is_rush_hour) #----------------------------------------------> לסדר את זה אחר כך' ולהתייעץ על זה עם וויקרם
]

FEATURE_TRANSFORMATION_STEPS = [
    _add_cyclical_hour_encoding,   # sin/cos of pickup_hour
]


# Raw columns kept in the baseline (no engineering applied)
BASELINE_FEATURE_COLS = [
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "day_of_week"
]


def run_baseline_pipeline(df, scaler=None, is_training=True, scaler_save_path=None):
    """
    Minimal pipeline: target variable + raw non-leaky columns only.

    No feature creation, no transformation, no feature store.
    Used as a controlled baseline to isolate the value added by engineering.
    The same scaling is applied as in the full pipeline so that the comparison
    is fair for linear models.

    Args:
        df           : Raw DataFrame (train or test split)
        scaler       : fitted StandardScaler  [test mode only]
        is_training  : bool
        scaler_save_path : optional path to save the fitted scaler

    Returns:
        (feature_df, scaler)   — scaler is None in test mode
    """
    if not is_training and scaler is None:
        raise ValueError("scaler must be provided when is_training=False.")

    df = df.copy()


    df = _add_pickup_hour(df) # needed for baseline features #----------------------------------------------> לסדר את זה אחר כך
    df = _add_day_of_week(df) # needed for baseline features #----------------------------------------------> לסדר את זה אחר כך
    df = _add_total_fare_amount(df) #----------------------------------------------> לסדר את זה אחר כך
    

    if is_training:#-----------------------------------------------------------------> ,לדבר עם וויקרם, אני הוספתי שזה רק לאימון כי זה יכול להיות שונה בנתונים חדשים, אבל אני לא בטוח אם זה נכון או לא
        n_before = len(df)
        df = df[df[TARGET_COL] > 0].reset_index(drop=True) # data quality guard: drop rows with non-positive target #----------------> לסדר את זה אחר כך
        if len(df) < n_before:
            print(f"  Dropped {n_before - len(df):,} rows with non-positive total fare amount")
            

    if is_training:
        scaler_created = False
        if scaler is None:
            scaler = StandardScaler()
            scaler_created = True
        df[SCALE_FEATURES] = scaler.fit_transform(df[SCALE_FEATURES])
        
        if scaler_created and scaler_save_path:
            Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_save_path)
            print(f"  Saved baseline scaler → {scaler_save_path}")
    else:
        df[SCALE_FEATURES] = scaler.transform(df[SCALE_FEATURES])

    keep = BASELINE_FEATURE_COLS + [TARGET_COL]
    return df[keep], scaler


def run_feature_pipeline(df, scaler=None, is_training=True, custom_creation_steps=None, scaler_save_path=None):
    """
    Execute the complete feature engineering pipeline.

    Training mode  (is_training=True):
        - Computes the target variable
        - Runs all creation and transformation steps
        - Fits the scaler on SCALE_FEATURES
        - Returns (feature_df, scaler)

    Inference/test mode  (is_training=False):
        - Runs the same creation and transformation steps
        - Applies the pre-fitted scaler (no refit)
        - Returns (feature_df, None)

    Args:
        df                    : Raw DataFrame (train or test split)
        scaler                : fitted StandardScaler  [test mode only]
        is_training           : bool
        custom_creation_steps : optional list of step functions to use instead
                                of FEATURE_CREATION_STEPS.  Pass a filtered
                                list from drift_mitigation.drop_drifted_feature_steps()
                                to retrain without drifted feature steps.
        scaler_save_path      : optional path to save the fitted scaler

    Returns:
        (feature_df, scaler)  — scaler is None in test mode
    """
    if not is_training and scaler is None:
        raise ValueError("scaler must be provided when is_training=False.")

    df = df.copy()

    # ── Step 1: Compute target variable ───────────────────────────────────────
    df = _add_total_fare_amount(df)

    # Remove rows where target is zero or negative (data quality guard)
    if is_training: #-----------------------------------------------------------------> לדבר עם וויקרם, אני הוספתי שזה רק לאימון כי זה יכול להיות שונה בנתונים חדשים, אבל אני לא בטוח אם זה נכון או לא
        n_before = len(df)
        df = df[df[TARGET_COL] > 0].reset_index(drop=True)
        if len(df) < n_before:
            print(f"  Dropped {n_before - len(df):,} rows with non-positive total fare amount")

    # ── Step 2: Feature creation (plug-and-play) ───────────────────────────────
    creation_steps = custom_creation_steps if custom_creation_steps is not None \
                     else FEATURE_CREATION_STEPS
    for step in creation_steps:
        df = step(df)

    # ── Step 3: Feature transformation ────────────────────────────────────────
    for step in FEATURE_TRANSFORMATION_STEPS:
        df = step(df)

    # ── Step 4: Feature scaling ────────────────────────────────────────────────
    if is_training:
        scaler_created = False
        if scaler is None:
            scaler = StandardScaler()
            scaler_created = True
        df[SCALE_FEATURES] = scaler.fit_transform(df[SCALE_FEATURES])
        
        if scaler_created and scaler_save_path:
            Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_save_path)
            print(f"  Saved feature scaler → {scaler_save_path}")
    else:
        df[SCALE_FEATURES] = scaler.transform(df[SCALE_FEATURES])

    # ── Step 5: Drop leaky and consumed columns ────────────────────────────────
    cols_to_drop = LEAKY_COLUMNS + _DATETIME_COLS_TO_DROP 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df, scaler if is_training else None
