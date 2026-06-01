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

SCALE_FEATURES = [
    "trip_distance",
    "pickup_hour",
    "day_of_week",
    "is_weekend",
    "time_of_day_bucket",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "distance_x_time_of_day",
    "pickup_zone_x_hour",
    "distance_x_rush_hour",
    "est_base_fare",
    "est_congestion_surcharge",
    "est_extra",
    "est_airport_fee",
    "est_total_fare_without_tolls"
]


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

_ZONES_CACHED = None
_MANHATTAN_SOUTH_96 = None
_MANHATTAN_ZONES = None

def _get_zone_mappings():
    global _ZONES_CACHED, _MANHATTAN_SOUTH_96, _MANHATTAN_ZONES
    if _ZONES_CACHED is None:
        try:
            lookup_path = Path(__file__).parent.parent / "notebooks" / "taxi_zone_lookup.csv"
            if not lookup_path.exists():
                lookup_path = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
            
            if lookup_path.exists():
                zones_df = pd.read_csv(lookup_path)
                _MANHATTAN_ZONES = set(zones_df[zones_df["Borough"] == "Manhattan"]["LocationID"].tolist())
                _MANHATTAN_SOUTH_96 = set(zones_df[(zones_df["Borough"] == "Manhattan") & (zones_df["service_zone"] == "Yellow Zone")]["LocationID"].tolist())
            else:
                raise FileNotFoundError()
        except Exception:
            _MANHATTAN_ZONES = {4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263}
            _MANHATTAN_SOUTH_96 = {4, 12, 13, 24, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144, 148, 151, 158, 161, 162, 163, 164, 170, 186, 194, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 246, 249, 261, 262, 263}
        _ZONES_CACHED = True
    return _MANHATTAN_ZONES, _MANHATTAN_SOUTH_96

def _add_domain_features(df):
    """
    Add domain-specific fare components estimated solely from non-leaky input features.
    - est_base_fare: Flat $70 for JFK-Manhattan trips, else $3.00 + $3.50 * distance
    - est_congestion_surcharge: $2.50 if pickup or dropoff is in Manhattan south of 96
    - est_extra: $1.00 overnight surcharge, $2.50 rush hour surcharge ($5.00 for JFK trips)
    - est_airport_fee: $2.50 for LaGuardia / JFK pickups
    - est_improvement_surcharge: $1.00 flat
    - est_mta_tax: $0.50 flat
    - est_total_fare_without_tolls: sum of the above
    """
    manhattan_zones, manhattan_south_96 = _get_zone_mappings()
    
    pu = df["PULocationID"]
    do = df["DOLocationID"]
    dist = df["trip_distance"]
    hour = df["pickup_hour"]
    day = df["day_of_week"]
    
    # 1. JFK Flat Rate check
    is_pu_jfk = pu == 132
    is_do_jfk = do == 132
    is_pu_man = pu.isin(manhattan_zones)
    is_do_man = do.isin(manhattan_zones)
    is_jfk_trip = (is_pu_jfk & is_do_man) | (is_do_jfk & is_pu_man)
    
    # Base fare
    df["est_base_fare"] = np.where(is_jfk_trip, 70.0, 3.0 + 3.50 * dist)
    
    # 2. Congestion Surcharge
    is_pu_congestion = pu.isin(manhattan_south_96)
    is_do_congestion = do.isin(manhattan_south_96)
    df["est_congestion_surcharge"] = np.where(is_pu_congestion | is_do_congestion, 2.50, 0.0)
    
    # 3. Extra (Night / Rush Hour)
    is_overnight = (hour >= 20) | (hour < 6)
    is_rush = (day < 5) & (hour >= 16) & (hour < 20)
    
    extra = np.zeros(len(df))
    extra[is_overnight] += 1.0
    extra[is_rush] += np.where(is_jfk_trip[is_rush], 5.0, 2.5)
    df["est_extra"] = extra
    
    # 4. Airport Fee ($2.50 access fee for JFK/LGA pickups + EWR/LGA surcharges)
    airport_fee = np.zeros(len(df))
    # $2.50 access fee for pickups at JFK (132) and LGA (138)
    airport_fee[pu.isin({132, 138})] += 2.50
    # $5.00 surcharge for LaGuardia trips (pickup or dropoff)
    airport_fee[pu.isin({138}) | do.isin({138})] += 5.00
    # $20.00 surcharge for Newark trips
    airport_fee[pu.isin({1}) | do.isin({1})] += 20.00
    df["est_airport_fee"] = airport_fee
    
    # 5. Fixed surcharges
    df["est_improvement_surcharge"] = 1.0
    df["est_mta_tax"] = 0.50
    
    # 6. Sum total estimate (without tolls)
    df["est_total_fare_without_tolls"] = (
        df["est_base_fare"] + 
        df["est_congestion_surcharge"] + 
        df["est_extra"] + 
        df["est_airport_fee"] + 
        df["est_improvement_surcharge"] + 
        df["est_mta_tax"]
    )
    
    return df

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
    _add_domain_features,          # 5. domain-based TLC surcharges (JFK, LGA, EWR, congestion)
    # _add_distance_x_time_of_day,   # 6. interaction    (needs trip_distance, time_of_day_bucket)
    # _add_pickup_zone_x_hour,       # 7. interaction    (needs PULocationID, pickup_hour)
    # _add_distance_x_rush_hour,     # 8. interaction    (needs trip_distance, is_rush_hour) #----------------------------------------------> לסדר את זה אחר כך' ולהתייעץ על זה עם וויקרם
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
            

    scale_cols = [c for c in SCALE_FEATURES if c in df.columns]
    if is_training:
        scaler_created = False
        if scaler is None:
            scaler = StandardScaler()
            scaler_created = True
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        if scaler_created and scaler_save_path:
            Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_save_path)
            print(f"  Saved baseline scaler -> {scaler_save_path}")
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

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
    scale_cols = [c for c in SCALE_FEATURES if c in df.columns]
    if is_training:
        scaler_created = False
        if scaler is None:
            scaler = StandardScaler()
            scaler_created = True
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        if scaler_created and scaler_save_path:
            Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_save_path)
            print(f"  Saved feature scaler -> {scaler_save_path}")
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    # ── Step 5: Drop leaky and consumed columns ────────────────────────────────
    cols_to_drop = LEAKY_COLUMNS + _DATETIME_COLS_TO_DROP 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df, scaler if is_training else None
