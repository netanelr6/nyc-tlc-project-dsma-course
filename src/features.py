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
import pandas as pd
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
    "is_holiday",
    "haversine_distance",
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

BOROUGH_CENTROIDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR": (40.6895, -74.1745),
    "Unknown": (40.7831, -73.9712)
}

COORD_LOOKUP = {
    1: (40.6895, -74.1745),   # Newark Airport (EWR)
    132: (40.6413, -73.7781), # JFK Airport
    138: (40.7769, -73.8740), # LGA Airport
    43: (40.7829, -73.9654),  # Central Park
    230: (40.7580, -73.9855), # Times Square / Theatre District
    186: (40.7506, -73.9935), # Penn Station / Union Sq West
    162: (40.7527, -73.9772), # Grand Central / Murray Hill
    263: (40.7061, -74.0092), # Wall Street / Financial District
    100: (40.7538, -73.9904), # Garment District
    140: (40.7736, -73.9566), # Lenox Hill East
    141: (40.7781, -73.9515), # Lenox Hill West
    236: (40.7801, -73.9538), # Upper East Side North
    237: (40.7736, -73.9641), # Upper East Side South
    238: (40.8033, -73.9674), # Upper West Side North
    239: (40.7879, -73.9799), # Upper West Side South
    142: (40.7750, -73.9818), # Lincoln Square
    48: (40.7618, -73.9984),  # Clinton East
    246: (40.7513, -74.0044), # Chelsea West
    68: (40.7431, -73.9973),  # Chelsea East
    79: (40.7289, -73.9892),  # East Village
    107: (40.7324, -73.9873), # Gramercy
    170: (40.7454, -73.9782), # Murray Hill
    90: (40.7405, -74.0071),  # Flatiron / Union Sq
    234: (40.7402, -73.9896), # Union Sq
    113: (40.7325, -73.9973), # Greenwich Village North
    114: (40.7291, -74.0012), # Greenwich Village South
    125: (40.7247, -74.0084), # Hudson Sq
    249: (40.7346, -74.0062), # West Village
    231: (40.7202, -74.0102), # TriBeCa
    148: (40.7161, -73.9912), # Lower East Side
    224: (40.7214, -73.9798), # Stuyvesant Town/Peter Cooper Village
    262: (40.7757, -73.9431), # Yorkville East
    261: (40.7075, -74.0113), # World Trade Center
}

def _build_location_coord_mapping():
    zones_df = _get_zone_df()
    mapping = {}
    for _, row in zones_df.iterrows():
        try:
            loc_id = int(row["LocationID"])
            boro = str(row["Borough"])
            if loc_id in COORD_LOOKUP:
                mapping[loc_id] = COORD_LOOKUP[loc_id]
            else:
                mapping[loc_id] = BOROUGH_CENTROIDS.get(boro, BOROUGH_CENTROIDS["Unknown"])
        except Exception:
            pass
    return mapping

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 3956 # Radius of earth in miles
    return c * r

def _get_us_holidays(year):
    import datetime
    holidays_set = set()
    holidays_set.add(datetime.date(year, 1, 1))   # New Year's
    holidays_set.add(datetime.date(year, 6, 19))  # Juneteenth
    holidays_set.add(datetime.date(year, 7, 4))   # Independence Day
    holidays_set.add(datetime.date(year, 11, 11)) # Veterans Day
    holidays_set.add(datetime.date(year, 12, 25)) # Christmas
    
    def get_nth_weekday(y, m, n, w):
        first_day = datetime.date(y, m, 1)
        first_occur = 1 + (w - first_day.weekday()) % 7
        return datetime.date(y, m, first_occur + 7 * (n - 1))
        
    try:
        holidays_set.add(get_nth_weekday(year, 1, 3, 0)) # MLK
        holidays_set.add(get_nth_weekday(year, 2, 3, 0)) # Presidents'
        
        # Memorial Day (last Monday of May)
        memorial = datetime.date(year, 5, 31)
        memorial -= datetime.timedelta(days=memorial.weekday())
        holidays_set.add(memorial)
        
        holidays_set.add(get_nth_weekday(year, 9, 1, 0))  # Labor Day
        holidays_set.add(get_nth_weekday(year, 10, 2, 0)) # Columbus Day
        holidays_set.add(get_nth_weekday(year, 11, 4, 3)) # Thanksgiving
    except Exception:
        pass
    return holidays_set

_HOLIDAYS_CACHE = {}

def is_holiday_date(dt):
    import datetime
    if isinstance(dt, datetime.datetime):
        d = dt.date()
    else:
        d = dt
    y = d.year
    if y not in _HOLIDAYS_CACHE:
        _HOLIDAYS_CACHE[y] = _get_us_holidays(y)
    return 1 if d in _HOLIDAYS_CACHE[y] else 0

def _add_is_holiday(df):
    unique_dates = pd.to_datetime(df["tpep_pickup_datetime"]).dt.date.unique()
    holiday_map = {d: is_holiday_date(d) for d in unique_dates}
    df["is_holiday"] = df["tpep_pickup_datetime"].dt.date.map(holiday_map).astype("int8")
    return df

def _add_haversine_distance(df):
    coord_map = _build_location_coord_mapping()
    if not coord_map:
        df["haversine_distance"] = 0.0
        return df
    max_id = max(coord_map.keys())
    lat_arr = np.zeros(max_id + 1)
    lon_arr = np.zeros(max_id + 1)
    default_lat, default_lon = BOROUGH_CENTROIDS["Unknown"]
    
    lat_arr[:] = default_lat
    lon_arr[:] = default_lon
    
    for loc_id, (lat, lon) in coord_map.items():
        lat_arr[loc_id] = lat
        lon_arr[loc_id] = lon
        
    pu = df["PULocationID"].values
    do = df["DOLocationID"].values
    
    pu = np.clip(pu, 0, max_id)
    do = np.clip(do, 0, max_id)
    
    lat1 = lat_arr[pu]
    lon1 = lon_arr[pu]
    lat2 = lat_arr[do]
    lon2 = lon_arr[do]
    
    df["haversine_distance"] = haversine_vectorized(lon1, lat1, lon2, lat2)
    return df


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
    if "total_amount" in df.columns and "tip_amount" in df.columns:
        df[TARGET_COL] = df["total_amount"] - df["tip_amount"]
    else:
        df[TARGET_COL] = 0.0
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

_ZONES_DF_CACHED = None
_ZONES_CACHED = None
_MANHATTAN_SOUTH_96 = None
_MANHATTAN_ZONES = None

def _get_zone_df():
    global _ZONES_DF_CACHED
    if _ZONES_DF_CACHED is None:
        try:
            lookup_path = Path(__file__).parent.parent / "notebooks" / "taxi_zone_lookup.csv"
            if not lookup_path.exists():
                lookup_path = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
            
            if lookup_path.exists():
                _ZONES_DF_CACHED = pd.read_csv(lookup_path)
            else:
                _ZONES_DF_CACHED = pd.DataFrame(columns=["LocationID", "Borough", "Zone", "service_zone"])
        except Exception:
            _ZONES_DF_CACHED = pd.DataFrame(columns=["LocationID", "Borough", "Zone", "service_zone"])
    return _ZONES_DF_CACHED

def _get_zone_mappings():
    global _ZONES_CACHED, _MANHATTAN_SOUTH_96, _MANHATTAN_ZONES
    if _ZONES_CACHED is None:
        zones_df = _get_zone_df()
        if not zones_df.empty:
            _MANHATTAN_ZONES = set(zones_df[zones_df["Borough"] == "Manhattan"]["LocationID"].tolist())
            _MANHATTAN_SOUTH_96 = set(zones_df[(zones_df["Borough"] == "Manhattan") & (zones_df["service_zone"] == "Yellow Zone")]["LocationID"].tolist())
        else:
            _MANHATTAN_ZONES = {4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263}
            _MANHATTAN_SOUTH_96 = {4, 12, 13, 24, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144, 148, 151, 158, 161, 162, 163, 164, 170, 186, 194, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 246, 249, 261, 262, 263}
        _ZONES_CACHED = True
    return _MANHATTAN_ZONES, _MANHATTAN_SOUTH_96

def _add_geographic_features(df):
    """
    Map PULocationID and DOLocationID to Borough and service_zone,
    then apply One-Hot Encoding with predefined categories.
    """
    zones_df = _get_zone_df()
    
    # Define known categories
    BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR", "Unknown"]
    SERVICE_ZONES = ["Yellow Zone", "Boro Zone", "Airports", "EWR", "Unknown"]
    
    # Create mapping dictionaries
    boro_map = dict(zip(zones_df["LocationID"], zones_df["Borough"]))
    sz_map = dict(zip(zones_df["LocationID"], zones_df["service_zone"]))
    
    # Map pickup
    df["pickup_borough"] = df["PULocationID"].map(boro_map).fillna("Unknown")
    df["pickup_service_zone"] = df["PULocationID"].map(sz_map).fillna("Unknown")
    
    # Map dropoff
    df["dropoff_borough"] = df["DOLocationID"].map(boro_map).fillna("Unknown")
    df["dropoff_service_zone"] = df["DOLocationID"].map(sz_map).fillna("Unknown")
    
    # Ensure they are categorical types with fixed categories so get_dummies produces consistent columns
    df["pickup_borough"] = pd.Categorical(df["pickup_borough"], categories=BOROUGHS)
    df["pickup_service_zone"] = pd.Categorical(df["pickup_service_zone"], categories=SERVICE_ZONES)
    df["dropoff_borough"] = pd.Categorical(df["dropoff_borough"], categories=BOROUGHS)
    df["dropoff_service_zone"] = pd.Categorical(df["dropoff_service_zone"], categories=SERVICE_ZONES)
    
    # One-hot encode using pd.get_dummies
    df = pd.get_dummies(
        df, 
        columns=["pickup_borough", "pickup_service_zone", "dropoff_borough", "dropoff_service_zone"],
        prefix=["pickup_boro", "pickup_sz", "dropoff_boro", "dropoff_sz"],
        dtype="int8"
    )
    
    return df


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
    _add_is_holiday,               # 3.5. US federal holiday flag
    _add_haversine_distance,       # 3.6. Haversine distance
    _add_time_of_day_bucket,       # 4. rush/off-peak/overnight + is_rush_hour (needs pickup_hour)
    _add_domain_features,          # 5. domain-based TLC surcharges (JFK, LGA, EWR, congestion)
    _add_geographic_features,      # 6. geographic features (boroughs and service zones OHE)
    # _add_distance_x_time_of_day,   # 7. interaction    (needs trip_distance, time_of_day_bucket)
    # _add_pickup_zone_x_hour,       # 8. interaction    (needs PULocationID, pickup_hour)
    # _add_distance_x_rush_hour,     # 9. interaction    (needs trip_distance, is_rush_hour) #----------------------------------------------> לסדר את זה אחר כך' ולהתייעץ על זה עם וויקרם
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


def optimize_dataframe_dtypes(df):
    """Downcasts float and integer columns to save memory."""
    for col in df.columns:
        dt = df[col].dtype
        if dt == 'float64':
            df[col] = df[col].astype('float32')
        elif dt in ['int64', 'int32']:
            min_val, max_val = df[col].min(), df[col].max()
            if min_val >= -128 and max_val <= 127:
                df[col] = df[col].astype('int8')
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
        elif dt == 'object':
            df[col] = df[col].astype('category')
    return df


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
    res_df = df[keep].copy()
    res_df = optimize_dataframe_dtypes(res_df)
    return res_df, scaler


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

    df = optimize_dataframe_dtypes(df)

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

    df = optimize_dataframe_dtypes(df)
    return df, scaler if is_training else None
