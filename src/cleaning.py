"""
NYC TLC Yellow Taxi -- Data Cleaning
======================================

Applies targeted NaN handling and basic data quality filters to the raw
parquet before it enters the feature engineering pipeline.

Design principles
-----------------
- Drop rows only when a column is critical and cannot be reasonably imputed.
- Fill NaNs with sensible domain defaults when the column is non-critical.
- Filter logically invalid rows (e.g., negative amounts or zero distance).
- Keep only columns needed for downstream feature engineering.
"""

import pandas as pd
from pathlib import Path


# ── Configuration & Strategies ────────────────────────────────────────────────

NAN_FILL_STRATEGIES = {
    "RatecodeID":         1,      # 1 = standard metered rate (most common)
    "store_and_fwd_flag": "N",    # N = data transmitted in real time
    "Airport_fee":        0.0,    # no airport fee for non-airport trips
}

# Columns that are critical — rows with nulls here are dropped
CRITICAL_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "total_amount",
    "tip_amount"
]

# Columns to retain after cleaning (Passed to features.py)
RELEVANT_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "RatecodeID", # For further analysis, not training
    "payment_type", # For further analysis, not training
    "total_amount",  # Kept to calculate target variable later
    "tip_amount"     # Kept to calculate target variable later
]


# ── Cleaning functions ────────────────────────────────────────────────────────



def drop_critical_nulls(df):
    """Drops rows missing essential temporal, spatial, or monetary data."""
    before = len(df)
    df = df.dropna(subset=[c for c in CRITICAL_COLUMNS if c in df.columns])
    dropped = before - len(df)
    if dropped:
        print(f"  drop_critical_nulls     : dropped {dropped:,} rows")
    else:
        print(f"  drop_critical_nulls  : no rows dropped")
    return df

def fill_non_critical_nulls(df):
    """
    Fill NaNs in non-critical columns using domain-appropriate defaults
    defined in NAN_FILL_STRATEGIES.
    """
    filled_report = []
    for col, fill_value in NAN_FILL_STRATEGIES.items():
        if col not in df.columns:
            continue
        n_null = df[col].isna().sum()
        if n_null > 0:
            df[col] = df[col].fillna(fill_value)
            filled_report.append(f"{col} ({n_null:,} → {fill_value!r})")

    if filled_report:
        print(f"  fill_non_critical_nulls : filled — {', '.join(filled_report)}")
    else:
        print(f"  fill_non_critical_nulls : no nulls to fill")
    return df


def drop_pre_december_2023(df):
    """
    Drop any trip whose pickup datetime is before December 2023.
    The dataset should only contain trips from December 2023 onwards.
    """
    before = len(df)
    df = df[df["tpep_pickup_datetime"] >= "2023-12-01"]
    dropped = before - len(df)
    if dropped:
        print(f"  drop_pre_december_2023  : dropped {dropped:,} rows")
    else:
        print(f"  drop_pre_december_2023  : no rows dropped")
    return df


def select_relevant_columns(df):
    """
    Retain only the columns required for the ETA prediction problem.
    All other columns are dropped as irrelevant.
    """
    cols = [c for c in RELEVANT_COLUMNS if c in df.columns]
    dropped_cols = [c for c in df.columns if c not in cols]
    if dropped_cols:
        print(f"  select_relevant_columns : dropped {len(dropped_cols)} columns — {dropped_cols}")
    return df[cols]

def drop_remaining_nulls(df):
    """
    Safety net: drop any rows that still contain NaNs after the targeted
    fill step.  In a healthy dataset this should remove zero rows.
    """
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  drop_remaining_nulls : dropped {dropped:,} unexpected null rows")
    else:
        print(f"  drop_remaining_nulls : no remaining nulls")
    return df


def filter_valid_trips(df):
    """
    Removes logically invalid trips:
    - total_amount must be strictly positive (filters out refunds/errors).
    - trip_distance must be strictly positive.
    """
    before = len(df)
    
    df = df[(df["total_amount"] > 0) & (df["trip_distance"] > 0)]
            
    dropped = before - len(df)
    if dropped:
        print(f"  filter_valid_trips      : dropped {dropped:,} invalid rows")
    return df

# ── Main entry point ──────────────────────────────────────────────────────────

def clean_dataframe(df):
    """
    Executes the sequential data cleaning pipeline.
    """
    print(f"  Input  rows : {len(df):,}")
    df = drop_critical_nulls(df)
    df = drop_pre_december_2023(df)
    df = fill_non_critical_nulls(df)
    df = select_relevant_columns(df)
    df = drop_remaining_nulls(df)
    df = filter_valid_trips(df)
    print(f"  Output rows : {len(df):,}")
    return df.reset_index(drop=True)


def clean_parquet(input_path, output_path=None, sample_size=None):
    """
    Loads, cleans, and optionally saves a parquet dataset.
    
    Args:
        input_path  (str): Path to the raw .parquet file.
        output_path (str): Destination for the cleaned file (optional).
        sample_size (int): If provided, processes only the top N rows for rapid testing.
    
    Returns:
        pd.DataFrame  Cleaned DataFrame.
    """


    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Modularity: subset the data if testing
    if sample_size is not None:
        print(f"  [TEST MODE] Sampling {sample_size:,} rows for rapid execution.")
        df = df.head(sample_size)

    df_clean = clean_dataframe(df)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(output_path, index=False)
        print(f"  Saved cleaned file → {output_path}\n")

    return df_clean


if __name__ == "__main__":
    # Test the cleaning pipeline on a tiny subset
    input_file = "data/raw/yellow_tripdata_2024-01.parquet"
    output_file = "data/processed/yellow_tripdata_2024-01_clean.parquet"
    
    if Path(input_file).exists():
        clean_parquet(input_file, output_file, sample_size=10000)
    else:
        print(f"Cannot find {input_file}. Please run download_data.py first.")