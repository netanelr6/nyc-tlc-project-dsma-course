"""
NYC TLC Yellow Taxi -- Custom Data Validation
=============================================
Applies data quality gates to raw datasets using flexible logic 
and thresholding parameters. Aligned with course principles.
"""

import pandas as pd


# ---------------------------------------------------------------------------
#  check helper functions
# ---------------------------------------------------------------------------

def _check_required_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    passed  = len(missing) == 0
    return {
        "name":   "required_columns_present",
        "column": "TABLE",
        "passed": passed,
        "detail": "OK" if passed else f"missing columns: {missing}",
    }

def _check_row_count(df, min_value, max_value):
    n      = len(df)
    passed = min_value <= n <= max_value
    return {
        "name":   f"row_count_between({min_value:,}, {max_value:,})",
        "column": "TABLE",
        "passed": passed,
        "detail": "OK" if passed else f"actual row count: {n:,}",
    }


def check_not_null(df, col, mostly=1.0):
    """
    Verifies if a column contains non-null values up to a specific ratio threshold.
    """
    n_null = df[col].isna().sum()
    actual_valid_ratio = 1.0 - (n_null / len(df))
    passed = actual_valid_ratio >= mostly
    
    detail = 'OK' if passed else f'Failed threshold: {actual_valid_ratio:.2%} valid < {mostly:.2%} (Found {n_null:,} nulls)'
    return {"name": "not_null", "column": col, "passed": passed, "detail": detail}


def _check_dtype_datetime(df, col):
    is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
    return {
        "name":   "dtype_is_datetime",
        "column": col,
        "passed": is_dt,
        "detail": "OK" if is_dt else f"actual dtype: {df[col].dtype}",
    }


def check_between(df, col, min_value=None, max_value=None, mostly=1.0):
    """
    Verifies if values within a column fall within a specified range [min_value, max_value].
    Allows a percentage of exceptions defined by the 'mostly' parameter.
    """
    mask = pd.Series(True, index=df.index)
    if min_value is not None:
        mask &= df[col] >= min_value
    if max_value is not None:
        mask &= df[col] <= max_value
        
    actual_valid_ratio = mask.mean()
    passed = actual_valid_ratio >= mostly
    
    detail = 'OK' if passed else f'Failed threshold: {actual_valid_ratio:.2%} within range < {mostly:.2%}'
    return {"name": f"between[{min_value},{max_value}]", "column": col, "passed": passed, "detail": detail}


def check_pair_ordering(df, col_before, col_after):
    """
    Verifies chronological or logical ordering between two columns.
    Rule: col_after must be strictly greater than or equal to col_before.
    """
    valid_mask = df[col_before].notna() & df[col_after].notna()
    violating_rows = (df[col_after] < df[col_before]) & valid_mask
    fail_rate = violating_rows.mean()
    passed = fail_rate == 0
    
    detail = 'OK' if passed else f'{fail_rate:.2%} of rows violate ordering constraint ({col_after} < {col_before})'
    return {"name": f"{col_after}_geq_{col_before}", "column": f"{col_before}, {col_after}", "passed": passed, "detail": detail}


def check_cash_tip_logic(df):
    """
    Custom Business Logic Check:
    According to the NYC TLC data dictionary, cash tips are not recorded.
    If payment_type == 2 (Cash) and tip_amount > 0, it represents a data anomaly.
    """
    cash_mask = df["payment_type"] == 2
    invalid_tips = (df["tip_amount"] > 0) & cash_mask
    fail_rate = invalid_tips.mean()
    
    # We tolerate a tiny fraction of logging anomalies, let's say less than 0.5%
    passed = fail_rate <= 0.005
    detail = 'OK' if passed else f'{fail_rate:.2%} of rows erroneously logged tips for cash payments'
    return {"name": "cash_tip_anomaly_check", "column": "payment_type, tip_amount", "passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
#  Main Pipeline Entry Point
# ---------------------------------------------------------------------------

def validate_nyc_taxi_data(df):
    """
    Executes a suite of validation rules on the dataset and returns a summary.
    """
    results = []
    
    # 1. Temporal Constraints
    results.append(check_not_null(df, "tpep_pickup_datetime"))
    results.append(check_not_null(df, "tpep_dropoff_datetime"))
    results.append(check_pair_ordering(df, "tpep_pickup_datetime", "tpep_dropoff_datetime"))
    
    # 2. Spatial Constraints
    results.append(check_not_null(df, "PULocationID"))
    results.append(check_between(df, "PULocationID", min_value=1, max_value=265))
    results.append(check_not_null(df, "DOLocationID"))
    results.append(check_between(df, "DOLocationID", min_value=1, max_value=265))
    
    # 3. Trip Metrics & Business Logic
    results.append(check_between(df, "trip_distance", min_value=0.0))
    results.append(check_cash_tip_logic(df))
    
    # 4. Monetary Targets (Crucial for Fare Prediction task)
    results.append(check_not_null(df, "total_amount"))
    results.append(check_between(df, "total_amount", min_value=0.0))
    results.append(check_between(df, "fare_amount", min_value=3.0, mostly=0.95)) # Base flag-drop rate in NYC is $3.0
    
    # Aggregate success status
    success = all(r["passed"] for r in results)
    
    return {"success": success, "results": results}