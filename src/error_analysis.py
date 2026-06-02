"""
NYC TLC -- Per-Sample Error Analysis (Trip Fare Prediction)
===========================================================
Breaks down model errors by meaningful data segments so you can see
where the model struggles before asking why it struggles over time.
All labels and outputs are configured to use USD ($) instead of minutes.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Choose matplotlib backend safely
try:
    if 'ipykernel' not in sys.modules:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
except Exception:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


AIRPORT_ZONES = {1, 132, 138}  # EWR = 1, JFK = 132, LGA = 138
DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def build_error_df(y_test, y_pred, feature_df=None):
    """
    Build a per-sample error DataFrame in USD.
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error_df = pd.DataFrame({
        "actual":    y_test,
        "predicted": y_pred,
    })
    error_df["abs_error"] = np.abs(error_df["actual"] - error_df["predicted"])

    # Percentage error - exclude low fares (< $1.0) to avoid division near zero
    mask = error_df["actual"] >= 1.0
    error_df["pct_error"] = np.nan
    error_df.loc[mask, "pct_error"] = (
        error_df.loc[mask, "abs_error"] / error_df.loc[mask, "actual"] * 100
    )

    if feature_df is None:
        return error_df

    feat = feature_df.reset_index(drop=True)

    # ── Rush hour ─────────────────────────────────────────────────────────────
    if "is_rush_hour" in feat.columns:
        error_df["rush_hour_label"] = (
            feat["is_rush_hour"]
            .map({1: "Rush Hour", 0: "Off-Peak"})
            .fillna("Unknown")
        )

    # ── Day of week ───────────────────────────────────────────────────────────
    if "day_of_week" in feat.columns:
        error_df["day_name"] = feat["day_of_week"].map(DAY_NAMES).fillna("Unknown")

    # ── Airport vs non-airport ────────────────────────────────────────────────
    if "PULocationID" in feat.columns:
        error_df["trip_type"] = (
            feat["PULocationID"]
            .isin(AIRPORT_ZONES)
            .map({True: "Airport", False: "Non-Airport"})
        )

    # ── Distance bucket (quartile-based) ──────────────────────────────────────
    if "trip_distance" in feat.columns:
        try:
            error_df["distance_bucket"] = pd.qcut(
                feat["trip_distance"],
                q      = 4,
                labels = ["Q1 (short)", "Q2", "Q3", "Q4 (long)"],
                duplicates = "drop",
            )
        except Exception:
            pass

    # ── Time-of-day bucket ────────────────────────────────────────────────────
    if "time_of_day_bucket" in feat.columns:
        error_df["time_of_day"] = feat["time_of_day_bucket"].map(
            {0: "Overnight", 1: "Off-Peak", 2: "Rush Hour"}
        ).fillna("Unknown")

    return error_df


def plot_error_by_segment(error_df, group_col, title=None, output_dir=None):
    """
    Horizontal bar chart showing mean absolute error in USD ($) per group value.
    """
    if group_col not in error_df.columns:
        print(f"  Skipping segment plot -- '{group_col}' not in error_df.")
        return None

    grouped = (
        error_df.groupby(group_col, observed=True)["abs_error"]
        .agg(mean_mae="mean", n_trips="count")
        .sort_values("mean_mae", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(3, len(grouped) * 0.65)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grouped)))
    bars   = ax.barh(grouped.index.astype(str), grouped["mean_mae"], color=colors)

    # Value labels on bars in USD
    ax.bar_label(bars, fmt="$%.2f", padding=5, fontsize=9)

    # Trip count inside each bar
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(
            grouped["mean_mae"].min() * 0.05, i,
            f"n={row['n_trips']:,}",
            va="center", ha="left", fontsize=8,
            color="white", fontweight="bold",
        )

    ax.set_xlabel("Mean Absolute Error (dollars)")
    ax.set_title(title or f"MAE by {group_col}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"error_by_{group_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {path}")

    return fig


def plot_error_histogram_by_segment(error_df, group_col, title=None, output_dir=None, bins=40):
    """
    Layered histogram of absolute error in USD ($).
    """
    if group_col not in error_df.columns:
        print(f"  Skipping histogram -- '{group_col}' not in error_df.")
        return None

    groups = error_df[group_col].dropna().unique()

    # Clip to the visible x range (e.g. $100 max) before computing bin edges
    all_vals = error_df["abs_error"].dropna().clip(upper=100)
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10.colors

    for i, grp in enumerate(sorted(groups, key=str)):
        vals = error_df.loc[error_df[group_col] == grp, "abs_error"].dropna().clip(upper=100)
        ax.hist(
            vals,
            bins    = bin_edges,
            alpha   = 0.45,
            label   = f"{grp} (n={len(vals):,})",
            color   = colors[i % len(colors)],
            edgecolor = "none",
            density = True,
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Absolute Error (dollars)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Error Distribution by {group_col}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"error_hist_{group_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {path}")

    return fig


_SEGMENTS = [
    ("rush_hour_label",  "Rush Hour vs Off-Peak"),
    ("time_of_day",      "Time of Day Bucket"),
    ("distance_bucket",  "Trip Distance Quartile"),
    ("trip_type",        "Airport vs Non-Airport"),
    ("day_name",         "Day of Week"),
]


def run_error_analysis(X_test, y_test, model, output_dir=None):
    """
    Full error analysis pipeline for Trip Fare Prediction.
    """
    y_pred   = model.predict(X_test)
    error_df = build_error_df(y_test, y_pred, feature_df=X_test)

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n  {'Metric':<22} {'Value':>10}")
    print("  " + "-" * 35)
    print(f"  {'Mean Abs Error':<22} ${error_df['abs_error'].mean():>9.2f}")
    print(f"  {'Median Abs Error':<22} ${error_df['abs_error'].median():>9.2f}")
    print(f"  {'90th pct Error':<22} ${error_df['abs_error'].quantile(0.9):>9.2f}")
    print(f"  {'Max Error':<22} ${error_df['abs_error'].max():>9.2f}")
    valid_pct = error_df["pct_error"].dropna()
    if len(valid_pct):
        print(f"  {'Mean Pct Error':<22} {valid_pct.mean():>9.1f} %")

    # ── Per-segment breakdown ─────────────────────────────────────────────────
    figs = {}
    for col, title in _SEGMENTS:
        if col not in error_df.columns:
            continue
        if error_df[col].isna().all():
            continue

        print(f"\n  Error by {title}:")
        grouped = (
            error_df.groupby(col, observed=True)["abs_error"]
            .mean()
            .sort_values(ascending=False)
        )
        for group, mae in grouped.items():
            print(f"    {str(group):<25}  MAE = ${mae:.2f}")

        fig = plot_error_by_segment(
            error_df, col, title=f"MAE by {title}", output_dir=output_dir
        )
        if fig is not None:
            figs[col] = fig

        hist_fig = plot_error_histogram_by_segment(
            error_df, col, title=f"Error Distribution by {title}", output_dir=output_dir
        )
        if hist_fig is not None:
            figs[f"{col}_hist"] = hist_fig

    return error_df, figs
