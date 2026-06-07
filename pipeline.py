"""
NYC TLC Trip Fare Prediction — Full Pipeline
=============================================

Run this script to execute the complete end-to-end pipeline structured by Acts:

  Act 1 — Data Exploration & Prep
    - Step 1.1: Data download configuration (downloads raw data if missing)
    - Step 1.2: Data Exploration (prints descriptive stats and plots)
    - Step 1.3: Data Validation (runs validation checks)
    - Step 1.4: Data Cleaning (Raw Train & Test -> Cleaned Parquets)
    - Step 1.5: Feature Engineering (saves baseline & engineered features to parquets)

  Act 2 — Model Building & Tuning
    - Step 3 & 4: Model Training and Baseline/Engineered Evaluations
    - Step 5: Head-to-Head Comparison
    - Step 7: Hyperparameter Tuning (Sweeps & Retraining)

  Act 3 — Model Evaluation & Experiments
    - Step 6: Champion & Feature Importance
    - Step 8: Error Analysis (sample-level errors, breakdowns in USD)
    - Step 9: Drift Detection (PSI + KS on Train vs Test, label/concept drift)

  Act 4 — Productionize Model
    - Step 9.1: Evidently AI Drift Detection (Train vs Dec 2024)
    - Step 10: Drift Mitigation & W&B Artifact Versioning

Modularity note
---------------
Each concern lives in its own src/ module. pipeline.py is pure orchestration:
it calls modular Act functions, passes outputs between them, and logs results to W&B
through ExperimentTracker. No business logic lives here.
"""

import argparse
import numpy as np
np.float_ = np.float64
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
import time

# Load W&B credentials and other environment variables from local .env
load_dotenv()

from src.download_data            import run_download_pipeline
from src.validation               import validate_nyc_taxi_data
from src.cleaning                 import clean_parquet, clean_dataframe
from src.features                 import run_feature_pipeline, run_baseline_pipeline, TARGET_COL
from src.models                   import train_all_models, load_model, CANDIDATE_MODELS
from src.evaluation               import evaluate_all_models, select_champion, plot_feature_importance
from src.experiment_tracking      import (ExperimentTracker, log_monthly_drift_run,
                                          configure_wandb, log_eval_run)
from src.tuning                   import (RANDOM_SEARCH_CONFIG, GRID_SEARCH_CONFIGS,
                                          run_wandb_sweep, retrain_best_model,
                                          get_stratified_tuning_sample)
# from src.error_analysis           import run_error_analysis
# from src.drift_detection          import (build_drift_report, detect_label_drift,
#                                           detect_concept_drift, plot_feature_distributions,
#                                           plot_label_drift_distribution)
# from src.drift_detection_evidently import (run_evidently_drift_report, parse_drift_results,
#                                            run_evidently_concept_drift_report,
#                                            parse_concept_drift_results,
#                                            select_mitigation_strategy)
# from src.drift_mitigation         import mitigate, plot_mitigation_comparison
# from src.versioning               import log_data_artifact, log_model_artifact, log_feature_artifact


# ── Path configuration ────────────────────────────────────────────────────────

RAW_TRAIN_DIR                 = "data/raw/train"
RAW_TEST_DIR                  = "data/raw/test"
TRAIN_CLEANED_PARQUET         = "data/processed/DF_test_2024_2025_cleaned.parquet"
TEST_CLEANED_PARQUET          = "data/processed/Df_test_2026_cleaned.parquet"
PROCESSED_DIR                 = "data/processed"
MODEL_DIR_BASELINE            = "models/baseline"
MODEL_DIR_ENGINEERED          = "models/engineered"
MODEL_DIR_TUNED               = "models/tuned"
MODEL_DIR_MITIGATED           = "models/mitigated"
PLOTS_DIR                     = "outputs/plots"
DRIFT_RAW_PARQUET             = "data/yellow_tripdata_2024-12.parquet"

# Feature and scaler storage paths
BASELINE_TRAIN_PARQUET        = "data/processed/baseline_train.parquet"
BASELINE_TEST_PARQUET         = "data/processed/baseline_test.parquet"
ENGINEERED_TRAIN_PARQUET      = "data/processed/engineered_train.parquet"
ENGINEERED_TEST_PARQUET       = "data/processed/engineered_test.parquet"
SCALER_SAVE_PATH_BSSELINE     = "data/feature_stores/baseline_scaler.pkl"
SCALER_SAVE_PATH_ENGINEERD    = "data/feature_stores/engineered_scaler.pkl"


# ── W&B configuration ─────────────────────────────────────────────────────────

WANDB_PROJECT = "dsma-nyc-tlc-taxi"
WANDB_ENTITY  = "dsma_fit_happens"
TUNING_SAMPLE_SIZE = 2_500_000


# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def _print_small_header(title):
    print("\n" + "*" * 30)
    print(title)
    print("*" * 30)


def _comparison_table(baseline_results, engineered_results):
    merged = baseline_results[["model", "mae", "rmse"]].merge(
        engineered_results[["model", "mae", "rmse"]],
        on="model", suffixes=("_baseline", "_engineered"),
    )
    merged["mae_improvement_%"] = (
        (merged["mae_baseline"] - merged["mae_engineered"]) / merged["mae_baseline"] * 100
    ).round(1)

    print(f"\n  {'Model':<25} {'Baseline MAE':>13} {'Engineered MAE':>15} {'Improvement':>12}")
    print("  " + "-" * 68)
    for _, row in merged.iterrows():
        print(
            f"  {row['model']:<25} "
            f"${row['mae_baseline']:>12.2f} "
            f"${row['mae_engineered']:>14.2f} "
            f"{row['mae_improvement_%']:>11.1f}%"
        )
    return merged

# ================================================================================
# ==========================   Act Functions  ==================================
# ================================================================================

# ── Act 1 ─────────────────────────────────────────────────────────────────────

def run_act1(sample_size=None):
    """
    Act 1 — Data Exploration & Prep
    - Step 1.1: Data download configuration (downloads raw data if missing)
    - Step 1.2: Data Exploration (prints descriptive stats and plots)
    - Step 1.3: Data Validation (runs validation checks)
    - Step 1.4: Data Cleaning (Raw Train & Test -> Cleaned Parquets)
    - Step 1.5: Feature Engineering (saves baseline & engineered features to parquets)
    """
    _print_header("ACT 1 — Data Exploration & Prep")
    act1_start = time.time()

    # <> 1.1 Download data if missing <><><><><><><>
    _print_small_header("1.1 Data Download")
    t0 = time.time()
    print("Running download pipeline check...")
    run_download_pipeline()
    print(f"\033[33m  [Duration] Data download finished in {time.time() - t0:.2f}s\033[0m")


    # <> 1.2 Fundamentals <><><><><><><><><><><><><><><><><><><>
    _print_small_header("1.2 Data Fundamentals")
    t0 = time.time()

    # Load raw datasets
    print("\nLoading raw datasets...")
    df_raw_train = pd.read_parquet(RAW_TRAIN_DIR, engine='pyarrow')
    df_raw_test  = pd.read_parquet(RAW_TEST_DIR, engine='pyarrow')
    if sample_size is not None:
        print(f"  [SMOKE TEST] Sampling {sample_size:,} rows from train and test raw datasets.")
        df_raw_train = df_raw_train.sample(n=min(sample_size, len(df_raw_train)), random_state=42).reset_index(drop=True)
        df_raw_test  = df_raw_test.sample(n=min(sample_size, len(df_raw_test)), random_state=42).reset_index(drop=True)


    print(f"  Train Data: Loaded {len(df_raw_train):,} rows x {df_raw_train.shape[1]} columns")
    print(f"  Test Data:  Loaded {len(df_raw_test):,} rows x {df_raw_test.shape[1]} columns")
    print("\n  Train Data Columns and Dtypes:")
    for col, dtype in zip(df_raw_train.columns, df_raw_train.dtypes):
        print(f"    - {col:<25} : {dtype}")
    print("\n  Train Data Summary Statistics (Transposed):")
    print(df_raw_train.describe().T.to_string())
    print(f"\033[33m  [Duration] Data fundamentals finished in {time.time() - t0:.2f}s\033[0m")

    # <> 1.3 Validation <><><><><><><><><><><><><><><><><><><>
    _print_small_header("1.3 Data Validation")
    t0 = time.time()

    validation_report = validate_nyc_taxi_data(df_raw_train)
    passed_n = sum(r['passed'] for r in validation_report['results'])
    total_n  = len(validation_report['results'])
    status   = 'ALL PASSED' if validation_report['success'] else 'SOME CHECKS FAILED'
    
    print(f"  Result: {status} ({passed_n}/{total_n} checks passed)")
    print("\n  {:<4} {:<44} {:<48} {}".format('#', 'Column', 'Check', 'Result'))
    print("  " + "-" * 105)
    for i, r in enumerate(validation_report['results'], 1):
        col    = r['column']
        name   = r['name']
        status_str = "\033[92mOK\033[0m  " if r['passed'] else "\033[91mFAIL\033[0m"
        print("  {:<4} {:<44} {:<48} {}".format(i, col, name, status_str))
        if not r['passed']:
            detail = r['detail']
            print(f"     >> \033[91m{detail}\033[0m")
    print(f"\033[33m  [Duration] Data validation finished in {time.time() - t0:.2f}s\033[0m")

    # <> 1.4 Clean datasets <><><><><><><><><><><><><><><><><><><>
    _print_small_header("1.4 Data Cleaning")
    t0 = time.time()
    
    print("\nCleaning training datasets...")
    train_clean = clean_parquet(df_raw_train, TRAIN_CLEANED_PARQUET, is_train=True)
    print("\nCleaning test datasets...")
    test_clean  = clean_parquet(df_raw_test, TEST_CLEANED_PARQUET, is_train=False)

    print(f"  Training set: {len(train_clean):,} rows")
    print(f"  Test set:     {len(test_clean):,} rows")
    print(f"\033[33m  [Duration] Data cleaning finished in {time.time() - t0:.2f}s\033[0m")

    # <> 1.5 Prepare features <><><><><><><><><><><><><><><><><><><>
    _print_small_header("1.5 Prepare Features")
    t0 = time.time()
    
    print("\nRunning baseline feature engineering on clean datasets...")
    baseline_train, baseline_scaler = run_baseline_pipeline(train_clean, is_training=True)
    baseline_test, _ = run_baseline_pipeline(test_clean, scaler=baseline_scaler, is_training=False)

    print("\nRunning engineered feature engineering on clean datasets...")
    eng_train, eng_scaler = run_feature_pipeline(train_clean, is_training=True)
    eng_test, _ = run_feature_pipeline(test_clean, scaler=eng_scaler, is_training=False)

    # Save scales and feature stores
    Path("data/feature_stores").mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR_ENGINEERED).mkdir(parents=True, exist_ok=True)
    
    joblib.dump(baseline_scaler, SCALER_SAVE_PATH_BSSELINE)
    joblib.dump(eng_scaler, SCALER_SAVE_PATH_ENGINEERD)
    joblib.dump(eng_scaler, Path(MODEL_DIR_ENGINEERED) / "scaler.pkl")
    print(f"  Baseline scaler saved -> {SCALER_SAVE_PATH_BSSELINE}")
    print(f"  Engineered scaler saved -> {SCALER_SAVE_PATH_ENGINEERD}")

    # Save features as parquet files
    print("\nSaving feature sets to disk...")
    baseline_train.to_parquet(BASELINE_TRAIN_PARQUET, index=False)
    baseline_test.to_parquet(BASELINE_TEST_PARQUET, index=False)
    eng_train.to_parquet(ENGINEERED_TRAIN_PARQUET, index=False)
    eng_test.to_parquet(ENGINEERED_TEST_PARQUET, index=False)
    print(f"  Baseline features saved -> {BASELINE_TRAIN_PARQUET}, {BASELINE_TEST_PARQUET}")
    print(f"  Engineered features saved -> {ENGINEERED_TRAIN_PARQUET}, {ENGINEERED_TEST_PARQUET}")
    print(f"\033[33m  [Duration] Features preparation finished in {time.time() - t0:.2f}s\033[0m")

    print(f"\n\033[33m>>> [Duration] ACT 1 completed in {time.time() - act1_start:.2f}s\033[0m")




# ── Act 2 ─────────────────────────────────────────────────────────────────────

def run_act2(wandb_project, wandb_entity, tuning_sample_size=100000):
    """
    Act 2 — Model Building & Tuning
    - Steps 3 & 4: Model Training and Baseline/Engineered Evaluations
    - Step 5: Head-to-Head Comparison
    - Step 7: Hyperparameter Tuning (Sweeps & Retraining)
    """
    _print_header("ACT 2 — Model Building & Tuning")
    act2_start = time.time()

    # Load datasets from disk
    t0 = time.time()
    print("Loading feature sets from disk...")
    baseline_train = pd.read_parquet(BASELINE_TRAIN_PARQUET)
    baseline_test  = pd.read_parquet(BASELINE_TEST_PARQUET)
    eng_train      = pd.read_parquet(ENGINEERED_TRAIN_PARQUET)
    eng_test       = pd.read_parquet(ENGINEERED_TEST_PARQUET)
    print(f"\033[33m  [Duration] Loaded feature sets in {time.time() - t0:.2f}s\033[0m")

    # ── Experiment A: Baseline Models (Commented out by request)
    # print("\n--- Training Baseline Models ---")
    # t0 = time.time()
    # X_train_base = baseline_train.drop(columns=[TARGET_COL])
    # y_train_base = baseline_train[TARGET_COL]
    # train_all_models(X_train_base, y_train_base, MODEL_DIR_BASELINE)
    #
    # X_test_base = baseline_test.drop(columns=[TARGET_COL])
    # y_test_base = baseline_test[TARGET_COL]
    # print("\nBaseline model results:")
    # baseline_results = evaluate_all_models(X_test_base, y_test_base, MODEL_DIR_BASELINE)
    # print(f"\033[33m  [Duration] Baseline models training & evaluation finished in {time.time() - t0:.2f}s\033[0m")
    #
    # for _, row in baseline_results.iterrows():
    #     log_eval_run(
    #         model_name=row["model"],
    #         metrics={"rmse": float(row["rmse"]), "mae": float(row["mae"]), "mape": float(row["mape"])},
    #         config={"features": "baseline"},
    #         group="baseline",
    #         project=wandb_project,
    #         entity=wandb_entity
    #     )

    # ── Experiment B: Engineered Models
    print("\n--- Training Engineered Models ---")
    t0 = time.time()
    X_train_eng = eng_train.drop(columns=[TARGET_COL])
    y_train_eng = eng_train[TARGET_COL]
    train_all_models(X_train_eng, y_train_eng, MODEL_DIR_ENGINEERED)

    X_test_eng = eng_test.drop(columns=[TARGET_COL])
    y_test_eng = eng_test[TARGET_COL]
    print("\nEngineered model results:")
    engineered_results = evaluate_all_models(X_test_eng, y_test_eng, MODEL_DIR_ENGINEERED)
    print(f"\033[33m  [Duration] Engineered models training & evaluation finished in {time.time() - t0:.2f}s\033[0m")

    for _, row in engineered_results.iterrows():
        log_eval_run(
            model_name=row["model"],
            metrics={"rmse": float(row["rmse"]), "mae": float(row["mae"]), "mape": float(row["mape"])},
            config={"features": "engineered"},
            group="engineered",
            project=wandb_project,
            entity=wandb_entity
        )

    # ── Head-to-head Comparison (Commented out by request)
    # print("\n--- Head-to-Head Comparison ---")
    # comparison = _comparison_table(baseline_results, engineered_results)
    # best_model_row = comparison.loc[comparison["mae_improvement_%"].idxmax()]
    # print(f"\n  Largest gain : {best_model_row['model']} "
    #       f"improved by {best_model_row['mae_improvement_%']:.1f}% with feature engineering")

    # ── Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning Sweeps ---")
    t0 = time.time()
    
    # Perform Stratified Subsampling for sweeps
    print(f"  Preparing stratified sample of {tuning_sample_size:,} rows for tuning sweeps...")
    X_train_tuning, y_train_tuning = get_stratified_tuning_sample(
        X_train_eng, y_train_eng, sample_size=tuning_sample_size, random_state=42
    )
    print(f"  Stratified sample ready: {len(X_train_tuning):,} rows.")

    print("  Phase 1 — Random Search (both model families, wide grid)")
    random_sweep_id, best_random_config, best_random_mae = run_wandb_sweep(
        X_train      = X_train_tuning,
        y_train      = y_train_tuning,
        sweep_config = RANDOM_SEARCH_CONFIG,
        project      = wandb_project,
        entity       = wandb_entity,
        n_runs       = 12,
    )
    winning_family = best_random_config.get("model_type", "random_forest")
    print(f"\n  Random search complete. Best family: {winning_family}")

    print(f"\n  Phase 2 — Grid Search ({winning_family}, narrow grid, all combinations)")
    grid_config = GRID_SEARCH_CONFIGS[winning_family]
    grid_sweep_id, best_grid_config, best_grid_mae = run_wandb_sweep(
        X_train      = X_train_tuning,
        y_train      = y_train_tuning,
        sweep_config = grid_config,
        project      = wandb_project,
        entity       = wandb_entity,
        n_runs       = 6,
    )
    print(f"\n  Grid search complete. Best config: {best_grid_config}")

    print("\n  Retraining tuned champion on full training set...")
    tuned_champion_model = retrain_best_model(
        best_config = best_grid_config,
        X_train     = X_train_eng,
        y_train     = y_train_eng,
        model_dir   = MODEL_DIR_TUNED,
    )
    print(f"\033[33m  [Duration] Hyperparameter tuning sweeps & champion retraining finished in {time.time() - t0:.2f}s\033[0m")

    print(f"\n\033[33m>>> [Duration] ACT 2 completed in {time.time() - act2_start:.2f}s\033[0m")

    return best_grid_config, winning_family, engineered_results




# ── Act 3 ────────────────────────────────────────────────────────────────────

def run_act3(best_grid_config, winning_family, engineered_results, wandb_project, wandb_entity):
    """
    Act 3 — Model Evaluation & Experiments
    - Step 6: Champion Model & Feature Importance & W&B logging
    - Step 8: Error Analysis (Tables + Segment breakdowns)
    - Step 9: Drift Detection (PSI + KS on Train vs Test)
    """
    _print_header("ACT 3 — Model Evaluation & Experiments")

    # Load datasets and models from disk
    train_clean = pd.read_parquet(TRAIN_CLEANED_PARQUET)
    test_clean  = pd.read_parquet(TEST_CLEANED_PARQUET)
    eng_test    = pd.read_parquet(ENGINEERED_TEST_PARQUET)
    eng_scaler  = joblib.load(Path(MODEL_DIR_ENGINEERED) / "scaler.pkl")
    
    tuned_model_path = Path(MODEL_DIR_TUNED) / f"tuned_{winning_family}.pkl"
    tuned_champion_model = joblib.load(tuned_model_path)

    X_test_eng = eng_test.drop(columns=[TARGET_COL])
    y_test_eng = eng_test[TARGET_COL]

    # ── Champion & Feature Importance
    champion_name  = select_champion(engineered_results, metric="mae")
    champion_model = load_model(champion_name, MODEL_DIR_ENGINEERED)
    champion_row   = engineered_results.loc[engineered_results["model"] == champion_name].iloc[0]

    plot_feature_importance(
        model         = champion_model,
        feature_names = X_test_eng.columns.tolist(),
        model_name    = champion_name,
        X_val         = X_test_eng,
        y_val         = y_test_eng,
        output_dir    = PLOTS_DIR,
    )

    tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = "champion-eval",
        tags     = ["champion", "engineered-features"],
        config   = {"champion_model": champion_name},
    )
    tracker.log_summary({
        "champion_model": champion_name,
        "mae":            float(champion_row["mae"]),
        "rmse":           float(champion_row["rmse"]),
        "mape":           float(champion_row["mape"]),
    })
    fi_path = Path(PLOTS_DIR) / f"feature_importance_{champion_name}.png"
    if fi_path.exists():
        tracker.log_image_file(fi_path, "feature_importance")
    tracker.log_code()
    url = tracker.finish()
    if url:
        print(f"\n  W&B run logged -> {url}")

    # ── Evaluate Tuned Model and Log
    y_pred_tuned = tuned_champion_model.predict(X_test_eng)
    tuned_mae  = float(np.mean(np.abs(y_test_eng.values - y_pred_tuned)))
    tuned_rmse = float(np.sqrt(np.mean((y_test_eng.values - y_pred_tuned) ** 2)))

    print(f"\n  Tuned model evaluation:")
    print(f"    MAE  : ${tuned_mae:.2f}  (champion baseline: ${float(champion_row['mae']):.2f})")
    print(f"    RMSE : ${tuned_rmse:.2f}  (champion baseline: ${float(champion_row['rmse']):.2f})")

    tuning_tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = f"tuned-{winning_family}",
        tags     = ["tuned", "grid-search"],
        config   = best_grid_config,
    )
    tuning_tracker.log_summary({"mae": tuned_mae, "rmse": tuned_rmse})

    tuned_pkl = Path(MODEL_DIR_TUNED) / f"tuned_{winning_family}.pkl"
    log_model_artifact(
        tuning_tracker, tuned_pkl, "tuned-champion",
        metadata={"source": "grid-search", "mae": tuned_mae},
    )
    log_feature_artifact(
        tuning_tracker,
        Path(MODEL_DIR_ENGINEERED) / "scaler.pkl",
        active_feature_steps=X_test_eng.columns.tolist(),
        metadata={"n_features": len(X_test_eng.columns)},
    )
    url = tuning_tracker.finish()
    if url:
        print(f"  W&B run logged -> {url}")

    # ── Error Analysis
    error_df, error_figs = run_error_analysis(
        X_test     = X_test_eng,
        y_test     = y_test_eng,
        model      = tuned_champion_model,
        output_dir = PLOTS_DIR,
    )

    error_tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = "error-analysis",
        tags     = ["error-analysis", "tuned"],
        config   = {"model": winning_family, "n_test_samples": len(error_df)},
    )
    error_tracker.log_table(
        error_df[["actual", "predicted", "abs_error", "pct_error",
                  "rush_hour_label", "trip_type", "distance_bucket",
                  "day_name", "time_of_day"]].dropna(how="all"),
        table_name = "per_sample_errors",
    )
    error_tracker.log_summary({"mae": float(error_df["abs_error"].mean()),
                                "p90_abs_error":  float(error_df["abs_error"].quantile(0.9))})
    for col, fig in error_figs.items():
        error_tracker.log_plot(fig, f"error_by_{col}")
    url = error_tracker.finish()
    if url:
        print(f"\n  W&B run logged → {url}")

    # ── Drift Detection (Train vs Test)
    drift_report = build_drift_report(train_clean, test_clean, ["trip_distance", "PULocationID", "DOLocationID"])
    print("\nDrift report:")
    print(drift_report)

    label_drift = detect_label_drift(train_clean, test_clean)
    print("\nLabel drift:")
    print(label_drift)

    concept_drift = detect_concept_drift(train_clean, test_clean, tuned_champion_model, eng_scaler)
    print("\nConcept drift:")
    print(concept_drift)

    print("\nGenerating drift distribution plots...")
    plot_feature_distributions(train_clean, test_clean, "trip_distance", cur_label="Jan 2025 Test", output_dir=PLOTS_DIR)
    plot_label_drift_distribution(train_clean, test_clean, cur_label="Jan 2025 Test", output_dir=PLOTS_DIR)

    return y_pred_tuned




# # ── Act 4 ─────────────────────────────────────────────────────────────────────

# def run_act4(winning_family, y_pred_tuned, wandb_project, wandb_entity):
#     """
#     Act 4 — Productionize Model
#     - Step 9.1: Evidently AI Drift Detection (Train vs. Dec 2024)
#     - Step 10: Drift Mitigation & Before/After Comparison
#     """
#     _print_header("ACT 4 — Productionize Model (Evidently AI Drift & Mitigation)")

#     drift_month_path = Path(DRIFT_RAW_PARQUET)
#     if not drift_month_path.exists():
#         print(f"  Drift month parquet not found at {DRIFT_RAW_PARQUET}")
#         print("  Skipping Steps 9.1 and 10. Place the file there to run Evidently drift analysis.")
#         return

#     DRIFT_TRAIN_SAMPLE = 20000
#     DRIFT_EVAL_SAMPLE  = 5000
#     DRIFT_SEED         = 42

#     drift_df = pd.read_parquet(DRIFT_RAW_PARQUET)
#     drift_df = clean_dataframe(drift_df)
#     drift_df["tpep_pickup_datetime"] = pd.to_datetime(drift_df["tpep_pickup_datetime"])

#     # Split by calendar date — first 3 weeks for mitigation, last week for eval
#     drift_train_raw = (
#         drift_df[drift_df["tpep_pickup_datetime"].dt.day <= 21]
#         .sample(DRIFT_TRAIN_SAMPLE, random_state=DRIFT_SEED)
#         .reset_index(drop=True)
#     )
#     drift_eval_raw = (
#         drift_df[drift_df["tpep_pickup_datetime"].dt.day >= 22]
#         .sample(DRIFT_EVAL_SAMPLE, random_state=DRIFT_SEED)
#         .reset_index(drop=True)
#     )
#     print(f"  Drift train set : {len(drift_train_raw):,} rows  (Dec 1–21,  seed={DRIFT_SEED})")
#     print(f"  Drift eval set  : {len(drift_eval_raw):,}  rows  (Dec 22–31, seed={DRIFT_SEED})")

#     # Load January scaler and datasets from disk
#     eng_scaler  = joblib.load(Path(MODEL_DIR_ENGINEERED) / "scaler.pkl")
#     train_clean = pd.read_parquet(TRAIN_CLEANED_PARQUET)
#     eng_train   = pd.read_parquet(ENGINEERED_TRAIN_PARQUET)
#     eng_test    = pd.read_parquet(ENGINEERED_TEST_PARQUET)
    
#     tuned_model_path = Path(MODEL_DIR_TUNED) / f"tuned_{winning_family}.pkl"
#     tuned_champion_model = joblib.load(tuned_model_path)

#     # Engineer features
#     drift_train_eng, _ = run_feature_pipeline(drift_train_raw, scaler=eng_scaler, is_training=False)
#     drift_eval_eng,  _ = run_feature_pipeline(drift_eval_raw,  scaler=eng_scaler, is_training=False)

#     X_train_eng = eng_train.drop(columns=[TARGET_COL])
#     y_train_eng = eng_train[TARGET_COL]

#     ref_eng_df = X_train_eng.copy()
#     ref_eng_df[TARGET_COL] = y_train_eng.values

#     # Run Evidently report
#     print("\n  Running Evidently dataset + label drift report ...")
#     evidently_report = run_evidently_drift_report(ref_eng_df, drift_train_eng)
#     drift_results    = parse_drift_results(evidently_report)

#     print(f"\n  Overall drift detected : {drift_results['overall_drift']}")
#     print(f"  Features drifted       : {drift_results['n_drifted']} "
#           f"({drift_results['share_drifted']:.1%} of feature columns)")
#     if drift_results["drifted_features"]:
#         print(f"  Drifted feature names  : {drift_results['drifted_features']}")
#     print(f"  Target (label) drift   : {drift_results['target_drift']}  "
#           f"(score={drift_results['target_drift_score']:.4f})")

#     Path("outputs").mkdir(exist_ok=True)
#     evidently_html = Path("outputs") / "evidently_drift_report.html"
#     evidently_report.save_html(str(evidently_html))
#     print(f"\n  Evidently dataset drift HTML  -> {evidently_html}")

#     # Concept drift
#     print("\n  Running Evidently concept drift report ...")
#     X_test_eng = eng_test.drop(columns=[TARGET_COL])
#     y_test_eng = eng_test[TARGET_COL]

#     ref_perf_df = X_test_eng.copy()
#     ref_perf_df[TARGET_COL]    = y_test_eng.values
#     ref_perf_df["prediction"]  = tuned_champion_model.predict(X_test_eng)

#     cur_perf_df = drift_eval_eng.copy()
#     cur_perf_df["prediction"]  = tuned_champion_model.predict(drift_eval_eng.drop(columns=[TARGET_COL]))

#     concept_drift_report   = run_evidently_concept_drift_report(ref_perf_df, cur_perf_df)
#     concept_drift_results  = parse_concept_drift_results(concept_drift_report)

#     print(f"\n  Concept drift detected : {concept_drift_results['concept_drift_detected']}")
#     print(f"  Reference MAE (Jan)    : ${concept_drift_results['ref_mae']:.2f}")
#     print(f"  Current MAE (Dec eval) : ${concept_drift_results['cur_mae']:.2f}")
#     print(f"  MAE increase           : {concept_drift_results['mae_pct_increase']:.1%}")

#     concept_drift_html = Path("outputs") / "evidently_concept_drift_report.html"
#     concept_drift_report.save_html(str(concept_drift_html))
#     print(f"\n  Evidently concept drift HTML  -> {concept_drift_html}")

#     selected_strategy = select_mitigation_strategy(drift_results, concept_drift_results)
#     print(f"\n  Selected strategy      : {selected_strategy}")

#     # ── Drift Mitigation
#     _print_header("STEP 10 — Drift Mitigation + Before / After Comparison")

#     drift_eval_parquet = Path(PROCESSED_DIR) / "drift_eval.parquet"
#     drift_eval_raw.to_parquet(drift_eval_parquet, index=False)

#     y_drift_eval      = drift_eval_eng[TARGET_COL].values
#     y_pred_drift_base = tuned_champion_model.predict(drift_eval_eng.drop(columns=[TARGET_COL]))
#     baseline_drift_mae = float(np.mean(np.abs(y_drift_eval - y_pred_drift_base)))
#     print(f"\n  Tuned champion MAE — Dec eval (pre-mitigation) : ${baseline_drift_mae:.2f}")

#     if selected_strategy == "none":
#         print("  No mitigation required — skipping Step 10.")
#         return

#     print(f"\n  Applying strategy: {selected_strategy}")
#     mitigated_model, mitigated_scaler, eval_steps = mitigate(
#         strategy         = selected_strategy,
#         train_df         = train_clean,
#         recent_df        = drift_train_raw,
#         model_name       = winning_family,
#         model_dir        = MODEL_DIR_MITIGATED,
#         base_model       = tuned_champion_model,
#         drifted_features = drift_results["drifted_features"],
#     )

#     if mitigated_model is None:
#         mitigated_model = tuned_champion_model

#     active_scaler = mitigated_scaler if mitigated_scaler is not None else eng_scaler
#     drift_eval_eng_mit, _ = run_feature_pipeline(
#         drift_eval_raw, scaler=active_scaler,
#         is_training=False, custom_creation_steps=eval_steps,
#     )

#     y_pred_drift_mit   = mitigated_model.predict(drift_eval_eng_mit.drop(columns=[TARGET_COL]))
#     mitigated_drift_mae = float(np.mean(np.abs(y_drift_eval - y_pred_drift_mit)))
#     improvement_pct   = (baseline_drift_mae - mitigated_drift_mae) / baseline_drift_mae * 100
#     print(f"  Mitigated model MAE — Dec eval (post-mitigation) : ${mitigated_drift_mae:.2f}")
#     print(f"  Improvement                                       : {improvement_pct:+.1f}%")

#     comparison_fig = plot_mitigation_comparison(
#         {
#             "Champion — Jan (in-dist)":             np.abs(y_test_eng.values - y_pred_tuned),
#             "Champion — Dec (drifted)":             np.abs(y_drift_eval - y_pred_drift_base),
#             f"Mitigated ({selected_strategy}) — Dec": np.abs(y_drift_eval - y_pred_drift_mit),
#         },
#         output_dir = PLOTS_DIR,
#     )

#     mitigation_tracker = ExperimentTracker(
#         project  = wandb_project,
#         entity   = wandb_entity,
#         run_name = f"mitigation-{selected_strategy}-dec",
#         tags     = ["drift-mitigation", "december", selected_strategy],
#         config   = {
#             "strategy":          selected_strategy,
#             "drifted_features":  drift_results["drifted_features"],
#             "n_drift_train":       len(drift_train_raw),
#             "n_drift_eval":        len(drift_eval_raw),
#             "drift_seed":          DRIFT_SEED,
#         },
#     )
#     mitigation_tracker.log_summary({
#         "jan_mae":             float(np.mean(np.abs(y_test_eng.values - y_pred_tuned))),
#         "baseline_drift_mae":    baseline_drift_mae,
#         "mitigated_drift_mae":   mitigated_drift_mae,
#         "mae_improvement_pct": improvement_pct,
#     })
#     mitigation_tracker.log_plot(comparison_fig, "mitigation_comparison")

#     log_data_artifact(
#         mitigation_tracker, drift_eval_parquet, "december-eval-set",
#         metadata={"month": "December 2024", "n_rows": len(drift_eval_raw), "seed": DRIFT_SEED},
#     )

#     mitigated_scaler_path = Path(MODEL_DIR_MITIGATED) / "scaler_mitigated.pkl"
#     joblib.dump(active_scaler, mitigated_scaler_path)

#     mitigated_feature_cols = [c for c in drift_eval_eng_mit.columns if c != TARGET_COL]

#     log_feature_artifact(
#         mitigation_tracker,
#         mitigated_scaler_path,
#         active_feature_steps=mitigated_feature_cols,
#         metadata={"strategy": selected_strategy, "n_features": len(mitigated_feature_cols)},
#     )

#     mitigated_pkl_name = {
#         "reweight_retrain": f"{winning_family}_reweighted.pkl",
#         "drop_features":    f"{winning_family}_drop_features.pkl",
#     }.get(selected_strategy)

#     if mitigated_pkl_name:
#         log_model_artifact(
#             mitigation_tracker,
#             Path(MODEL_DIR_MITIGATED) / mitigated_pkl_name,
#             "mitigated-model",
#             metadata={
#                 "strategy":          selected_strategy,
#                 "mae":               mitigated_drift_mae,
#                 "improvement_pct":   improvement_pct,
#                 "drifted_features":  drift_results["drifted_features"],
#             },
#         )

#     mitigation_tracker.log_artifact(
#         evidently_html, artifact_name="evidently-drift-report", artifact_type="report",
#     )

#     url = mitigation_tracker.finish()
#     if url:
#         print(f"\n  W&B run logged -> {url}")


# =========================================================================  
# ===================  Pipeline Orchestration  ============================  
# =========================================================================  

def run_pipeline(wandb_project=WANDB_PROJECT, wandb_entity=WANDB_ENTITY, sample_size=None, tuning_sample_size=TUNING_SAMPLE_SIZE):
    configure_wandb()
    # # Act 1 — Data Prep
    # run_act1(sample_size = sample_size)

    # Act 2 — Model Building & Tuning
    best_grid_config, winning_family, engineered_results = run_act2(
        wandb_project  = wandb_project,
        wandb_entity   = wandb_entity,
        tuning_sample_size = tuning_sample_size,
    )

    # # Act 3 — Model Evaluation & Experiments
    # y_pred_tuned = run_act3(
    #     best_grid_config   = best_grid_config,
    #     winning_family     = winning_family,
    #     engineered_results = engineered_results,
    #     wandb_project      = wandb_project,
    #     wandb_entity       = wandb_entity,
    # )

    # # Act 4 — Productionize Model
    # run_act4(
    #     winning_family = winning_family,
    #     y_pred_tuned   = y_pred_tuned,
    #     wandb_project  = wandb_project,
    #     wandb_entity   = wandb_entity,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        help="W&B project name to log runs into")
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY,
                        help="W&B entity namespace to log runs into")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Sample size for training and testing raw data (for smoke testing)")
    parser.add_argument("--tuning-sample-size", type=int, default=TUNING_SAMPLE_SIZE,
                        help="Sample size for hyperparameter tuning sweeps (stratified by hour and day)")
    args = parser.parse_args()
    
    run_pipeline(
        wandb_project = args.wandb_project,
        wandb_entity  = args.wandb_entity,
        sample_size   = args.sample_size,
        tuning_sample_size = args.tuning_sample_size,
    )
