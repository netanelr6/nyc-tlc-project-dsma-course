"""
NYC TLC -- Experiment Tracking
==============================
A thin, reusable wrapper around W&B that keeps all tracking logic out of
pipeline.py. Handles connectivity issues gracefully, supporting offline runs
and showing clear color-coded console logs in case of success or connection failures.
"""
import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from a local .env file
load_dotenv()


def configure_wandb():
    # If WANDB_MODE is already configured, do nothing
    if "WANDB_MODE" in os.environ:
        return

    # If WANDB_API_KEY is present, log in automatically
    if os.getenv("WANDB_API_KEY"):
        return

    import sys
    is_jupyter = "ipykernel" in sys.modules
    is_interactive = sys.stdout.isatty() and sys.stdin.isatty()

    if not (is_jupyter or is_interactive):
        # Non-interactive, default to disabled to prevent blocking
        os.environ["WANDB_MODE"] = "disabled"
        return

    print("\n" + "=" * 60)
    print(" Weights & Biases (W&B) API Key is missing.")
    print(" Choose how to proceed:")
    print("  (1) Proceed with manual W&B login (standard W&B menu)")
    print("  (2) Skip W&B completely (Run fully local/offline, no W&B tracking)")
    print("=" * 60)

    try:
        choice = input("Enter choice [1-2] (default: 2): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "2"

    if not choice:
        choice = "2"

    if choice == "2":
        os.environ["WANDB_MODE"] = "disabled"
        print("\033[93m[W&B] Weights & Biases is disabled. Running fully offline/locally.\033[0m\n")
    else:
        print("\033[92m[W&B] Proceeding with standard W&B manual login...\033[0m\n")


class ExperimentTracker:
    """
    Wraps a single W&B run.
    """

    def __init__(self, project, entity=None, run_name=None, tags=None, config=None):
        self.enabled = False
        self.project = project
        self.run = None

        configure_wandb()
        if os.environ.get("WANDB_MODE") == "disabled":
            self.enabled = False
            return

        try:
            # If an API key is present in environment, log in automatically
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
            else:
                # Check if user is authenticated/logged in. Prompts interactively if no key is configured.
                if not wandb.login(anonymous="never", relogin=False):
                    raise ValueError("Not logged in to W&B.")

            self.run = wandb.init(
                project = project,
                entity  = entity,
                id      = wandb.util.generate_id(),
                name    = run_name,
                tags    = tags or [],
                config  = config or {},
                reinit  = True,
            )
            self.enabled = True
            # Green success message in terminal
            print("\033[92m[W&B] Successfully connected and logged to W&B.\033[0m")
        except Exception as e:
            self.run = None
            self.enabled = False
            # Red warning message in terminal
            print(f"\033[91m[W&B] Was supposed to be logged to W&B, but a key was not provided or valid. Here is the result:\nError: {e}\033[0m")

    # ── Metrics ───────────────────────────────────────────────────────────────

    def log_metrics(self, metrics: dict):
        """Log step-level metrics if W&B is enabled."""
        if self.enabled and self.run is not None:
            self.run.log(metrics)

    def log_summary(self, metrics: dict):
        """Log final run summary metrics if W&B is enabled."""
        if self.enabled and self.run is not None:
            self.run.summary.update(metrics)

    # ── Interactive table ─────────────────────────────────────────────────────

    def log_table(self, df: pd.DataFrame, table_name: str):
        """Log a pandas DataFrame as an interactive W&B table."""
        if self.enabled and self.run is not None:
            table = wandb.Table(dataframe=df.reset_index(drop=True))
            self.run.log({table_name: table})

    # ── Plots ─────────────────────────────────────────────────────────────────

    def log_plot(self, fig, name: str):
        """
        Log matplotlib Figure to W&B and close it to free memory.
 
        The table is fully interactive in the W&B UI — you can filter
        by any column, sort by error magnitude, and slice by categorical
        features without writing any additional code.

        Ideal use: the per-sample error DataFrame from error_analysis.py.
        
        """
        if self.enabled and self.run is not None:
            self.run.log({name: wandb.Image(fig)})
        plt.close(fig)

    def log_image_file(self, image_path, name: str):
        """
        Log an existing image file to the run.

        Keeps all charts in one place (the W&B run page) rather than
        scattered across an outputs/ folder.
        """
        if self.enabled and self.run is not None:
            self.run.log({name: wandb.Image(str(image_path))})

    # ── Artifacts (model versioning) ──────────────────────────────────────────

    def log_artifact(self, file_path, artifact_name: str,
                     artifact_type: str = "model", metadata: dict = None):
        """
        Version a file (model .pkl, scaler, feature store) as a W&B Artifact.

        Each call creates a new version (v0, v1, v2 …) automatically.
        The lineage graph in W&B shows exactly which run produced each version.

        Ideal use: log every saved model so that you can compare
        model:v1 (baseline) → model:v2 (tuned) → model:v3 (post-mitigation).
        """
        if self.enabled and self.run is not None:
            artifact = wandb.Artifact(
                name     = artifact_name,
                type     = artifact_type,
                metadata = metadata or {},
            )
            artifact.add_file(str(file_path))
            self.run.log_artifact(artifact)

    # ── Alerts ────────────────────────────────────────────────────────────────

    def alert(self, title: str, text: str, level: str = "WARN"):
        """
        Fire a W&B alert.

        You will receive an email/Slack notification when this is called.
        Use it when drift is detected or MAE exceeds a threshold — a
        production monitoring moment in two lines of code.

        level options: "INFO", "WARN", "ERROR"
        """
        if self.enabled and self.run is not None:
            wandb.alert(title=title, text=text, level=level)
        else:
            print(f"[Alert] {level} - {title}: {text}")

    # ── Code snapshot ─────────────────────────────────────────────────────────

    def log_code(self, root: str = "."):
        """Log a snapshot of code files."""
        if self.enabled and self.run is not None:
            self.run.log_code(root)

    # ── Finish ────────────────────────────────────────────────────────────────

    def finish(self) -> str:
        """Close the run and return its URL."""
        if self.enabled and self.run is not None:
            url = self.run.url
            self.run.finish()
            return url
        return None


# ── Standalone monthly drift tracking ─────────────────────────────────────────

def log_monthly_drift_run(
    month_label:   str,
    month_num:     int,
    mae:           float,
    drift_report:  pd.DataFrame,
    project:       str,
    entity:        str   = None,
    mae_delta:     float = None,
    n_trips:       int   = None,
    label_drift:   dict  = None,
):
    """
    Log one evaluation month as its own W&B run.

    Running this for every month in the evaluation set produces a set of
    runs that share the same project.  In W&B, select all of them and plot
    MAE vs. month_num to get the model-degradation curve automatically —
    no extra code required.

    Args:
        month_label  : human-readable label, e.g. "Feb" or "2024-02"
        month_num    : integer month (1–12) — used as the x-axis in comparisons
        mae          : model MAE on this month's data
        drift_report : DataFrame from build_drift_report()
        project      : W&B project name
        entity       : W&B entity name (optional)
        mae_delta    : MAE increase vs. reference month (optional)
        n_trips      : number of trips evaluated (optional)
        label_drift  : dict from detect_label_drift() (optional)
    """
    configure_wandb()
    if os.environ.get("WANDB_MODE") == "disabled":
        return

    try:
        # If an API key is present in environment, log in automatically
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        else:
            # Check if user is authenticated/logged in. Prompts interactively if no key is configured.
            if not wandb.login(anonymous="never", relogin=False):
                raise ValueError("Not logged in to W&B.")

        run = wandb.init(
            project = project,
            entity  = entity,
            id      = wandb.util.generate_id(),
            name    = f"drift-eval-{month_label}",
            tags    = ["drift-monitoring", month_label],
            config  = {
                "evaluation_month": month_label,
                "month_num":        month_num,
            },
            reinit  = True,
        )

        run.summary["mae"]       = mae
        run.summary["month_num"] = month_num

        if mae_delta is not None:
            run.summary["mae_delta"] = mae_delta
        if n_trips is not None:
            run.summary["n_trips"] = n_trips

        if label_drift is not None:
            run.summary["label_psi"]        = label_drift["psi"]
            run.summary["label_ks_pvalue"]  = label_drift["ks_pvalue"]
            run.summary["label_drifted"]    = label_drift["drifted"]
            run.summary["label_ref_mean"]   = label_drift["ref_mean"]
            run.summary["label_cur_mean"]   = label_drift["cur_mean"]

        for _, row in drift_report.iterrows():
            feat = row["feature"]
            run.summary[f"psi_{feat}"]      = row["psi"]
            run.summary[f"ks_pvalue_{feat}"] = row["ks_pvalue"]
            run.summary[f"drifted_{feat}"]  = bool(row["drifted"])

        run.summary["n_drifted_features"] = int(drift_report["drifted"].sum())
        run.log({"drift_report": wandb.Table(dataframe=drift_report)})
        run.finish()
        
        print(f"\033[92m[W&B] Successfully logged monthly drift run for {month_label}.\033[0m")
    except Exception as e:
        print(f"\033[91m[W&B] Was supposed to be logged to W&B, but a key was not provided or valid. Here is the result:\nError: {e}\nLocal summary for {month_label}: MAE={mae:.4f}, n_trips={n_trips}\033[0m")
