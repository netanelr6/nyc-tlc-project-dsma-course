import os
import google.generativeai as genai

# Models to attempt in priority order (hybrid approach with fallback)
MODELS_TO_TRY = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemma-4-31b",
    "gemini-2.0-flash-lite"
]

def analyze_drift_with_gemini(drift_results: dict, concept_drift_results: dict) -> str:
    """
    Analyzes model drift and performance metrics using Google Gemini models.
    Supports graceful fallback if the Gemini API key is missing or if specific models
    experience rate limits or other transient errors.

    Args:
        drift_results (dict): Dictionary with dataset and target drift statistics.
        concept_drift_results (dict): Dictionary with concept drift and performance statistics.

    Returns:
        str: AI-generated analysis of the drift report, or None if skipped/failed.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n  [Gemini AI] GEMINI_API_KEY environment variable not found.")
        print("              Skipping AI-assisted drift analysis.")
        return None

    # Configure the Google GenAI SDK
    genai.configure(api_key=api_key)

    # Format metrics for the prompt
    overall_drift = drift_results.get("overall_drift", False)
    share_drifted = drift_results.get("share_drifted", 0.0)
    n_drifted = drift_results.get("n_drifted", 0)
    drifted_features = drift_results.get("drifted_features", [])
    target_drift = drift_results.get("target_drift", False)
    target_drift_score = drift_results.get("target_drift_score", 0.0)

    concept_drift_detected = concept_drift_results.get("concept_drift_detected", False)
    ref_mae = concept_drift_results.get("ref_mae", 0.0)
    cur_mae = concept_drift_results.get("cur_mae", 0.0)
    mae_pct_increase = concept_drift_results.get("mae_pct_increase", 0.0)

    # Build the prompt
    prompt = f"""
Analyze the following MLOps data and concept drift metrics for the NYC TLC Taxi trip fare prediction model:

1. Dataset Drift (Feature Distributions shift):
- Overall Dataset Drift Detected: {overall_drift}
- Number of Drifted Feature Columns: {n_drifted} ({share_drifted:.1%} of all features)
- Drifted Feature Names: {drifted_features}

2. Target/Label Drift (Fare Amount shift):
- Target Drift Detected: {target_drift}
- Target Drift Score: {target_drift_score:.4f}

3. Concept Drift (Model Performance degradation):
- Concept Drift (MAE growth > 10%) Detected: {concept_drift_detected}
- Reference MAE (Jan 2024 Test Set): ${ref_mae:.2f}
- Current MAE (Dec 2024 Eval Set): ${cur_mae:.2f}
- MAE Percentage Increase: {mae_pct_increase:.1%}

Based on these results, write a concise, professional analysis (around 3 to 4 sentences) for an ML engineering team. Explain:
- The probable cause and severity of the drift (correlating the drifted features to the model error increase).
- Whether a mitigation strategy (such as retraining or feature dropping) is highly recommended.
- A brief concluding recommendation.
"""

    for model_name in MODELS_TO_TRY:
        try:
            print(f"  [Gemini AI] Attempting analysis using model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Check if we got a valid text response
            if response and response.text:
                print(f"  [Gemini AI] Analysis completed successfully using {model_name}!")
                return response.text
            else:
                print(f"  [Gemini AI Warning] Empty response from {model_name}. Trying fallback...")
        except Exception as e:
            print(f"  [Gemini AI Warning] Model {model_name} failed: {e}")
            print("                      Trying fallback model...")

    print("  [Gemini AI Error] All Gemini/Gemma models failed or returned errors. Skipping analysis.")
    return None
