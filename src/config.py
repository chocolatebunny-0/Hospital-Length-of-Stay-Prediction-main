"""
Central constants for the Hospital Length of Stay application.

These values mirror the preprocessing performed in
`Hospital_Length_of_Stay_Prediction_Using_Machine_Learning.ipynb` and must stay
in sync with it. They are not re-derived at runtime so that the app never
drifts from the notebook's source-of-truth encoding.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEPLOYMENT_PACKAGE_PATH = ARTIFACTS_DIR / "hospital_los_predictor.pkl"
MODEL_RESULTS_PATH = ARTIFACTS_DIR / "model_results.csv"
MODEL_DATASET_PATH = ARTIFACTS_DIR / "model_dataset.csv"
RAW_DATASET_PATH = ROOT_DIR / "data" / "healthcare_risk_factors_dataset.csv"

TARGET_COLUMN = "LengthOfStay"

# Medical Condition categories as they appear in the raw dataset.
# "Arthritis" is the baseline category dropped by pd.get_dummies(drop_first=True)
# in the notebook, so it is never represented by its own column and every
# Medical Condition_* dummy is 0 when a patient has Arthritis.
MEDICAL_CONDITIONS = [
    "Arthritis",
    "Asthma",
    "Cancer",
    "Diabetes",
    "Healthy",
    "Hypertension",
    "Obesity",
]
MEDICAL_CONDITION_BASELINE = "Arthritis"

GENDERS = ["Female", "Male"]
GENDER_BASELINE = "Female"

# Exact column order expected by the fitted scaler / model, taken from the
# notebook's `X.columns.tolist()` (also verified against artifacts/model_dataset.csv).
FEATURE_ORDER = [
    "Age",
    "Glucose",
    "Blood Pressure",
    "BMI",
    "Oxygen Saturation",
    "Cholesterol",
    "Triglycerides",
    "HbA1c",
    "Smoking",
    "Alcohol",
    "Physical Activity",
    "Diet Score",
    "Family History",
    "Stress Level",
    "Sleep Hours",
    "Gender_Male",
    "Medical Condition_Asthma",
    "Medical Condition_Cancer",
    "Medical Condition_Diabetes",
    "Medical Condition_Healthy",
    "Medical Condition_Hypertension",
    "Medical Condition_Obesity",
]

# Display-friendly labels for the one-hot dummy columns, matching the renames
# applied to SHAP plots in the notebook (cell "Create display-friendly feature names").
DISPLAY_NAME_OVERRIDES = {
    "Gender_Male": "Male",
    "Medical Condition_Asthma": "Asthma",
    "Medical Condition_Cancer": "Cancer",
    "Medical Condition_Diabetes": "Diabetes",
    "Medical Condition_Healthy": "Healthy",
    "Medical Condition_Hypertension": "Hypertension",
    "Medical Condition_Obesity": "Obesity",
}

# Plain-language phrases for each display feature name, used to turn a bare
# name like "BMI" or "Healthy" into a readable clause (e.g. "their BMI",
# "not having an underlying condition on record") for a lay audience.
FEATURE_PHRASES = {
    "Age": "their age",
    "Glucose": "their glucose level",
    "Blood Pressure": "their blood pressure",
    "BMI": "their BMI",
    "Oxygen Saturation": "their oxygen saturation level",
    "Cholesterol": "their cholesterol level",
    "Triglycerides": "their triglyceride level",
    "HbA1c": "their HbA1c level",
    "Smoking": "smoking",
    "Alcohol": "alcohol use",
    "Physical Activity": "their physical activity level",
    "Diet Score": "their diet score",
    "Family History": "a family history of illness",
    "Stress Level": "their stress level",
    "Sleep Hours": "how much they sleep",
    "Male": "being male",
    "Asthma": "having asthma",
    "Cancer": "having cancer",
    "Diabetes": "having diabetes",
    "Healthy": "not having an underlying condition on record",
    "Hypertension": "having hypertension",
    "Obesity": "having obesity",
}

# General adult clinical reference ranges, used to flag whether a patient's
# measurement is within a normal range on the Prediction page. These are
# widely-cited general reference ranges, not a substitute for clinical
# guidance, and only apply to the continuous clinical measurement fields.
CLINICAL_REFERENCE_RANGES = {
    "Glucose": {"label": "Blood glucose", "low": 70, "high": 99, "unit": "mg/dL"},
    "Blood Pressure": {"label": "Blood pressure", "low": 90, "high": 120, "unit": "mmHg"},
    "BMI": {"label": "BMI", "low": 18.5, "high": 24.9, "unit": ""},
    "Oxygen Saturation": {"label": "Oxygen saturation", "low": 95, "high": 100, "unit": "%"},
    "Cholesterol": {"label": "Cholesterol", "low": 0, "high": 200, "unit": "mg/dL"},
    "Triglycerides": {"label": "Triglycerides", "low": 0, "high": 150, "unit": "mg/dL"},
    "HbA1c": {"label": "HbA1c", "low": 4.0, "high": 5.6, "unit": "%"},
}

# Short bare nouns for each feature, used when several features are listed
# together in a sentence (e.g. "glucose, triglycerides and oxygen saturation
# suggested a shorter recovery period") -- unlike FEATURE_PHRASES, these drop
# "level"/"their"/"having" since repeating a suffix across a list reads
# clunkily ("glucose level, triglyceride level and oxygen saturation level").
FEATURE_BARE_NOUNS = {
    "Age": "age",
    "Glucose": "glucose",
    "Blood Pressure": "blood pressure",
    "BMI": "BMI",
    "Oxygen Saturation": "oxygen saturation",
    "Cholesterol": "cholesterol",
    "Triglycerides": "triglycerides",
    "HbA1c": "HbA1c",
    "Smoking": "smoking",
    "Alcohol": "alcohol use",
    "Physical Activity": "physical activity",
    "Diet Score": "diet score",
    "Family History": "family history",
    "Stress Level": "stress level",
    "Sleep Hours": "sleep",
    "Male": "being male",
    "Asthma": "asthma",
    "Cancer": "cancer",
    "Diabetes": "diabetes",
    "Healthy": "not having an underlying condition on record",
    "Hypertension": "hypertension",
    "Obesity": "obesity",
}

# Display names of binary/flag features (0 or 1 in the raw data). When listing
# top SHAP contributors for a patient, these should only be surfaced when the
# patient's actual value is 1/true -- SHAP assigns a value to every one-hot
# dummy column regardless of whether it's 0 or 1 for that patient, and
# displaying e.g. "Diabetes" as a contributor when the patient does NOT have
# diabetes reads as though they do.
BINARY_FEATURE_NAMES = {
    "Male",
    "Asthma",
    "Cancer",
    "Diabetes",
    "Healthy",
    "Hypertension",
    "Obesity",
    "Smoking",
    "Alcohol",
    "Family History",
}

SHAP_BACKGROUND_SIZE = 100
RANDOM_STATE = 42

# Healthcare analytics colour palette.
COLORS = {
    "primary": "#0F6E63",       # deep teal
    "primary_light": "#14A38C",
    "accent": "#2D6CDF",        # clinical blue
    "background": "#F5F8F8",
    "surface": "#FFFFFF",
    "text": "#1B2B2A",
    "text_muted": "#5B6B6A",
    "border": "#E1E8E7",
    "success": "#1E8E5A",
    "warning": "#C77F1A",
    "danger": "#C0392B",
}
