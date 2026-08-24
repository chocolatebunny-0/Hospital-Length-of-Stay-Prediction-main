"""
Prediction and explanation logic.

This module is the single place that touches the fitted `model` and `scaler`
from the deployment package. It re-creates, for a single new patient, exactly
the same encoding and scaling steps the notebook applied to the training data:

    raw inputs -> one-hot encode (Gender, Medical Condition) -> order columns
    to match `feature_names` -> scaler.transform() -> model.predict()

No preprocessing logic is changed or re-fitted; the scaler and model are used
strictly in inference mode.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src import config
from src.model.loader import load_model_dataset

# Model classes SHAP's TreeExplainer supports directly. Anything else (e.g.
# Linear Regression) falls back to shap.Explainer's model-agnostic path.
TREE_MODEL_TYPES = (
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "DecisionTreeRegressor",
    "XGBRegressor",
)


@dataclass
class PatientInput:
    age: float
    gender: str
    medical_condition: str
    glucose: float
    blood_pressure: float
    bmi: float
    oxygen_saturation: float
    cholesterol: float
    triglycerides: float
    hba1c: float
    smoking: bool
    alcohol: bool
    physical_activity: float
    diet_score: float
    family_history: bool
    stress_level: float
    sleep_hours: float


def build_feature_row(
    patient: PatientInput,
    feature_names: list[str],
    medical_conditions: list[str],
) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching the notebook's encoded feature order.

    Categorical variables are expanded into the same one-hot dummy columns
    produced by `pd.get_dummies(..., drop_first=True)` in the notebook, with
    the baseline categories (Gender="Female", Medical Condition="Arthritis")
    represented implicitly as all-zero dummies.
    """
    row = {
        "Age": patient.age,
        "Glucose": patient.glucose,
        "Blood Pressure": patient.blood_pressure,
        "BMI": patient.bmi,
        "Oxygen Saturation": patient.oxygen_saturation,
        "Cholesterol": patient.cholesterol,
        "Triglycerides": patient.triglycerides,
        "HbA1c": patient.hba1c,
        "Smoking": int(patient.smoking),
        "Alcohol": int(patient.alcohol),
        "Physical Activity": patient.physical_activity,
        "Diet Score": patient.diet_score,
        "Family History": int(patient.family_history),
        "Stress Level": patient.stress_level,
        "Sleep Hours": patient.sleep_hours,
        "Gender_Male": int(patient.gender == "Male"),
    }

    for condition in medical_conditions:
        column = f"Medical Condition_{condition}"
        if column not in feature_names:
            # Baseline category dropped by pd.get_dummies(drop_first=True);
            # represented implicitly by all other dummies being 0.
            continue
        row[column] = int(patient.medical_condition == condition)

    return pd.DataFrame([row], columns=feature_names)


def resolve_model(package: dict, model_name: str | None):
    """Return the requested model, falling back to the deployed default."""
    models = package.get("models")
    if models and model_name in models:
        return models[model_name]
    return package["model"]


def predict(
    patient: PatientInput, package: dict, model_name: str | None = None
) -> tuple[float, pd.DataFrame]:
    """Scale the patient row with the fitted scaler and predict with the chosen model.

    Returns the predicted length of stay (days) and the scaled feature row
    (needed downstream for SHAP explanation).
    """
    feature_names = package["feature_names"]
    model = resolve_model(package, model_name)
    scaler = package["scaler"]
    medical_conditions = package["medical_conditions"]

    raw_row = build_feature_row(patient, feature_names, medical_conditions)
    scaled_values = scaler.transform(raw_row)
    scaled_row = pd.DataFrame(scaled_values, columns=feature_names, index=raw_row.index)

    prediction = float(model.predict(scaled_row)[0])
    return prediction, scaled_row


def background_sample(package: dict) -> pd.DataFrame:
    """Small scaled sample used as a SHAP background/masker for non-tree models."""
    feature_names = package["feature_names"]
    scaler = package["scaler"]
    dataset = load_model_dataset()
    sample = dataset[feature_names].sample(
        n=min(config.SHAP_BACKGROUND_SIZE, len(dataset)), random_state=config.RANDOM_STATE
    )
    scaled = scaler.transform(sample)
    return pd.DataFrame(scaled, columns=feature_names, index=sample.index)


def explain(
    scaled_row: pd.DataFrame, package: dict, model_name: str | None = None
) -> shap.Explanation:
    """Generate a SHAP explanation for a single scaled patient row, for the chosen model.

    Tree-based models (Random Forest, Gradient Boosting, Decision Tree,
    XGBoost) use shap.TreeExplainer, matching the notebook's methodology
    exactly. Non-tree models (Linear Regression) fall back to shap.Explainer's
    model-agnostic path with a small background sample as the masker.
    """
    model = resolve_model(package, model_name)
    display_row = scaled_row.rename(columns=config.DISPLAY_NAME_OVERRIDES)

    if type(model).__name__ in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(scaled_row)[0]
        base_value = np.asarray(explainer.expected_value).reshape(-1)[0]
    else:
        explainer = shap.Explainer(model.predict, background_sample(package))
        result = explainer(scaled_row)
        values = result.values[0]
        base_value = np.asarray(result.base_values).reshape(-1)[0]

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=display_row.iloc[0].values,
        feature_names=display_row.columns.tolist(),
    )


def _active_values_and_names(explanation: shap.Explanation) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values/names filtered to exclude inactive binary/flag features.

    Binary/flag features (e.g. Diabetes, Smoking) get a SHAP value whether
    they're 0 or 1 for this patient, since SHAP explains every column. Only
    surface them when they're actually true for this patient (a scaled value
    above 0 for a StandardScaler-transformed 0/1 feature always means the
    patient's raw value was 1), otherwise "Diabetes" would read as a
    contributor for a patient who doesn't have diabetes.
    """
    values = np.asarray(explanation.values)
    names = np.asarray(explanation.feature_names)
    data = np.asarray(explanation.data)

    active = np.array(
        [name not in config.BINARY_FEATURE_NAMES or data[i] > 0 for i, name in enumerate(names)]
    )
    return values[active], names[active]


def top_contributing_features(explanation: shap.Explanation, n: int = 3) -> dict:
    """Return the top-n features increasing and decreasing the prediction."""
    values, names = _active_values_and_names(explanation)
    order = np.argsort(values)

    decreasing = [(names[i], values[i]) for i in order[:n] if values[i] < 0]
    increasing = [(names[i], values[i]) for i in order[::-1][:n] if values[i] > 0]

    return {"increasing": increasing, "decreasing": decreasing}


def all_contributing_features(explanation: shap.Explanation, n: int = 8) -> list[tuple[str, float]]:
    """Return the top-n features by absolute impact, in either direction."""
    values, names = _active_values_and_names(explanation)
    order = np.argsort(-np.abs(values))[:n]
    return [(names[i], values[i]) for i in order]
