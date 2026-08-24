"""Prediction page: patient data entry, and a popup with the prediction result."""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_deployment_package
from src.model.predictor import (
    PatientInput,
    all_contributing_features,
    explain,
    predict,
    top_contributing_features,
)
from src.ui.components import card, page_header, section_title
from src.ui.icons import icon as get_icon
from src.utils.formatting import format_days

_MAX_HISTORY = 10

# Widget keys for every patient-data input, so a Reset button can clear them
# by deleting the keys (the widget then falls back to its default value).
_INPUT_KEYS = [
    "age", "gender", "medical_condition", "glucose", "blood_pressure", "bmi",
    "oxygen_saturation", "cholesterol", "triglycerides", "hba1c",
    "smoking", "alcohol", "family_history", "physical_activity",
    "diet_score", "stress_level", "sleep_hours",
]

# Maps a clinical-measurement feature name to the PatientInput attribute
# holding its raw (unscaled) value, for the Clinical Observations panel.
_FEATURE_TO_PATIENT_ATTR = {
    "Glucose": "glucose",
    "Blood Pressure": "blood_pressure",
    "BMI": "bmi",
    "Oxygen Saturation": "oxygen_saturation",
    "Cholesterol": "cholesterol",
    "Triglycerides": "triglycerides",
    "HbA1c": "hba1c",
}


def _entry_form(package: dict) -> tuple[str, PatientInput | None]:
    with card():
        section_title("Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", min_value=0, max_value=120, value=0, step=1, key="age")
        gender = c2.selectbox(
            "Gender", package["gender_options"], index=None, placeholder="Enter or select gender", key="gender"
        )
        medical_condition = c3.selectbox(
            "Medical Condition",
            package["medical_conditions"],
            index=None,
            placeholder="Enter or select condition",
            key="medical_condition",
        )

    with card():
        section_title("Clinical Measurements")
        c1, c2, c3 = st.columns(3)
        glucose = c1.number_input("Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="glucose")
        blood_pressure = c2.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=300.0, value=0.0, step=1.0, key="blood_pressure")
        bmi = c3.number_input("BMI", min_value=0.0, max_value=80.0, value=0.0, step=0.1, key="bmi")

        c1, c2, c3 = st.columns(3)
        oxygen_saturation = c1.number_input("Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="oxygen_saturation")
        cholesterol = c2.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=0.0, step=1.0, key="cholesterol")
        triglycerides = c3.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=800.0, value=0.0, step=1.0, key="triglycerides")

        c1, _, _ = st.columns(3)
        hba1c = c1.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="hba1c")

    with card():
        section_title("Lifestyle Factors")
        c1, c2, c3 = st.columns(3)
        smoking = c1.toggle("Smoker", value=False, key="smoking")
        alcohol = c2.toggle("Alcohol Use", value=False, key="alcohol")
        family_history = c3.toggle("Family History of Illness", value=False, key="family_history")

        c1, c2 = st.columns(2)
        physical_activity = c1.slider("Physical Activity (hrs/week)", -5.0, 15.0, 0.0, 0.1, key="physical_activity")
        diet_score = c2.slider("Diet Score (0-12)", -3.0, 13.0, 0.0, 0.1, key="diet_score")

        c1, c2 = st.columns(2)
        stress_level = c1.slider("Stress Level (0-16)", -3.0, 16.0, 0.0, 0.1, key="stress_level")
        sleep_hours = c2.slider("Sleep Hours (per night)", 0.0, 12.0, 0.0, 0.1, key="sleep_hours")

    model_names = list(package.get("models", {}).keys()) or [package.get("default_model_name", "Random Forest")]
    default_name = package.get("default_model_name", model_names[0])
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        model_name = st.selectbox(
            "Model",
            model_names,
            index=model_names.index(default_name),
            help="Random Forest is the model recommended in this project's evaluation, "
            "but any of the five compared models can be used to generate a prediction.",
        )
    with c2:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        submitted = st.button("Predict Length of Stay", type="primary", width="stretch")
    with c3:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        reset = st.button("Reset", width="stretch", icon=":material/restart_alt:")

    if reset:
        for key in _INPUT_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

    if not submitted:
        return model_name, None

    if gender is None or medical_condition is None:
        st.error("Please select a Gender and Medical Condition before predicting.")
        return model_name, None

    return model_name, PatientInput(
        age=age,
        gender=gender,
        medical_condition=medical_condition,
        glucose=glucose,
        blood_pressure=blood_pressure,
        bmi=bmi,
        oxygen_saturation=oxygen_saturation,
        cholesterol=cholesterol,
        triglycerides=triglycerides,
        hba1c=hba1c,
        smoking=smoking,
        alcohol=alcohol,
        physical_activity=physical_activity,
        diet_score=diet_score,
        family_history=family_history,
        stress_level=stress_level,
        sleep_hours=sleep_hours,
    )


def _join_naturally(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f" and {phrases[-1]}"


def _narrative_summary(contributors: dict, prediction: float) -> str:
    """A short two-sentence summary: the headline prediction, then what drove it."""
    increasing = [name for name, _ in contributors["increasing"]]
    decreasing = [name for name, _ in contributors["decreasing"]]
    days = format_days(prediction)
    headline = f"The model predicts a hospital stay of around **{days}**."

    if not increasing and not decreasing:
        return f"{headline} No single factor stood out as a major driver for this patient."

    def phrase(names: list[str]) -> str:
        if len(names) == 1:
            return config.FEATURE_BARE_NOUNS.get(names[0], names[0])
        lead = config.FEATURE_BARE_NOUNS.get(names[0], names[0])
        return f"{lead} and several other factors"

    if increasing and decreasing:
        down = _join_naturally([config.FEATURE_BARE_NOUNS.get(name, name) for name in decreasing])
        return (
            f"{headline} For this patient, {phrase(increasing)} pushed the estimate "
            f"higher, while {down} pushed it lower."
        )

    if increasing:
        return f"{headline} For this patient, {phrase(increasing)} pushed the estimate higher."

    down = _join_naturally([config.FEATURE_BARE_NOUNS.get(name, name) for name in decreasing])
    return f"{headline} For this patient, {down} pushed the estimate lower."


def _clinical_observations(explanation, patient: PatientInput) -> list[dict]:
    """Reference-range status for the clinical measurements that most influenced this prediction."""
    values, names = explanation.values, explanation.feature_names
    ranked = sorted(
        (
            (name, abs(value))
            for name, value in zip(names, values)
            if name in config.CLINICAL_REFERENCE_RANGES
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    observations = []
    for name, _ in ranked[:4]:
        attr = _FEATURE_TO_PATIENT_ATTR[name]
        value = getattr(patient, attr)
        range_info = config.CLINICAL_REFERENCE_RANGES[name]

        if value < range_info["low"]:
            status, normal = "Low", False
        elif value > range_info["high"]:
            status, normal = "Elevated", False
        else:
            status, normal = "Normal", True

        unit = f" {range_info['unit']}" if range_info["unit"] else ""
        observations.append(
            {
                "label": range_info["label"],
                "status": status,
                "value_str": f"{value:g}{unit}",
                "normal": normal,
            }
        )
    return observations


def _contribution_chart(items: list[tuple[str, float]]) -> go.Figure:
    items = sorted(items, key=lambda item: item[1])
    names = [config.FEATURE_PHRASES.get(name, name).capitalize() for name, _ in items]
    values = [value for _, value in items]
    colors = [config.COLORS["danger"] if v > 0 else config.COLORS["accent"] for v in values]

    fig = go.Figure(
        go.Bar(x=values, y=names, orientation="h", marker_color=colors)
    )
    fig.update_layout(
        xaxis_title="Impact on predicted length of stay (days)",
        yaxis_title="",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"]),
    )
    return fig


def _record_history(state: dict) -> None:
    """Log this prediction (by reference, so later model switches stay in sync),
    most recent first, capped at _MAX_HISTORY."""
    history = st.session_state.setdefault("prediction_history", [])
    history.insert(0, state)
    del history[_MAX_HISTORY:]


def _recompute(state: dict, package: dict, model_name: str) -> None:
    """Re-run prediction/explanation for the same patient under a different model.

    Mutates state in place -- since history entries are stored by reference
    (see _record_history), this keeps the sidebar history in sync too,
    regardless of the entry's position in the list.
    """
    prediction, scaled_row = predict(state["patient"], package, model_name)
    explanation = explain(scaled_row, package, model_name)
    state["prediction"] = prediction
    state["explanation"] = explanation
    state["model_name"] = model_name


@st.dialog("Prediction Result", width="large")
def _show_result_dialog(state: dict, package: dict) -> None:
    model_names = list(package.get("models", {}).keys()) or [state.get("model_name", "Random Forest")]

    st.markdown(
        "<div style='text-align:center; margin: 0 0 0.5rem;'>", unsafe_allow_html=True
    )
    picked = st.selectbox(
        "Model used",
        model_names,
        index=model_names.index(state["model_name"]),
        key="dialog_model_select",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if picked != state["model_name"]:
        _recompute(state, package, picked)

    prediction = state["prediction"]
    explanation = state["explanation"]
    patient = state["patient"]

    st.markdown(
        f"<div style='text-align:center; margin: 0 0 1.25rem;'>"
        f"<div style='font-size:0.85rem; color:var(--clr-text-muted); text-transform:uppercase; "
        f"letter-spacing:0.05em;'>Predicted Length of Stay</div>"
        f"<div style='font-size:3.4rem; font-weight:800; color:var(--clr-primary); line-height:1.15;'>"
        f"{format_days(prediction)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    contributors = top_contributing_features(explanation, n=3)

    section_title("Summary")
    st.markdown(_narrative_summary(contributors, prediction))

    observations = _clinical_observations(explanation, patient)
    if observations:
        section_title("Clinical Observations")
        for obs in observations:
            icon_name = "check" if obs["normal"] else "alert-triangle"
            color = config.COLORS["success"] if obs["normal"] else config.COLORS["warning"]
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.6rem; padding:0.3rem 0;'>"
                f"<span style='color:{color}; display:inline-flex;'>{get_icon(icon_name, 18)}</span>"
                f"<span><strong>{obs['label']}:</strong> {obs['status']} ({obs['value_str']})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "The explanation below describes how this prediction was made. "
        "It should support, not replace, clinical judgement."
    )

    section_title("SHAP: Patient Factors Contributing to the Prediction")
    items = all_contributing_features(explanation, n=8)
    st.plotly_chart(_contribution_chart(items), width="stretch")
    st.caption(
        "How to read: Red factors increased the predicted stay, while blue factors "
        "reduced it. Longer bars indicate greater influence on this prediction. "
        "These are model contributions, not direct causes of length of stay."
    )


def render() -> None:
    page_header(
        "Length of Stay Prediction",
        "Enter patient information to generate an individual prediction.",
        icon=get_icon("stethoscope", 26),
    )

    package = load_deployment_package()

    model_name, patient = _entry_form(package)

    if patient is not None:
        prediction, scaled_row = predict(patient, package, model_name)
        explanation = explain(scaled_row, package, model_name)
        state = {
            "patient": patient,
            "prediction": prediction,
            "explanation": explanation,
            "model_name": model_name,
        }
        _record_history(state)
        _show_result_dialog(state, package)
    else:
        # Clicking an entry in the sidebar history navigates here and asks
        # to reopen its result popup (see src/ui/history.py).
        reopened = st.session_state.pop("reopen_history_state", None)
        if reopened is not None:
            _show_result_dialog(reopened, package)
