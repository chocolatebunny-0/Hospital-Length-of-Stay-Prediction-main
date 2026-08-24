"""Feature Importance page: per-model SHAP-based global feature importance."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src import config
from src.model.loader import load_deployment_package, load_model_dataset
from src.model.predictor import TREE_MODEL_TYPES, resolve_model
from src.ui.components import card, page_header, section_title
from src.ui.icons import icon as get_icon

_SAMPLE_SIZE = 150


@st.cache_data(show_spinner="Computing SHAP feature importance for this model...")
def _global_importance(_package: dict, model_name: str) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, over a sample of the dataset, for one model."""
    feature_names = _package["feature_names"]
    scaler = _package["scaler"]
    dataset = load_model_dataset()
    sample = dataset[feature_names].sample(
        n=min(_SAMPLE_SIZE, len(dataset)), random_state=config.RANDOM_STATE
    )
    scaled = scaler.transform(sample)
    scaled_df = pd.DataFrame(scaled, columns=feature_names, index=sample.index)

    model = resolve_model(_package, model_name)
    if type(model).__name__ in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(scaled_df))
    else:
        explainer = shap.Explainer(model.predict, scaled_df)
        shap_values = explainer(scaled_df).values

    mean_abs = np.abs(shap_values).mean(axis=0)
    display_names = [config.DISPLAY_NAME_OVERRIDES.get(name, name) for name in feature_names]

    importance = pd.DataFrame({"Feature": display_names, "Mean |SHAP value|": mean_abs})
    return importance.sort_values("Mean |SHAP value|", ascending=False).reset_index(drop=True)


def _importance_chart(df: pd.DataFrame, color: str, x_max: float | None = None) -> go.Figure:
    sorted_df = df.sort_values("Mean |SHAP value|")
    fig = go.Figure(
        go.Bar(
            x=sorted_df["Mean |SHAP value|"],
            y=sorted_df["Feature"],
            orientation="h",
            marker_color=color,
        )
    )
    fig.update_layout(
        xaxis_title="Mean |SHAP value| (days), average impact on the prediction",
        xaxis_range=[0, x_max] if x_max else None,
        yaxis_title="",
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"]),
    )
    return fig


def render() -> None:
    page_header(
        "Feature Importance",
        "See which patient factors most influence each model's predictions, "
        "based on average SHAP impact across a sample of the training data.",
        icon=get_icon("chart-bar", 26),
    )

    package = load_deployment_package()
    model_names = list(package.get("models", {}).keys()) or [
        package.get("default_model_name", "Random Forest")
    ]
    default_name = package.get("default_model_name", model_names[0])

    with card():
        section_title("Select Model")
        compare_mode = st.toggle(
            "Compare two models",
            help="Show the SHAP-based feature importance for two models side by side.",
        )

        if compare_mode:
            col_a, col_b = st.columns(2)
            with col_a:
                name_a = st.selectbox(
                    "Model A", model_names, index=model_names.index(default_name)
                )
            with col_b:
                other_names = [m for m in model_names if m != name_a] or model_names
                name_b = st.selectbox("Model B", other_names, index=0)
        else:
            selected_name = st.selectbox(
                "Select model",
                model_names,
                index=model_names.index(default_name),
                label_visibility="collapsed",
                help="View the SHAP-based feature importance for any of the evaluated models.",
            )

    if compare_mode:
        importance_a = _global_importance(package, name_a)
        importance_b = _global_importance(package, name_b)
        x_max = max(importance_a["Mean |SHAP value|"].max(), importance_b["Mean |SHAP value|"].max()) * 1.05

        col_a, col_b = st.columns(2)
        with col_a:
            with card():
                section_title(name_a)
                st.plotly_chart(
                    _importance_chart(importance_a, config.COLORS["primary"], x_max),
                    width="stretch",
                )
        with col_b:
            with card():
                section_title(name_b)
                st.plotly_chart(
                    _importance_chart(importance_b, config.COLORS["accent"], x_max),
                    width="stretch",
                )

        with st.expander("View Feature Importance Table"):
            merged = importance_a.merge(
                importance_b, on="Feature", suffixes=(f" ({name_a})", f" ({name_b})")
            ).sort_values(f"Mean |SHAP value| ({name_a})", ascending=False)
            st.dataframe(merged, width="stretch", hide_index=True)
    else:
        importance = _global_importance(package, selected_name)

        with card():
            section_title(f"{selected_name}: What Drives Its Predictions")
            st.caption(
                "Features are ranked by their average absolute effect on the predicted "
                "length of stay, in days, across a sample of patients."
            )
            st.plotly_chart(_importance_chart(importance, config.COLORS["primary"]), width="stretch")

        with st.expander("View Feature Importance Table"):
            st.dataframe(importance, width="stretch", hide_index=True)
