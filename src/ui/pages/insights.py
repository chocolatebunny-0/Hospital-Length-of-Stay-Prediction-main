"""Model Evaluation page: compares the evaluated models and explains why the deployed one was chosen."""

import streamlit as st

from src.model.loader import load_deployment_package
from src.ui.components import card, page_header, section_title
from src.ui.icons import icon as get_icon
from src.ui.pages import performance


def render() -> None:
    page_header(
        "Model Evaluation",
        "Compare the performance of the evaluated machine learning models and "
        "understand why the selected model was chosen for deployment.",
        icon=get_icon("chart-bar", 26),
    )

    package = load_deployment_package()
    model_names = list(package.get("models", {}).keys()) or [package.get("default_model_name", "Random Forest")]
    default_name = package.get("default_model_name", model_names[0])

    with card():
        section_title("Select Model")
        selected_name = st.selectbox(
            "Select model",
            model_names,
            index=model_names.index(default_name),
            label_visibility="collapsed",
            help="Random Forest is the model deployed for prediction, but you can "
            "view how any of the five evaluated models performed.",
        )

    performance.render_body(selected_name)
