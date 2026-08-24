"""Entry point: page config, navigation and page dispatch."""

import streamlit as st

from src.ui import theme
from src.ui.history import render_sidebar_history
from src.ui.icons import icon
from src.ui.pages import about, feature_importance, home, insights, prediction

st.set_page_config(
    page_title="Hospital Length of Stay Predictor",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject_theme()

PAGES = {
    "Home": home,
    "Prediction": prediction,
    "Model Evaluation": insights,
    "Feature Importance": feature_importance,
    "About": about,
}
PAGE_NAMES = list(PAGES.keys())

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGE_NAMES[0]

if "requested_page" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("requested_page")

with st.sidebar:
    st.markdown(
        "<div style='padding: 0.5rem 0 1rem 0;'>"
        "<span style='font-size:1.3rem; font-weight:700; color: var(--clr-primary); "
        "display:inline-flex; align-items:center; gap:0.4rem;'>"
        f"{icon('hospital', 22)} LOS Predictor</span>"
        "<div style='color: var(--clr-text-muted); font-size:0.85rem;'>Clinical Decision Support</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.radio("Navigate", PAGE_NAMES, key="nav_page", label_visibility="collapsed")
    render_sidebar_history()

PAGES[st.session_state["nav_page"]].render()
