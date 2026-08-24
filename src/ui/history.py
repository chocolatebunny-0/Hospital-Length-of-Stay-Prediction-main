"""Sidebar prediction-history panel, shared across all pages via app.py."""

import streamlit as st

from src.utils.formatting import format_days


def render_sidebar_history() -> None:
    """Fills the sidebar's remaining space with a running log of predictions made
    this session. Clicking an entry jumps to the Prediction page and reopens
    that prediction's result popup.
    """
    st.markdown(
        "<div style='color: var(--clr-text-muted); font-size:0.78rem; font-weight:600; "
        "text-transform:uppercase; letter-spacing:0.05em; margin: 1.5rem 0 0.5rem;'>"
        "Prediction History</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state.get("prediction_history")
    if not history:
        st.caption("No predictions yet this session. Results will appear here.")
        return

    for i, entry in enumerate(history):
        patient = entry["patient"]
        label = f"{patient.medical_condition} · {format_days(entry['prediction'])}"
        help_text = f"{patient.gender}, {patient.age:.0f} · {entry['model_name']}"
        if st.button(label, key=f"history_entry_{i}", width="stretch", help=help_text):
            st.session_state["reopen_history_state"] = entry
            st.session_state["requested_page"] = "Prediction"
            st.rerun()
