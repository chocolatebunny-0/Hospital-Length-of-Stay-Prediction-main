"""Global theme injection: CSS that overrides Streamlit's default appearance."""

import streamlit as st

from src.config import COLORS

_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    :root {{
        --clr-primary: {COLORS["primary"]};
        --clr-primary-light: {COLORS["primary_light"]};
        --clr-accent: {COLORS["accent"]};
        --clr-bg: {COLORS["background"]};
        --clr-surface: {COLORS["surface"]};
        --clr-text: {COLORS["text"]};
        --clr-text-muted: {COLORS["text_muted"]};
        --clr-border: {COLORS["border"]};
        --clr-success: {COLORS["success"]};
        --clr-warning: {COLORS["warning"]};
        --clr-danger: {COLORS["danger"]};
    }}

    .stApp {{
        background: var(--clr-bg);
    }}

    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    h1, h2, h3, h4 {{
        color: var(--clr-text);
        font-weight: 700;
        letter-spacing: -0.01em;
    }}

    p, li, span, label {{
        color: var(--clr-text);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: var(--clr-surface);
        border-right: 1px solid var(--clr-border);
    }}

    /* Buttons */
    .stButton > button, .stButton > button p {{
        color: white !important;
    }}
    .stButton > button {{
        background: var(--clr-primary);
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: background 0.15s ease, transform 0.1s ease;
    }}
    .stButton > button[kind="primary"], .stButton > button[kind="primaryFormSubmit"] {{
        background: var(--clr-primary);
    }}
    .stButton > button:hover {{
        background: var(--clr-primary-light);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Sidebar prediction-history entries */
    section[data-testid="stSidebar"] .stButton > button {{
        background: var(--clr-primary) !important;
        color: white !important;
        border: none !important;
        text-align: left;
        justify-content: flex-start;
    }}
    section[data-testid="stSidebar"] .stButton > button p {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: var(--clr-primary-light) !important;
    }}

    /* Floating "Make a Prediction" button on the home page. */
    div[class*="st-key-floating_predict_btn"] {{
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        z-index: 999;
        width: auto;
    }}
    div[class*="st-key-floating_predict_btn"] .stButton > button {{
        border-radius: 999px;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 14px rgba(15, 110, 99, 0.35);
        background: var(--clr-primary);
        color: #fff;
    }}

    /* Generic card: Streamlit's native bordered container, restyled.
       Used via the `card()` context manager in src/ui/components.py, which
       gives each container a key "card_N" -> CSS class "st-key-card_N". */
    div[class*="st-key-card_"] {{
        background: rgba(15, 110, 99, 0.06) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 3px rgba(15, 110, 99, 0.06);
        margin-bottom: 1.25rem;
        padding: 1.5rem 1.75rem;
    }}

    div[class*="st-key-card_"] h3,
    div[class*="st-key-card_"] h4 {{
        margin-top: 0;
    }}

    /* Metric tile */
    .metric-tile {{
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: left;
    }}
    .metric-tile .metric-label {{
        font-size: 0.82rem;
        color: var(--clr-text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }}
    .metric-tile .metric-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--clr-primary);
        line-height: 1.1;
    }}
    .metric-tile .metric-sub {{
        font-size: 0.82rem;
        color: var(--clr-text-muted);
        margin-top: 0.25rem;
    }}

    /* Pill / badge */
    .pill {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(15, 110, 99, 0.1);
        color: var(--clr-primary);
    }}

    .section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--clr-primary);
        margin-bottom: 0.75rem;
        margin-top: 0.25rem;
    }}

    hr {{
        border: none;
        border-top: 1px solid var(--clr-primary);
        opacity: 0.35;
    }}

    /* Dataframe / table polish */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--clr-border);
        border-radius: 12px;
        overflow: hidden;
    }}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
