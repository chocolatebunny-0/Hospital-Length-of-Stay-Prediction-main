"""Reusable UI building blocks shared across pages."""

import itertools
from contextlib import contextmanager

import streamlit as st

_card_counter = itertools.count()


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    icon_html = (
        f"<span style='display:inline-flex; color: var(--clr-primary); "
        f"margin-right: 0.5rem;'>{icon}</span>"
        if icon
        else ""
    )
    st.markdown(
        f"<h2 style='display:flex; align-items:center;'>{icon_html}{title}</h2>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<p style='color: var(--clr-text-muted); margin-top: -0.5rem;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def metric_tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return (
        "<div class='metric-tile'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"{sub_html}"
        "</div>"
    )


def render_metric_row(metrics: list[dict]) -> None:
    """metrics: list of {"label", "value", "sub"} dicts, rendered in equal columns."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(
                metric_tile(m["label"], m["value"], m.get("sub", "")),
                unsafe_allow_html=True,
            )


@contextmanager
def card():
    """A bordered card container. Use as `with card(): ...` instead of manually
    opening/closing a <div> across separate st.markdown calls -- Streamlit parses
    each markdown call as its own isolated HTML fragment, so an unclosed <div>
    opened in one call and closed in another never actually wraps the widgets
    rendered in between; it just leaves a stray empty, styled div behind.

    Each card gets a unique key ("card_N") so Streamlit doesn't raise a
    duplicate-element error; every such key becomes a CSS class "st-key-card_N",
    which theme.py styles via a "st-key-card" prefix match.
    """
    key = f"card_{next(_card_counter)}"
    with st.container(border=True, key=key):
        yield


def pill(text: str) -> str:
    return f"<span class='pill'>{text}</span>"
