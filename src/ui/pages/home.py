"""Home page: an executive Hospital Operations Dashboard summarising the training dataset."""

import math

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_raw_dataset
from src.ui.components import card, page_header, render_metric_row
from src.ui.icons import icon as get_icon

_CHART_HEIGHT = 320
_GREEN = config.COLORS["primary"]
_GREEN_LIGHT = config.COLORS["primary_light"]


def _chart_layout(fig: go.Figure, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    fig.update_layout(
        height=_CHART_HEIGHT,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"], size=12),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(15, 110, 99, 0.08)")
    fig.update_yaxes(gridcolor="rgba(15, 110, 99, 0.08)")
    return fig


def _length_of_stay_histogram(df) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=df["LengthOfStay"],
            marker_color=_GREEN,
            nbinsx=19,
            hovertemplate="Length of Stay: %{x} days<br>Patients: %{y}<extra></extra>",
        )
    )
    return _chart_layout(fig, "Length of Stay (Days)", "Number of Patients")


def _condition_distribution_chart(df) -> go.Figure:
    counts = df["Medical Condition"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker_color=_GREEN,
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:,} patients<extra></extra>",
        )
    )
    return _chart_layout(fig, "Number of Patients", "")


def _avg_los_by_condition_chart(df) -> go.Figure:
    avg_los = df.groupby("Medical Condition")["LengthOfStay"].mean().sort_values()
    fig = go.Figure(
        go.Bar(
            x=avg_los.values, y=avg_los.index, orientation="h",
            marker_color=_GREEN,
            text=[f"{v:.1f}" for v in avg_los.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f} days<extra></extra>",
        )
    )
    return _chart_layout(fig, "Average Length of Stay (Days)", "")


def _grouped_metric_chart(df, columns: list[str]) -> go.Figure:
    values = [df[col].mean() for col in columns]
    order = sorted(range(len(columns)), key=lambda i: values[i])
    sorted_columns = [columns[i] for i in order]
    sorted_values = [values[i] for i in order]
    fig = go.Figure(
        go.Bar(
            x=sorted_values, y=sorted_columns, orientation="h",
            marker_color=_GREEN,
            text=[f"{v:.1f}" for v in sorted_values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    return _chart_layout(fig, "Average Value", "")


def _lifestyle_risk_chart(df) -> go.Figure:
    labels = ["Smoking", "Alcohol", "Family History"]
    percentages = [df[col].mean() * 100 for col in labels]
    order = sorted(range(len(labels)), key=lambda i: percentages[i])
    sorted_labels = [labels[i] for i in order]
    sorted_percentages = [percentages[i] for i in order]
    fig = go.Figure(
        go.Bar(
            x=sorted_percentages, y=sorted_labels, orientation="h",
            marker_color=_GREEN_LIGHT,
            text=[f"{v:.0f}%" for v in sorted_percentages],
            textposition="outside",
            hovertemplate="%{y}: %{x:.0f}%% of patients<extra></extra>",
        )
    )
    return _chart_layout(fig, "% of Patients", "")


def render() -> None:
    page_header(
        "Hospital Operations Dashboard",
        "Executive overview of the patient dataset used to develop the Length of Stay prediction model.",
        icon=get_icon("hospital", 26),
    )

    df = load_raw_dataset()

    with card():
        render_metric_row(
            [
                {"label": "Total Patients", "value": f"{len(df):,}"},
                {"label": "Avg. Length of Stay", "value": f"{math.ceil(df['LengthOfStay'].mean())} days"},
                {"label": "Medical Conditions", "value": str(df["Medical Condition"].nunique())},
            ]
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    if st.button(
        ":material/monitor_heart: Make a Prediction",
        key="floating_predict_btn",
        help="Go to the prediction page",
    ):
        st.session_state["requested_page"] = "Prediction"
        st.rerun()

    row1 = st.columns(2)
    with row1[0]:
        with card():
            st.markdown("**Distribution of Length of Stay**")
            st.plotly_chart(_length_of_stay_histogram(df), width="stretch")
    with row1[1]:
        with card():
            st.markdown("**Patients by Medical Condition**")
            st.plotly_chart(_condition_distribution_chart(df), width="stretch")

    row2 = st.columns(2)
    with row2[0]:
        with card():
            st.markdown("**Average Length of Stay by Condition**")
            st.plotly_chart(_avg_los_by_condition_chart(df), width="stretch")
    with row2[1]:
        with card():
            st.markdown("**Average Clinical Measurements**")
            clinical_cols = [
                "Glucose", "Blood Pressure", "BMI", "Cholesterol",
                "Triglycerides", "HbA1c", "Oxygen Saturation",
            ]
            st.plotly_chart(_grouped_metric_chart(df, clinical_cols), width="stretch")

    row3 = st.columns(2)
    with row3[0]:
        with card():
            st.markdown("**Lifestyle Risk Profile**")
            st.plotly_chart(_lifestyle_risk_chart(df), width="stretch")
    with row3[1]:
        with card():
            st.markdown("**Wellness Indicators**")
            wellness_cols = ["Physical Activity", "Diet Score", "Stress Level", "Sleep Hours"]
            st.plotly_chart(_grouped_metric_chart(df, wellness_cols), width="stretch")
