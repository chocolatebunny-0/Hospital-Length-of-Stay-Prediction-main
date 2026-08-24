"""Model Evaluation page: why a model was chosen, its performance, and how it compares."""

import time

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.model.loader import load_deployment_package, load_model_dataset, load_model_results
from src.model.predictor import resolve_model
from src.ui.components import card, render_metric_row, section_title


def _measure_prediction_time(package: dict, model_name: str) -> float:
    """Time a single-row prediction for the given model, in seconds."""
    model = resolve_model(package, model_name)
    dataset = load_model_dataset()
    feature_names = package["feature_names"]
    sample = dataset[feature_names].iloc[[0]]
    scaled = package["scaler"].transform(sample)

    start = time.perf_counter()
    model.predict(scaled)
    return time.perf_counter() - start


def _comparison_chart(df) -> go.Figure:
    sorted_df = df.sort_values("RMSE", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sorted_df["MAE"], y=sorted_df["Model"], name="MAE",
            orientation="h", marker_color=config.COLORS["accent"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=sorted_df["RMSE"], y=sorted_df["Model"], name="RMSE",
            orientation="h", marker_color=config.COLORS["primary"],
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Error (days), lower is better",
        yaxis_title="",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"]),
    )
    return fig


def _training_time_chart(df, highlight: str) -> go.Figure:
    sorted_df = df.sort_values("Training Time (s)")
    colors = [
        config.COLORS["primary"] if model == highlight else config.COLORS["border"]
        for model in sorted_df["Model"]
    ]
    fig = go.Figure(
        go.Bar(
            x=sorted_df["Training Time (s)"], y=sorted_df["Model"],
            orientation="h", marker_color=colors,
            text=sorted_df["Training Time (s)"], textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Training time (seconds)",
        yaxis_title="",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=config.COLORS["text"]),
    )
    return fig


def render_body(model_name: str | None = None) -> None:
    results = load_model_results()
    results_sorted = results.sort_values("RMSE").reset_index(drop=True)
    model_names = results_sorted["Model"].tolist()
    selected_name = model_name if model_name in model_names else model_names[0]
    best_name = model_names[0]

    selected = results_sorted[results_sorted["Model"] == selected_name].iloc[0]

    render_metric_row(
        [
            {
                "label": "Selected Model",
                "value": selected["Model"],
                "sub": "Best overall performer" if selected_name == best_name else "",
            },
            {"label": "Mean Absolute Error", "value": f"≈ {selected['MAE']:.1f} days", "sub": "Average prediction error"},
            {"label": "Root Mean Square Error", "value": f"≈ {selected['RMSE']:.1f} days", "sub": "Penalises larger errors"},
            {"label": "Variation Explained", "value": f"{selected['R²'] * 100:.0f}%", "sub": "R² score"},
        ]
    )

    with card():
        section_title("Which model predicted most accurately?")
        st.caption("Lower values indicate more accurate predictions.")
        st.plotly_chart(_comparison_chart(results), width="stretch")

    with card():
        section_title("Model Training Time")
        st.caption(
            "Training happens only once during development, so longer training times "
            "do not affect how fast predictions are made for new patients."
        )
        st.plotly_chart(_training_time_chart(results, selected_name), width="stretch")

        package = load_deployment_package()
        prediction_time = _measure_prediction_time(package, selected_name)
        render_metric_row(
            [
                {"label": "Training Time", "value": f"{selected['Training Time (s)']:.1f}s", "sub": "One-time, during development"},
                {"label": "Prediction Time", "value": f"{prediction_time * 1000:.0f}ms", "sub": "What a user actually waits for"},
            ]
        )

    with st.expander("View Detailed Comparison Table"):
        st.dataframe(results_sorted, width="stretch", hide_index=True)
