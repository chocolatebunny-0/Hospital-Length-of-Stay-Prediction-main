"""Loading of the pre-trained deployment package. No retraining happens here."""

import joblib
import pandas as pd
import streamlit as st

from src import config


@st.cache_resource(show_spinner="Loading prediction model...")
def load_deployment_package() -> dict:
    """Load the notebook-produced deployment package unchanged.

    Returns the dict saved by the notebook:
    {"model", "scaler", "feature_names", "results"}.
    """
    return joblib.load(config.DEPLOYMENT_PACKAGE_PATH)


@st.cache_data(show_spinner=False)
def load_model_results() -> pd.DataFrame:
    """Load the model comparison table saved from the notebook."""
    return pd.read_csv(config.MODEL_RESULTS_PATH)


@st.cache_data(show_spinner=False)
def load_model_dataset() -> pd.DataFrame:
    """Load the fully preprocessed (encoded, unscaled) dataset saved from the notebook."""
    return pd.read_csv(config.MODEL_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    """Load the original raw dataset, used for descriptive context only."""
    return pd.read_csv(config.RAW_DATASET_PATH)
