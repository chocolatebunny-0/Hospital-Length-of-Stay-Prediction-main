"""About page: project overview, workflow, technologies, dataset, author."""

import streamlit as st

from src.model.loader import load_raw_dataset
from src.ui.components import card, page_header, render_metric_row, section_title
from src.ui.icons import icon as get_icon


def render() -> None:
    page_header("About This Project", icon=get_icon("info", 26))

    with card():
        section_title("Project Overview")
        st.markdown(
            """
            This application predicts hospital length of stay using a Random
            Forest model trained on a healthcare risk-factors dataset combining
            demographic, clinical and lifestyle information for 30,000 patients.
            It was developed as part of an MSc dissertation project exploring how
            predictive modelling and explainable AI can support clinical
            decision-making around hospital resource planning.

            Reliable length-of-stay estimates support bed and staffing planning,
            discharge coordination and early identification of patients who may
            require extended care.
            """
        )

    with card():
        section_title("Dataset Summary")
        df = load_raw_dataset()
        render_metric_row(
            [
                {"label": "Patient Records", "value": f"{len(df):,}"},
                {"label": "Original Variables", "value": str(df.shape[1])},
                {"label": "Predictors Used", "value": "22"},
                {"label": "Target Variable", "value": "LengthOfStay"},
            ]
        )
        st.caption(
            "Source: Healthcare Risk Factors Dataset. Two non-informative columns "
            "(random text and synthetic noise) were identified and removed during "
            "preprocessing."
        )

    with card():
        section_title("Machine Learning Workflow")
        st.markdown(
            """
            1. **Data understanding:** exploratory analysis of 30,000 patient
               records and 20 variables, including missing-value and correlation
               analysis.
            2. **Data preparation:** removal of non-informative columns, median
               imputation for missing numerical values, mode imputation for
               missing categorical values, one-hot encoding of categorical
               variables, an 80/20 train-test split and feature standardisation
               with `StandardScaler`.
            3. **Model development:** five regression models were trained and
               evaluated on the same test set: Linear Regression, Decision Tree,
               Random Forest, Gradient Boosting and XGBoost.
            4. **Model selection:** Random Forest was selected based on the
               lowest RMSE and highest R² among the candidates.
            5. **Explainability:** SHAP (TreeExplainer) was used to generate
               global and local explanations of the selected model's predictions.
            6. **Deployment:** the fitted model, scaler and feature metadata
               were packaged for use in this Streamlit application, without any
               retraining or changes to preprocessing.
            """
        )

    with card():
        section_title("Why Random Forest Was Selected")
        st.markdown(
            """
            Five regression models were trained and evaluated on the same
            held-out test set: Linear Regression, Decision Tree, Random Forest,
            Gradient Boosting and XGBoost. **Random Forest** achieved the lowest
            RMSE and one of the lowest MAE values, alongside the highest R²
            score, indicating both the smallest average prediction error and the
            strongest ability to explain variation in hospital length of stay.
            Although it required the longest training time, this was a one-time
            cost during development and does not affect prediction speed at
            inference.
            """
        )

    with card():
        section_title("Understanding SHAP Explanations")
        st.markdown(
            """
            SHAP (SHapley Additive exPlanations) is a way of breaking down a
            model's prediction into the contribution of each individual patient
            detail. Start from the model's average prediction, then add or
            subtract a little for each detail (age, glucose, medical condition,
            and so on) based on how it compares to a typical patient. Add all of
            those up and you get back the model's final prediction for that
            specific patient, so you can see not just *what* it predicted, but
            *why*.

            Across this dataset, medical condition, glucose, stress level,
            HbA1c, blood pressure and sleep tend to matter most to the model,
            while smoking, alcohol use and gender tend to matter least. The
            **Prediction** page shows this breakdown for an individual
            patient after each prediction. It's meant to support clinical
            judgement, not replace it.
            """
        )

    with card():
        section_title("Technologies Used")
        st.markdown(
            """
            - **Python:** core language for analysis and application development
            - **pandas / NumPy:** data manipulation
            - **scikit-learn:** preprocessing, model training and evaluation
            - **XGBoost:** gradient boosting model comparison
            - **SHAP:** model explainability
            - **Streamlit:** interactive web application
            - **Plotly / Matplotlib:** data visualisation
            """
        )

    with card():
        section_title("Author")
        st.markdown(
            """
            **Author:** _Add your name here_
            **Programme:** _Add MSc programme / institution here_
            **Contact:** _Add contact email here_
            """
        )
