"""One-off script: reproduce the notebook's preprocessing pipeline and train all
five candidate regression models (not just the selected Random Forest), so the
Streamlit app can offer a model selector.

Preprocessing exactly mirrors the notebook: drop noise columns, median/mode
impute, one-hot encode (drop_first=True), 80/20 split with random_state=42,
StandardScaler fit on the training split.

Note: the notebook's model-comparison loop had a bug where evaluate_model()
fit every model on the *unscaled* X_train (its X_train_scaled/X_test_scaled
parameters were unused), except for the final selected model, which was
explicitly refit on X_train_scaled before deployment (see notebook cell 55).
For the app to work correctly, every model must expect the same scaled input
format, so all five models here are fit on X_train_scaled, matching how the
deployed Random Forest was actually fit.

Run once: py -3 scripts/retrain_all_models.py
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RAW_DATASET_PATH = "data/healthcare_risk_factors_dataset.csv"
OUTPUT_PATH = "artifacts/hospital_los_predictor.pkl"
TARGET = "LengthOfStay"
NOISE_COLUMNS = ["random_notes", "noise_col"]
RANDOM_STATE = 42

df = pd.read_csv(RAW_DATASET_PATH)
model_df = df.drop(columns=NOISE_COLUMNS)

numerical_columns = model_df.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET)
categorical_columns = model_df.select_dtypes(include="object").columns

for column in numerical_columns:
    model_df[column] = model_df[column].fillna(model_df[column].median())
for column in categorical_columns:
    model_df[column] = model_df[column].fillna(model_df[column].mode()[0])

medical_conditions = sorted(df["Medical Condition"].dropna().unique().tolist())
gender_options = sorted(df["Gender"].dropna().unique().tolist())

model_df = pd.get_dummies(model_df, columns=categorical_columns, drop_first=True)

X = model_df.drop(columns=TARGET)
y = model_df[TARGET]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

models = {
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
}

results = []
fitted_models = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    fitted_models[name] = model
    results.append(
        {
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R²": round(r2, 3),
            "Training Time (s)": round(training_time, 3),
        }
    )
    print(f"{name}: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} time={training_time:.2f}s")

results_df = (
    pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
)
results_df.insert(0, "Rank", results_df.index + 1)

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")

deployment_package = {
    "model": fitted_models[best_model_name],
    "models": fitted_models,
    "default_model_name": best_model_name,
    "scaler": scaler,
    "feature_names": feature_names,
    "results": results_df,
    "target_variable": TARGET,
    "medical_conditions": medical_conditions,
    "gender_options": gender_options,
}

joblib.dump(deployment_package, OUTPUT_PATH)
results_df.to_csv("artifacts/model_results.csv", index=False)
print(f"\nSaved deployment package to {OUTPUT_PATH}")
