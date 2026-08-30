import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download


REPO_ID = "flork-18115/AQI_prediciton_models"

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]

from models.preprocessing import preprocess_data
from models.supabase_data import get_historical_data
from models.feature_engineering import create_features

@st.cache_data
def load_historical_data():
    return get_historical_data()

@st.cache_resource
def load_model(filename, model_type):
    model_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="model", token=os.getenv("HF_TOKEN"))

    if model_type == "catboost":
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model

    return joblib.load(model_path)

@st.cache_data
def prepare_shap_data(target_column):
    historical_df = load_historical_data()

    # Same feature engineering used during model training
    featured_df = create_features(historical_df)

    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(df=featured_df, target_column=target_column)

    return X_test, test_cities

@st.cache_data
def calculate_shap(target_column):
    if target_column == "target_24h":
        model = load_model("xgboost_24h.pkl", "xgboost")
    elif target_column == "target_48h":
        model = load_model("catboost_48h.cbm", "catboost")
    else:
        model = load_model("rf_model_72h.pkl", "random_forest")

    X_test, test_cities = prepare_shap_data(target_column)
    st.write(f"Model: {target_column}")
    st.write(f"X_TEST SHAPE: {X_test.shape}")
    st.write(f"Cities in SHAP data: {sorted(set(test_cities))}")

    if target_column == "target_48h":
        expected_features = model.get_feature_count()
    else:
        expected_features = model.n_features_in_

    st.write(f"MODEL FEATURES: {expected_features}")

    if expected_features != X_test.shape[1]:
        raise ValueError(f"Feature mismatch for {target_column}: model expects {expected_features} features, but X_test has {X_test.shape[1]}.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return X_test, test_cities, shap_values


def show_shap_analysis(city):
    st.divider()

    st.subheader(f"Model Explainability — {city}")
    horizon = st.radio("Forecast Horizon", [24, 48, 72], format_func=lambda x: f"{x}-Hour Forecast", horizontal=True, key=f"shap_horizon_{city}")
    target_column = f"target_{horizon}h"

    try:
        X_test, test_cities, shap_values = calculate_shap(target_column)
    except Exception as e:
        st.error(f"Could not calculate SHAP analysis: {e}")
        return
    city_mask = test_cities == city

    X_city = X_test.loc[city_mask].copy()
    shap_city = shap_values[city_mask]

    if len(X_city) == 0:
        st.warning(f"No test data available for {city}.")
        return

    st.write(f"SHAP analysis is based on {len(X_city):,} test observations for {city} using the {horizon}-hour model.")

    # FEATURE IMPORTANCE
    shap_importance = pd.DataFrame({"Feature": X_city.columns, "Importance": abs(shap_city).mean(axis=0)})
    shap_importance = shap_importance.sort_values("Importance", ascending=False).reset_index(drop=True)

    # Feature importance table
    st.markdown("### Feature Importance")
    st.dataframe(shap_importance.head(10), use_container_width=True, hide_index=True)

    # Feature importance graph
    st.markdown("### Top 10 Features")
    importance_chart = shap_importance.head(10).sort_values("Importance").set_index("Feature")

    st.bar_chart(importance_chart["Importance"], horizontal=True, use_container_width=True)

    # SHAP SUMMARY PLOT
    st.markdown("### Feature Influence")
    plt.figure()

    shap.summary_plot(shap_city, X_city, show=False)
    fig = plt.gcf()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.markdown("### Individual Feature Effects")
    top_features = shap_importance.head(3)["Feature"].tolist()

    selected_feature = st.selectbox("Select a feature", top_features, key=f"shap_feature_{city}_{horizon}")

    plt.figure()

    shap.dependence_plot(selected_feature, shap_city, X_city, show=False)
    fig = plt.gcf()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)