# streamlit_app/streamlit_shap.py

import os
import joblib
import shap
import pandas as pd
import os
import sys
from pathlib import Path
from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download
import streamlit as st


REPO_ID = "flork-18115/AQI_prediciton_models"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]
# important to import it
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

    # Apply the exact same feature engineering used during training
    featured_df = create_features(historical_df)

    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(
        df=featured_df,
        target_column=target_column)
    return X_test, test_cities

@st.cache_data
def prepare_shap_data(target_column):
    historical_df = load_historical_data()
    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(df=historical_df, target_column=target_column)

    return X_test, test_cities


@st.cache_data
def calculate_shap(model_name, target_column):
    if target_column == "target_24h":
        model = load_model("xgboost_24h.pkl", "xgboost")
    elif target_column == "target_48h":
        model = load_model("catboost_48h.cbm", "catboost")
    else:
        model = load_model("rf_model_72h.pkl", "random_forest")

    X_test, test_cities = prepare_shap_data(target_column)

    st.write("MODEL FEATURES:", model.n_features_in_)
    st.write("X_TEST SHAPE:", X_test.shape)
    st.write("X_TEST COLUMNS:", list(X_test.columns))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return X_test, test_cities, shap_values


def show_shap_analysis(city):
    st.subheader(f"Model Explainability — {city}")

    horizon = st.selectbox("Forecast Horizon", [24, 48, 72], format_func=lambda x: f"{x}-Hour Forecast", key=f"shap_horizon_{city}")
    target_column = f"target_{horizon}h"

    X_test, test_cities, shap_values = calculate_shap("model", target_column)
    city_mask = test_cities == city

    X_city = X_test.loc[city_mask].copy()
    shap_city = shap_values[city_mask]

    if len(X_city) == 0:
        st.warning(f"No test data available for {city}.")
        return

    st.write(f"SHAP analysis is based on {len(X_city):,} test observations for {city}.")

    shap_importance = pd.DataFrame({"Feature": X_city.columns, "Importance": abs(shap_city).mean(axis=0)})
    shap_importance = shap_importance.sort_values("Importance", ascending=False)

    st.markdown("### Feature Importance")
    st.dataframe(shap_importance.head(10), use_container_width=True, hide_index=True)

    st.markdown("### Feature Influence")
    fig = shap.summary_plot(shap_city, X_city, show=False)
    st.pyplot(fig, clear_figure=True)


    st.markdown("### Individual Feature Effects")
    top_features = shap_importance["Feature"].head(3).tolist()

    selected_feature = st.selectbox("Select a feature", top_features, key=f"shap_feature_{city}_{horizon}")

    shap.dependence_plot(selected_feature, shap_city, X_city, show=False)
    st.pyplot(clear_figure=True)