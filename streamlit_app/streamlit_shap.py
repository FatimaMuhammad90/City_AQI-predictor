import os
import sys
from pathlib import Path
import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from catboost import CatBoostRegressor, Pool
from huggingface_hub import hf_hub_download


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]


from models.preprocessing import preprocess_data
from models.supabase_data import get_historical_data
from models.feature_engineering import create_features


REPO_ID = "flork-18115/AQI_prediciton_models"
SHAP_SAMPLE_SIZE = 500


@st.cache_data
def load_historical_data():
    return get_historical_data()


@st.cache_data
def prepare_shap_data(target_column):
    historical_df = load_historical_data()

    featured_df = create_features(historical_df)

    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(df=featured_df, target_column=target_column)

    X_test = X_test.select_dtypes(
        include=["number", "bool"]
    ).copy()

    X_test = X_test.astype(float)

    test_cities = pd.Series(
        test_cities
    ).reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)

    return X_test, test_cities


@st.cache_resource
def load_model(filename, model_type):
    model_path = hf_hub_download( repo_id=REPO_ID, filename=filename,repo_type="model",token=os.getenv("HF_TOKEN"),)
    if model_type == "catboost":
        model = CatBoostRegressor()
        model.load_model(
            model_path
        )
        return model

    return joblib.load(
        model_path
    )


def get_model_feature_names(model, model_type):

    if model_type == "xgboost":

        if hasattr(model, "feature_names_in_"):
            names = list(
                model.feature_names_in_
            )

            if names:
                return names

        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            names = booster.feature_names

            if names:
                return list(names)

        return None

    if model_type == "random_forest":
        if hasattr(model, "feature_names_in_"):
            names = list(
                model.feature_names_in_
            )

            if names:
                return names

        return None

    if model_type == "catboost":
        try:
            names = model.feature_names_

            if names:
                return list(names)

        except Exception:
            pass

        return None

    return None


def align_features(X_test, model, model_type):
    X_test = X_test.copy()

    model_feature_names = (
        get_model_feature_names(
            model,
            model_type
        )
    )

    if model_feature_names:
        missing = [
            feature
            for feature in model_feature_names
            if feature not in X_test.columns
        ]

        if missing:
            raise ValueError(
                "SHAP feature mismatch.\n\n"
                f"The model requires these features, "
                f"but they are missing from X_test:\n"
                f"{missing}"
            )

        X_test = X_test[
            model_feature_names
        ]

        return X_test

    if model_type in (
        "xgboost",
        "random_forest",
    ):
        if hasattr(
            model,
            "n_features_in_"
        ):
            expected = (
                model.n_features_in_
            )

            actual = X_test.shape[1]

            if expected != actual:
                raise ValueError(
                    f"Feature mismatch: "
                    f"model expects {expected} "
                    f"features but X_test has "
                    f"{actual}."
                )

    return X_test


@st.cache_data
def calculate_shap(target_column, city):

    if target_column == "target_24h":
        model = load_model("xgboost_24h.pkl", "xgboost")
        model_type = "xgboost"

    elif target_column == "target_48h":
        model = load_model("catboost_48h.cbm", "catboost")
        model_type = "catboost"

    else:
        model = load_model("rf_model_72h.pkl", "random_forest")
        model_type = "random_forest"

    X_test, test_cities = prepare_shap_data(target_column)

    X_test = X_test.copy()

    X_test = X_test.select_dtypes(
        include=["number", "bool"]
    ).astype(float)

    test_cities = pd.Series(test_cities).reset_index(drop=True)

    city_mask = (test_cities == city).values

    X_city = X_test.loc[city_mask].copy()

    if len(X_city) > SHAP_SAMPLE_SIZE:
        X_city = X_city.sample(
            n=SHAP_SAMPLE_SIZE,
            random_state=42
        )

    if model_type == "xgboost":
        expected_features = model.n_features_in_

        if expected_features != X_city.shape[1]:
            raise ValueError(
                f"Feature mismatch for {target_column}: "
                f"model expects {expected_features} features, "
                f"but X_test has {X_city.shape[1]}."
            )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_city)

    elif model_type == "random_forest":
        expected_features = model.n_features_in_

        if expected_features != X_city.shape[1]:
            raise ValueError(
                f"Feature mismatch for {target_column}: "
                f"model expects {expected_features} features, "
                f"but X_test has {X_city.shape[1]}."
            )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_city)

    else:
        feature_names = list(X_city.columns)

        model.set_feature_names(feature_names)

        test_pool = Pool(
            X_city,
            feature_names=feature_names
        )

        shap_result = model.get_feature_importance(
            data=test_pool,
            type="ShapValues"
        )

        shap_values = shap_result[:, :-1]

    return X_city, shap_values


def show_shap_analysis(city):

    st.divider()

    st.subheader(
        f"Model Explainability — {city}"
    )

    horizon = st.radio(
        "Forecast Horizon",
        [24, 48, 72],
        format_func=lambda x:
            f"{x}-Hour Forecast",
        horizontal=True,
        key=f"shap_horizon_{city}",
    )

    target_column = (
        f"target_{horizon}h"
    )

    try:
        (
            X_city,
            shap_values,
        ) = calculate_shap(
            target_column,
            city,
        )

    except Exception as e:
        st.error(
            f"Could not calculate SHAP "
            f"analysis:\n\n{e}"
        )
        return

    if len(X_city) == 0:
        st.warning(
            f"No test data available "
            f"for {city}."
        )
        return

    st.write(
        f"SHAP analysis is based on "
        f"{len(X_city):,} observations "
        f"for {city} using the "
        f"{horizon}-hour model."
    )

    shap_importance = pd.DataFrame(
        {
            "Feature": X_city.columns,
            "Importance": (
                abs(shap_values)
                .mean(axis=0)
            ),
        }
    )

    shap_importance = (
        shap_importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    st.markdown(
        "### Feature Importance"
    )

    st.dataframe(
        shap_importance.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Top 10 Features"
    )

    importance_chart = (
        shap_importance
        .head(10)
        .sort_values(
            "Importance"
        )
        .set_index(
            "Feature"
        )
    )

    st.bar_chart(
        importance_chart[
            "Importance"
        ],
        horizontal=True,
        use_container_width=True,
    )

    st.markdown(
        "### Feature Influence"
    )

    fig = plt.figure()

    shap.summary_plot(
        shap_values,
        X_city,
        show=False,
    )

    fig = plt.gcf()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)

    st.markdown(
        "### Individual Feature Effects"
    )

    top_features = (
        shap_importance
        .head(3)[
            "Feature"
        ]
        .tolist()
    )

    selected_feature = (
        st.selectbox(
            "Select a feature",
            top_features,
            key=(
                f"shap_feature_"
                f"{city}_{horizon}"
            ),
        )
    )

    fig = plt.figure()

    shap.dependence_plot(
        selected_feature,
        shap_values,
        X_city,
        show=False,
    )

    fig = plt.gcf()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)