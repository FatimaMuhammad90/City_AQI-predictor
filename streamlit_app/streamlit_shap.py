import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from catboost import CatBoostRegressor, Pool
from huggingface_hub import hf_hub_download


# ============================================================
# PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]


from models.preprocessing import preprocess_data
from models.supabase_data import get_historical_data
from models.feature_engineering import create_features


# ============================================================
# CONSTANTS
# ============================================================

REPO_ID = "flork-18115/AQI_prediciton_models"


# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical_data():

    return get_historical_data()


# ============================================================
# FEATURE PREPARATION
# ============================================================

@st.cache_data
def prepare_shap_data(target_column):

    historical_df = load_historical_data()

    # --------------------------------------------------------
    # Same feature engineering used by the model
    # --------------------------------------------------------

    featured_df = create_features(
        historical_df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        train,
        test,
        df_processed,
        test_cities,
        test_origins,
    ) = preprocess_data(
        df=featured_df,
        target_column=target_column,
    )

    # --------------------------------------------------------
    # Keep only numerical columns
    # --------------------------------------------------------

    X_test = X_test.select_dtypes(
        include=["number", "bool"]
    ).copy()

    X_test = X_test.astype(float)

    test_cities = pd.Series(
        test_cities
    ).reset_index(drop=True)

    # Make sure indexes match
    X_test = X_test.reset_index(drop=True)

    return X_test, test_cities


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(filename, model_type):

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
    )

    if model_type == "catboost":

        model = CatBoostRegressor()

        model.load_model(
            model_path
        )

        return model

    return joblib.load(
        model_path
    )


# ============================================================
# MODEL FEATURE NAMES
# ============================================================

def get_model_feature_names(model, model_type):

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if model_type == "xgboost":

        # Some sklearn XGBoost models expose this
        if hasattr(model, "feature_names_in_"):

            names = list(
                model.feature_names_in_
            )

            if names:
                return names

        # Otherwise inspect underlying booster
        if hasattr(model, "get_booster"):

            booster = model.get_booster()

            names = booster.feature_names

            if names:
                return list(names)

        return None

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if model_type == "random_forest":

        if hasattr(model, "feature_names_in_"):

            names = list(
                model.feature_names_in_
            )

            if names:
                return names

        return None

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if model_type == "catboost":

        try:

            names = model.feature_names_

            if names:
                return list(names)

        except Exception:
            pass

        return None

    return None


# ============================================================
# ALIGN FEATURES WITH MODEL
# ============================================================

def align_features(X_test, model, model_type):

    X_test = X_test.copy()

    model_feature_names = (
        get_model_feature_names(
            model,
            model_type
        )
    )

    # --------------------------------------------------------
    # If model has explicit feature names
    # --------------------------------------------------------

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

        # Exact model ordering
        X_test = X_test[
            model_feature_names
        ]

        return X_test

    # --------------------------------------------------------
    # No feature names available
    # --------------------------------------------------------

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


# ============================================================
# SHAP CALCULATION
# ============================================================

@st.cache_data
def calculate_shap(
    target_column,
    max_samples=1500,
):

    # ========================================================
    # LOAD MODEL
    # ========================================================

    if target_column == "target_24h":

        model = load_model(
            "xgboost_24h.pkl",
            "xgboost",
        )

        model_type = "xgboost"

    elif target_column == "target_48h":

        model = load_model(
            "catboost_48h.cbm",
            "catboost",
        )

        model_type = "catboost"

    elif target_column == "target_72h":

        model = load_model(
            "rf_model_72h.pkl",
            "random_forest",
        )

        model_type = "random_forest"

    else:

        raise ValueError(
            f"Unknown target column: "
            f"{target_column}"
        )


    # ========================================================
    # PREPARE DATA
    # ========================================================

    X_test, test_cities = (
        prepare_shap_data(
            target_column
        )
    )


    # ========================================================
    # ALIGN FEATURES
    # ========================================================

    X_test = align_features(
        X_test,
        model,
        model_type,
    )


    # ========================================================
    # SAMPLE DATA
    # ========================================================
    #
    # SHAP over 17,000+ observations is unnecessarily slow.
    #
    # We only need a representative sample to determine
    # feature importance.
    #
    # This makes Streamlit dramatically faster.
    # ========================================================

    if len(X_test) > max_samples:

        sample_indices = (
            X_test.sample(
                n=max_samples,
                random_state=42,
            ).index
        )

        X_test = (
            X_test.loc[
                sample_indices
            ]
            .sort_index()
            .reset_index(drop=True)
        )

        test_cities = (
            test_cities.loc[
                sample_indices
            ]
            .reset_index(drop=True)
        )

    else:

        X_test = (
            X_test
            .reset_index(drop=True)
        )

        test_cities = (
            test_cities
            .reset_index(drop=True)
        )


    X_test = X_test.astype(float)


    # ========================================================
    # DEBUG
    # ========================================================

    st.write(
        f"SHAP model: {target_column}"
    )

    st.write(
        f"SHAP dataset: "
        f"{X_test.shape[0]:,} observations × "
        f"{X_test.shape[1]} features"
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    if model_type == "xgboost":

        try:

            explainer = shap.TreeExplainer(
                model
            )

            shap_values = (
                explainer.shap_values(
                    X_test
                )
            )

        except Exception as e:

            raise RuntimeError(
                "XGBoost SHAP calculation "
                "failed. This is usually caused "
                "by an incompatibility between "
                "the installed SHAP and XGBoost "
                "versions, or by a model trained "
                "with a different feature layout.\n\n"
                f"Original error: {e}"
            )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    elif model_type == "random_forest":

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = (
            explainer.shap_values(
                X_test
            )
        )


    # ========================================================
    # CATBOOST
    # ========================================================

    elif model_type == "catboost":

        feature_names = (
            list(X_test.columns)
        )

        # ----------------------------------------------------
        # Important:
        #
        # Do NOT call model.set_feature_names()
        # because the loaded model already has its own
        # feature structure.
        #
        # Instead, supply names to the Pool.
        # ----------------------------------------------------

        test_pool = Pool(
            X_test,
            feature_names=feature_names,
        )

        shap_result = (
            model.get_feature_importance(
                data=test_pool,
                type="ShapValues",
            )
        )

        # Last column = expected/base value
        shap_values = (
            shap_result[:, :-1]
        )


    # ========================================================
    # NORMALIZE SHAP OUTPUT
    # ========================================================

    # Some SHAP versions return a list
    # for certain model types.

    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[0]
        )

    shap_values = (
        pd.DataFrame(
            shap_values
        ).to_numpy()
    )


    # ========================================================
    # FINAL SHAPE CHECK
    # ========================================================

    if (
        shap_values.shape[0]
        != X_test.shape[0]
    ):

        raise ValueError(
            "SHAP output row count does "
            "not match X_test.\n"
            f"X_test: {X_test.shape}\n"
            f"SHAP: {shap_values.shape}"
        )

    if (
        shap_values.shape[1]
        != X_test.shape[1]
    ):

        raise ValueError(
            "SHAP output feature count "
            "does not match X_test.\n"
            f"X_test: {X_test.shape}\n"
            f"SHAP: {shap_values.shape}"
        )


    return (
        X_test,
        test_cities,
        shap_values,
    )


# ============================================================
# DISPLAY SHAP ANALYSIS
# ============================================================

def show_shap_analysis(city):

    st.divider()

    st.subheader(
        f"Model Explainability — {city}"
    )


    # ========================================================
    # HORIZON
    # ========================================================

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


    # ========================================================
    # CALCULATE SHAP
    # ========================================================

    try:

        (
            X_test,
            test_cities,
            shap_values,
        ) = calculate_shap(
            target_column
        )

    except Exception as e:

        st.error(
            f"Could not calculate SHAP "
            f"analysis:\n\n{e}"
        )

        return


    # ========================================================
    # CITY FILTER
    # ========================================================

    test_cities = (
        pd.Series(
            test_cities
        )
        .reset_index(drop=True)
    )

    city_mask = (
        test_cities == city
    ).values


    X_city = (
        X_test
        .loc[city_mask]
        .copy()
    )

    shap_city = (
        shap_values[
            city_mask
        ]
    )


    # ========================================================
    # NO DATA
    # ========================================================

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


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    shap_importance = pd.DataFrame(
        {
            "Feature": X_city.columns,
            "Importance": (
                abs(shap_city)
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


    # ========================================================
    # TABLE
    # ========================================================

    st.markdown(
        "### Feature Importance"
    )

    st.dataframe(
        shap_importance.head(10),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # BAR CHART
    # ========================================================

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


    # ========================================================
    # SHAP SUMMARY
    # ========================================================

    st.markdown(
        "### Feature Influence"
    )

    fig = plt.figure()

    shap.summary_plot(
        shap_city,
        X_city,
        show=False,
    )

    fig = plt.gcf()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)


    # ========================================================
    # INDIVIDUAL FEATURE EFFECT
    # ========================================================

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
        shap_city,
        X_city,
        show=False,
    )

    fig = plt.gcf()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)