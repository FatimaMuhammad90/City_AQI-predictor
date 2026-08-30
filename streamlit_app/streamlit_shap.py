import os
import sys
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]


# ============================================================
# IMPORTS
# ============================================================

from models.supabase_data import get_historical_data
from models.SHAP import calculate_shap_results


# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_historical_data_for_shap():

    return get_historical_data()


# ============================================================
# SHAP CACHE
# ============================================================

@st.cache_data(
    show_spinner=False
)
def get_cached_shap_results(target_column):

    historical_df = load_historical_data_for_shap()

    return calculate_shap_results(
        historical_df,
        target_column,
    )


# ============================================================
# SHAP UI
# ============================================================

def show_shap_analysis(city):

    st.divider()

    st.subheader(
        f"Model Explainability — {city}"
    )

    st.caption(
        "SHAP explains which features influenced the model's predictions."
    )

    # ========================================================
    # EXPLICITLY ENABLE SHAP
    # ========================================================

    show_explainability = st.checkbox(
        "Show Model Explainability",
        key=f"show_shap_{city}",
    )

    # IMPORTANT:
    # If unchecked, absolutely NO SHAP calculation happens.

    if not show_explainability:
        return

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

    target_column = f"target_{horizon}h"

    # ========================================================
    # CALCULATE / LOAD CACHED SHAP
    # ========================================================

    try:

        with st.spinner(
            f"Preparing SHAP analysis for the {horizon}-hour model..."
        ):

            (
                X_test,
                test_cities,
                shap_values,
                shap_importance,
            ) = get_cached_shap_results(
                target_column
            )

    except Exception as e:

        st.error(
            f"Could not calculate SHAP analysis: {e}"
        )

        return

    # ========================================================
    # CITY FILTER
    # ========================================================

    test_cities = (
        pd.Series(test_cities)
        .reset_index(drop=True)
    )

    city_mask = (
        test_cities == city
    ).values

    X_city = X_test.loc[
        city_mask
    ].copy()

    shap_city = shap_values[
        city_mask
    ]

    # ========================================================
    # NO CITY DATA
    # ========================================================

    if len(X_city) == 0:

        st.warning(
            f"No SHAP test data available for {city}."
        )

        return

    # ========================================================
    # CITY-SPECIFIC IMPORTANCE
    # ========================================================

    city_importance = pd.DataFrame(
        {
            "Feature": X_city.columns,
            "Importance": abs(
                shap_city
            ).mean(axis=0),
        }
    )

    city_importance = (
        city_importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # INFO
    # ========================================================

    st.write(
        f"SHAP analysis is based on "
        f"{len(X_city):,} test observations "
        f"for {city} using the "
        f"{horizon}-hour model."
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        "### Feature Importance"
    )

    st.dataframe(
        city_importance.head(10),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TOP 10 BAR CHART
    # ========================================================

    st.markdown(
        "### Top 10 Features"
    )

    importance_chart = (
        city_importance
        .head(10)
        .sort_values(
            "Importance"
        )
        .set_index("Feature")
    )

    st.bar_chart(
        importance_chart["Importance"],
        horizontal=True,
        use_container_width=True,
    )

    # ========================================================
    # SHAP SUMMARY
    # ========================================================

    st.markdown(
        "### Feature Influence"
    )

    fig = plt.figure(
        figsize=(10, 7)
    )

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
    # DEPENDENCE PLOT
    # ========================================================

    st.markdown(
        "### Individual Feature Effects"
    )

    top_features = (
        city_importance
        .head(3)["Feature"]
        .tolist()
    )

    selected_feature = st.selectbox(
        "Select a feature",
        top_features,
        key=(
            f"shap_feature_"
            f"{city}_{horizon}"
        ),
    )

    fig = plt.figure(
        figsize=(9, 6)
    )

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