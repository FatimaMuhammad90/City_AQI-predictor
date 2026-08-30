import pandas as pd
import streamlit as st

from supabase import (
    Client,
    create_client,
)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = st.secrets[
    "SUPABASE_URL"
]

SUPABASE_KEY = st.secrets[
    "SUPABASE_KEY"
]

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


# ============================================================
# LATEST PREDICTIONS
# ============================================================

@st.cache_data(ttl=300)
def get_latest_predictions(city):

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("city", city)
        .order(
            "prediction_time",
            desc=True,
        )
        .limit(1)
        .execute()
    )

    if not response.data:

        return None

    latest_time = (
        response.data[0][
            "prediction_time"
        ]
    )

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("city", city)
        .eq(
            "prediction_time",
            latest_time,
        )
        .execute()
    )

    rows = response.data

    if not rows:

        return None

    df = pd.DataFrame(rows)


    # --------------------------------------------------------
    # Make sure all three horizons exist
    # --------------------------------------------------------

    required_horizons = {
        24,
        48,
        72,
    }

    available_horizons = set(
        df["horizon"].astype(int)
    )

    missing = (
        required_horizons
        - available_horizons
    )

    if missing:

        raise ValueError(
            f"Missing prediction horizons "
            f"for {city}: {sorted(missing)}"
        )


    return {
        "city": city,

        "prediction_time":
            latest_time,

        "prediction_24h":
            float(
                df.loc[
                    df["horizon"] == 24,
                    "predicted_aqi",
                ].iloc[0]
            ),

        "prediction_48h":
            float(
                df.loc[
                    df["horizon"] == 48,
                    "predicted_aqi",
                ].iloc[0]
            ),

        "prediction_72h":
            float(
                df.loc[
                    df["horizon"] == 72,
                    "predicted_aqi",
                ].iloc[0]
            ),
    }


# ============================================================
# MONITORING DATA
# ============================================================

@st.cache_data(ttl=300)
def get_monitoring_data():

    response = (
        supabase
        .table("monitoring")
        .select("*")
        .order(
            "check_date",
            desc=True,
        )
        .execute()
    )

    return response.data


# ============================================================
# HISTORICAL PREDICTIONS
# ============================================================

@st.cache_data(ttl=300)
def get_predictions_with_actuals(
    city,
    horizon,
):

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq(
            "city",
            city,
        )
        .eq(
            "horizon",
            horizon,
        )
        .eq(
            "evaluated",
            True,
        )
        .order(
            "target_time",
            desc=True,
        )
        .limit(100)
        .execute()
    )

    return response.data


# ============================================================
# HOURLY BACKFILL STATUS
# ============================================================

@st.cache_data(ttl=300)
def get_hourly_backfill_status():

    response = (
        supabase
        .table("predictions")
        .select("evaluated")
        .execute()
    )

    total = len(
        response.data
    )

    evaluated = sum(
        1
        for prediction
        in response.data
        if prediction["evaluated"]
    )

    pending = (
        total - evaluated
    )

    return (
        total,
        evaluated,
        pending,
    )