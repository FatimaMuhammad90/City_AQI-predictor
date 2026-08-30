import pandas as pd
import streamlit as st

from supabase import (
    Client,
    create_client,
)

from streamlit_pred import (
    get_latest_predictions,
    get_monitoring_data,
    get_predictions_with_actuals,
    get_hourly_backfill_status,
)

from streamlit_shap import show_shap_analysis


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AQI Prediction System",
    layout="wide",
)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


# ============================================================
# CONSTANTS
# ============================================================

CITIES = [
    "Islamabad",
    "Lahore",
    "Peshawar",
    "Rawalpindi",
]


# ============================================================
# SESSION STATE
# ============================================================

if "predictions" not in st.session_state:
    st.session_state["predictions"] = None

if "prediction_city" not in st.session_state:
    st.session_state["prediction_city"] = None


# ============================================================
# HELPER
# ============================================================

def get_aqi_status(aqi):

    if aqi <= 50:
        return "Good", "#2ecc71"

    elif aqi <= 100:
        return "Moderate", "#f1c40f"

    elif aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#e67e22",
        )

    elif aqi <= 200:
        return "Unhealthy", "#e74c3c"

    elif aqi <= 300:
        return "Very Unhealthy", "#8e44ad"

    else:
        return "Hazardous", "#800000"


# ============================================================
# TITLE
# ============================================================

st.title("AQI Prediction System")

st.write(
    "Air quality forecast for the next "
    "24, 48 and 72 hours."
)

st.divider()


# ============================================================
# CITY
# ============================================================

city = st.selectbox( "Select city",
    CITIES,
    key="selected_city",
)


# ============================================================
# GET PREDICTION
# ============================================================

if st.button(
    "Get Prediction",
    type="primary",
):

    try:

        predictions = (
            get_latest_predictions(city)
        )

        if predictions is None:

            st.warning(
                f"No predictions available "
                f"for {city}."
            )

        else:

            st.session_state[
                "predictions"
            ] = predictions

            st.session_state[
                "prediction_city"
            ] = city

    except Exception as e:

        st.error(
            f"Could not retrieve predictions: {e}"
        )


# ============================================================
# FORECAST DISPLAY
# ============================================================

if st.session_state[
    "predictions"
] is not None:

    predictions = st.session_state[
        "predictions"
    ]

    prediction_city = st.session_state[
        "prediction_city"
    ]

    # --------------------------------------------------------
    # PREDICTION TIME
    # --------------------------------------------------------

    prediction_time = (
        pd.to_datetime(
            predictions[
                "prediction_time"
            ],
            utc=True,
        )
        .tz_convert("Asia/Karachi")
    )

    # --------------------------------------------------------
    # AQI VALUES
    # --------------------------------------------------------

    aqi_24 = predictions[
        "prediction_24h"
    ]

    aqi_48 = predictions[
        "prediction_48h"
    ]

    aqi_72 = predictions[
        "prediction_72h"
    ]

    # ========================================================
    # HEADER
    # ========================================================

    left, right = st.columns(
        [2, 1]
    )

    with left:

        st.subheader(
            f"{prediction_city} AQI Forecast"
        )

    with right:

        st.markdown(
            f"""
            <div style='text-align:right'>
                <div style='font-size:16px; color:#777;'>
                    Prediction generated
                </div>

                <div style='font-size:28px; font-weight:700;'>
                    {prediction_time.strftime('%d %b %Y')}
                </div>

                <div style='font-size:24px; font-weight:600;'>
                    {prediction_time.strftime('%I:%M %p PKT')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ========================================================
    # AQI CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    forecasts = [
        (
            col1,
            "Next 24 Hours",
            aqi_24,
        ),
        (
            col2,
            "Next 48 Hours",
            aqi_48,
        ),
        (
            col3,
            "Next 72 Hours",
            aqi_72,
        ),
    ]

    for col, horizon, aqi in forecasts:

        status, color = get_aqi_status(
            aqi
        )

        with col:

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding:20px;
                    border-radius:10px;
                    border:1px solid #ddd;
                ">

                    <div style="
                        font-size:18px;
                        margin-bottom:8px;
                    ">
                        {horizon}
                    </div>

                    <div style="
                        font-size:42px;
                        font-weight:700;
                        margin-bottom:10px;
                    ">
                        {aqi:.2f}
                    </div>

                    <div style="
                        background-color:{color};
                        color:white;
                        padding:8px;
                        border-radius:6px;
                        font-weight:600;
                    ">
                        {status}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # FORECAST GRAPH
    # ========================================================

    st.subheader(
        "Forecast Trend"
    )

    chart_data = pd.DataFrame(
        {
            "Forecast Horizon": [
                "24 Hours",
                "48 Hours",
                "72 Hours",
            ],
            "Predicted AQI": [
                aqi_24,
                aqi_48,
                aqi_72,
            ],
        }
    )

    st.line_chart(
        chart_data.set_index(
            "Forecast Horizon"
        )
    )

    # ========================================================
    # SHAP
    # ========================================================

    show_shap_analysis(
        prediction_city
    )


# ============================================================
# MODEL MONITORING
# ============================================================

st.divider()

st.header(
    "Model Performance Monitoring"
)

monitoring_data = (
    get_monitoring_data()
)


if not monitoring_data:

    st.info(
        "No monitoring data available yet. "
        "The system will start recording after Monday."
    )

else:

    df = pd.DataFrame(
        monitoring_data
    )

    # ========================================================
    # FLAGGED MODELS
    # ========================================================

    st.subheader(
        "Flagged Models (Need Retraining)"
    )

    flagged = df[
        df["flagged"] == True
    ]

    if flagged.empty:

        st.success(
            "No models flagged for retraining!"
        )

    else:

        for _, row in flagged.iterrows():

            with st.expander(
                f"ALERT: "
                f"{row['city']} - "
                f"{row['horizon']}h "
                f"({row['model']}) - "
                f"{row['check_date']}"
            ):

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    st.metric(
                        "MAE",
                        f"{row['mae']:.2f}",
                        delta=(
                            f"{row['mae'] - row['baseline_mae']:.2f}"
                        ),
                    )

                with col2:

                    st.metric(
                        "Consecutive Bad Days",
                        row[
                            "consecutive_bad_days"
                        ],
                    )

                with col3:

                    st.metric(
                        "Bad Days/Week",
                        f"{row['bad_predictions']}/"
                        f"{row['total_predictions']}",
                    )

    # ========================================================
    # MONITORING SUMMARY
    # ========================================================

    st.subheader(
        "Monitoring Summary"
    )

    latest = (
        df
        .sort_values("check_date")
        .groupby(
            ["city", "horizon"]
        )
        .last()
        .reset_index()
    )

    for _, row in latest.iterrows():

        with st.expander(
            f"{row['city']} - "
            f"{row['horizon']}h "
            f"({row['model']})"
        ):

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "MAE",
                    f"{row['mae']:.2f}",
                    delta=(
                        f"{row['mae'] - row['baseline_mae']:.2f}"
                    ),
                )

            with col2:

                st.metric(
                    "Consecutive Bad Days",
                    row[
                        "consecutive_bad_days"
                    ],
                )

            with col3:

                status = (
                    "Normal"
                    if not row["flagged"]
                    else "Flagged"
                )

                st.metric(
                    "Status",
                    status,
                )

            # ------------------------------------------------
            # HISTORICAL PREDICTIONS
            # ------------------------------------------------

            historical_predictions = (
                get_predictions_with_actuals(
                    row["city"],
                    row["horizon"],
                )
            )

            if historical_predictions:

                pred_df = pd.DataFrame(
                    historical_predictions
                )

                pred_df = (
                    pred_df
                    .sort_values(
                        "target_time"
                    )
                )

                pred_df[
                    "lagged_aqi"
                ] = (
                    pred_df[
                        "actual_aqi"
                    ].shift(1)
                )

                chart_df = (
                    pred_df
                    .set_index(
                        "target_time"
                    )[
                        [
                            "predicted_aqi",
                            "actual_aqi",
                            "lagged_aqi",
                        ]
                    ]
                )

                st.line_chart(
                    chart_df,
                    height=300,
                )

                st.caption(
                    "Predicted AQI vs Actual AQI "
                    "vs Previous-hour AQI"
                )


# ============================================================
# HOURLY BACKFILL
# ============================================================

st.divider()

st.subheader(
    "Hourly Backfill Status"
)

total, evaluated, pending = (
    get_hourly_backfill_status()
)

col1, col2, col3 = (
    st.columns(3)
)

with col1:

    st.metric(
        "Total Predictions",
        total,
    )

with col2:

    st.metric(
        "Evaluated",
        evaluated,
    )

with col3:

    st.metric(
        "Pending",
        pending,
    )