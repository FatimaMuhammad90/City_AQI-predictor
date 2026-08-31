import pandas as pd
import streamlit as st

from streamlit_pred import (
    get_latest_predictions,
    get_monitoring_data,
    get_predictions_with_actuals,
    get_hourly_backfill_status,
)

from streamlit_shap import (
    show_shap_analysis,
)

st.set_page_config(
    page_title="AQI Prediction System",
    layout="wide",
)

CITIES = [
    "Islamabad",
    "Lahore",
    "Peshawar",
    "Rawalpindi",
]


# SESSION STATE

if "predictions" not in st.session_state:
    st.session_state["predictions"] = None


if "prediction_city" not in st.session_state:
    st.session_state["prediction_city"] = None


# AQI STATUS

def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#2ecc71"
    elif aqi <= 100:
        return "Moderate", "#f1c40f"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22"
    elif aqi <= 200:
        return "Unhealthy", "#e74c3c"
    elif aqi <= 300:
        return "Very Unhealthy", "#8e44ad"
    else:
        return "Hazardous", "#800000"


st.title("AQI Prediction System")

st.write("Air quality forecast for the next 24, 48 and 72 hours.")

st.divider()


# CITY SELECTION

city = st.selectbox(
    "Select city",
    CITIES,
    key="selected_city",
)


if st.button("Get Prediction", type="primary"):
    try:
        predictions = get_latest_predictions(city)

        if predictions is None:
            st.warning(f"No predictions available for {city}.")
        else:
            st.session_state["predictions"] = predictions
            st.session_state["prediction_city"] = city

    except Exception as e:
        st.error(f"Could not retrieve predictions: {e}")

if st.session_state["predictions"] is not None:
    predictions = st.session_state["predictions"]
    prediction_city = st.session_state["prediction_city"]


    prediction_time = pd.to_datetime(
        predictions["prediction_time"],
        utc=True,
    ).tz_convert("Asia/Karachi")
    aqi_24 = predictions["prediction_24h"]
    aqi_48 = predictions["prediction_48h"]
    aqi_72 = predictions["prediction_72h"]

    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"{prediction_city} AQI Forecast")

    with right:
        st.caption("Prediction generated")
        st.write(prediction_time.strftime("%d %b %Y"))
        st.write(prediction_time.strftime("%I:%M %p PKT"))

    st.divider()

    # AQI CARDS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Next 24 Hours",
            f"{aqi_24:.2f}",
            get_aqi_status(aqi_24)[0]
        )

    with col2:
        st.metric(
            "Next 48 Hours",
            f"{aqi_48:.2f}",
            get_aqi_status(aqi_48)[0]
        )

    with col3:
        st.metric(
            "Next 72 Hours",
            f"{aqi_72:.2f}",
            get_aqi_status(aqi_72)[0]
        )

    st.subheader("Forecast Trend")

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
        chart_data.set_index("Forecast Horizon"),
        use_container_width=True,
    )

    # SHAP

    show_shap_analysis(prediction_city)


# MODEL MONITORING

st.divider()

st.header("Model Performance Monitoring")

monitoring_data = get_monitoring_data()

if not monitoring_data:
    st.info(
        "No monitoring data available yet. "
        "The system will start recording after Monday."
    )
else:
    df = pd.DataFrame(monitoring_data)

    st.subheader("Flagged Models (Need Retraining)")

    flagged = df[df["flagged"] == True]

    if flagged.empty:
        st.success("No models flagged for retraining!")
    else:
        for _, row in flagged.iterrows():
            with st.expander(
                f"ALERT: {row['city']} - {row['horizon']}h "
                f"({row['model']}) - {row['check_date']}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "MAE",
                        f"{row['mae']:.2f}",
                        delta=f"{row['mae'] - row['baseline_mae']:.2f}",
                    )

                with col2:
                    st.metric(
                        "Consecutive Bad Days",
                        row["consecutive_bad_days"],
                    )

                with col3:
                    st.metric(
                        "Bad Days/Week",
                        f"{row['bad_predictions']}/{row['total_predictions']}",
                    )

    # MONITORING SUMMARY

    st.subheader("Monitoring Summary")

    latest = (
        df
        .sort_values("check_date")
        .groupby(["city", "horizon"])
        .last()
        .reset_index()
    )

    for _, row in latest.iterrows():
        with st.expander(
            f"{row['city']} - {row['horizon']}h ({row['model']})"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE",
                    f"{row['mae']:.2f}",
                    delta=f"{row['mae'] - row['baseline_mae']:.2f}",
                )

            with col2:
                st.metric(
                    "Consecutive Bad Days",
                    row["consecutive_bad_days"],
                )

            with col3:
                status = "Normal" if not row["flagged"] else "Flagged"
                st.metric("Status", status)


            # HISTORICAL PREDICTIONS


            historical_predictions = get_predictions_with_actuals(
                row["city"],
                row["horizon"],
            )

            if historical_predictions:
                pred_df = pd.DataFrame(historical_predictions)
                pred_df = pred_df.sort_values("target_time")
                pred_df["lagged_aqi"] = pred_df["actual_aqi"].shift(1)

                chart_df = pred_df.set_index("target_time")[
                    [
                        "predicted_aqi",
                        "actual_aqi",
                        "lagged_aqi",
                    ]
                ]

                st.line_chart(
                    chart_df,
                    height=300,
                )

                st.caption(
                    "Predicted AQI vs Actual AQI vs Previous-hour AQI"
                )


st.divider()

st.subheader("Hourly Backfill Status")

total, evaluated, pending = get_hourly_backfill_status()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Predictions", total)
with col2:
    st.metric("Evaluated", evaluated)
with col3:
    st.metric("Pending", pending)