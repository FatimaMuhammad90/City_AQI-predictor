import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from supabase import Client, create_client

from src.prediction_store import get_latest_predictions

load_dotenv()

# ============================================================
# PAGE CONFIG - MUST BE FIRST
# ============================================================
st.set_page_config(page_title="AQI Prediction System", layout="wide")

# ============================================================
# SUPABASE CONNECTION
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("""
    Missing Supabase credentials.
    
    Please add to your `.env` file:
    SUPABASE_URL=your_url
    SUPABASE_KEY=your_key
    """)
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# CONSTANTS
# ============================================================
CITIES = ["Islamabad", "Lahore", "Peshawar", "Rawalpindi"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
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


def get_monitoring_data():
    """Fetch latest monitoring data"""
    response = (
        supabase.table("monitoring")
        .select("*")
        .order("check_date", desc=True)
        .execute()
    )
    return response.data


def get_predictions_with_actuals(city, horizon):
    """Fetch predictions with actuals for a city/horizon"""
    response = (
        supabase.table("predictions")
        .select("*")
        .eq("city", city)
        .eq("horizon", horizon)
        .eq("evaluated", True)
        .order("target_time", desc=True)
        .limit(100)
        .execute()
    )
    return response.data


def get_hourly_backfill_status():
    """Get backfill status counts"""
    response = supabase.table("predictions").select("evaluated").execute()
    total = len(response.data)
    evaluated = sum(1 for p in response.data if p["evaluated"])
    pending = total - evaluated
    return total, evaluated, pending


# ============================================================
# MAIN APP - FORECAST SECTION
# ============================================================
st.title("AQI Prediction System")
st.write("Air quality forecast for the next 24, 48 and 72 hours.")
st.divider()

city = st.selectbox("Select city", CITIES)

if st.button("Get Prediction", type="primary"):
    try:
        predictions = get_latest_predictions(city)

        if predictions is None:
            st.warning(f"No predictions available for {city}.")
        else:
            prediction_time = pd.to_datetime(
                predictions["prediction_time"], utc=True
            ).tz_convert("Asia/Karachi")

            aqi_24 = predictions["prediction_24h"]
            aqi_48 = predictions["prediction_48h"]
            aqi_72 = predictions["prediction_72h"]

            left, right = st.columns([2, 1])

            with left:
                st.subheader(f"{city} AQI Forecast")

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

            col1, col2, col3 = st.columns(3)

            for col, horizon, aqi in [
                (col1, "Next 24 Hours", aqi_24),
                (col2, "Next 48 Hours", aqi_48),
                (col3, "Next 72 Hours", aqi_72),
            ]:
                status, color = get_aqi_status(aqi)

                with col:
                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            padding:20px;
                            border-radius:10px;
                            border:1px solid #ddd;
                        ">
                            <div style="font-size:18px; margin-bottom:8px;">
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

            st.subheader("Forecast Trend")

            chart_data = pd.DataFrame(
                {
                    "Forecast": ["24 Hours", "48 Hours", "72 Hours"],
                    "AQI": [aqi_24, aqi_48, aqi_72],
                }
            )

            st.line_chart(chart_data.set_index("Forecast"))

    except Exception as e:
        st.error(f"Could not retrieve predictions: {e}")

# ============================================================
# MODEL MONITORING SECTION
# ============================================================
st.divider()
st.header("Model Performance Monitoring")

# Get monitoring data
monitoring_data = get_monitoring_data()

if not monitoring_data:
    st.info(
        "No monitoring data available yet. The system will start recording"
        " after Monday."
    )
else:
    df = pd.DataFrame(monitoring_data)

    # Show flagged models
    st.subheader("Flagged Models (Need Retraining)")

    flagged = df[df["flagged"] == True]

    if flagged.empty:
        st.success("No models flagged for retraining!")
    else:
        for _, row in flagged.iterrows():
            with st.expander(
                f"ALERT: {row['city']} - {row['horizon']}h ({row['model']}) -"
                f" {row['check_date']}"
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
                        "Consecutive Bad Days", row["consecutive_bad_days"]
                    )
                with col3:
                    st.metric(
                        "Bad Days/Week",
                        f"{row['bad_predictions']}/{row['total_predictions']}",
                    )

    # All monitoring summary
    st.subheader("Monitoring Summary")

    # Latest week per city/model
    latest = (
        df.sort_values("check_date")
        .groupby(["city", "horizon"])
        .last()
        .reset_index()
    )

    for _, row in latest.iterrows():
        with st.expander(f"{row['city']} - {row['horizon']}h ({row['model']})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "MAE",
                    f"{row['mae']:.2f}",
                    delta=f"{row['mae'] - row['baseline_mae']:.2f}",
                )
            with col2:
                st.metric(
                    "Consecutive Bad Days", row["consecutive_bad_days"]
                )
            with col3:
                status = "Normal" if not row["flagged"] else "Flagged"
                st.metric("Status", status)

            # Show actual vs predicted graph for this city/horizon
            predictions = get_predictions_with_actuals(
                row["city"], row["horizon"]
            )
            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_df = pred_df.sort_values("target_time")

                # Add lagged prediction (previous hour's AQI)
                pred_df["lagged_aqi"] = pred_df["actual_aqi"].shift(1)

                st.line_chart(
                    pred_df.set_index("target_time")[
                        ["predicted_aqi", "actual_aqi", "lagged_aqi"]
                    ],
                    height=300,
                )
                st.caption(
                    "Blue: Predicted | Green: Actual | Red: Lagged (previous"
                    " hour)"
                )

# ============================================================
# HOURLY BACKFILL STATUS
# ============================================================
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

