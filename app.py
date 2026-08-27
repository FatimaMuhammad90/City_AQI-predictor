import streamlit as st
import pandas as pd

from src.prediction_store import get_latest_predictions


st.set_page_config(
    page_title="AQI Prediction System",
    page_icon=None,
    layout="wide"
)


CITIES = [
    "Islamabad",
    "Lahore",
    "Peshawar",
    "Rawalpindi"
]


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

st.write(
    "Air quality forecast for the next 24, 48 and 72 hours."
)

st.divider()


city = st.selectbox(
    "Select city",
    CITIES
)


if st.button("Get Prediction", type="primary"):

    try:

        predictions = get_latest_predictions(city)

        if predictions is None:

            st.warning(
                f"No predictions available for {city}."
            )

        else:

            prediction_time = pd.to_datetime(
                predictions["prediction_time"],
                utc=True
            ).tz_convert("Asia/Karachi")

            aqi_24 = predictions["prediction_24h"]
            aqi_48 = predictions["prediction_48h"]
            aqi_72 = predictions["prediction_72h"]

            left, right = st.columns([2, 1])

            with left:

                st.subheader(
                    f"{city} AQI Forecast"
                )

            with right:

                st.markdown(
                    "<div style='text-align:right'>"
                    "<div style='font-size:16px; color:#777;'>"
                    "Prediction generated"
                    "</div>"
                    "<div style='font-size:28px; "
                    "font-weight:700;'>"
                    f"{prediction_time.strftime('%d %b %Y')}"
                    "</div>"
                    "<div style='font-size:24px; "
                    "font-weight:600;'>"
                    f"{prediction_time.strftime('%I:%M %p PKT')}"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

            st.divider()

            col1, col2, col3 = st.columns(3)

            for col, horizon, aqi in [
                (col1, "Next 24 Hours", aqi_24),
                (col2, "Next 48 Hours", aqi_48),
                (col3, "Next 72 Hours", aqi_72)
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
                        unsafe_allow_html=True
                    )

            st.subheader("Forecast")

            chart_data = pd.DataFrame(
                {
                    "Forecast": [
                        "24 Hours",
                        "48 Hours",
                        "72 Hours"
                    ],
                    "AQI": [
                        aqi_24,
                        aqi_48,
                        aqi_72
                    ]
                }
            )

            st.line_chart(
                chart_data.set_index("Forecast")
            )

    except Exception as e:

        st.error(
            f"Could not retrieve predictions: {e}"
        )