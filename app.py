import streamlit as st
from src.prediction_store import get_latest_predictions


st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌫️",
    layout="centered"
)


CITIES = [
    "Islamabad",
    "Lahore",
    "Peshawar",
    "Rawalpindi"
]


st.title("🌫️ AQI Prediction System")

st.write(
    "Air quality forecast for the next 24, 48 and 72 hours."
)


city = st.selectbox(
    "Select a city",
    CITIES
)


if st.button("Get Prediction"):

    try:

        predictions = get_latest_predictions(city)

        if predictions is None:
            st.warning(
                f"No predictions available for {city}."
            )

        else:

            st.subheader(
                f"{city} AQI Forecast"
            )

            st.caption(
                f"Prediction generated: "
                f"{predictions['prediction_time']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Next 24 Hours",
                    f"{predictions['prediction_24h']:.2f}"
                )

            with col2:
                st.metric(
                    "Next 48 Hours",
                    f"{predictions['prediction_48h']:.2f}"
                )

            with col3:
                st.metric(
                    "Next 72 Hours",
                    f"{predictions['prediction_72h']:.2f}"
                )

    except Exception as e:

        st.error(
            f"Could not retrieve predictions: {e}"
        )