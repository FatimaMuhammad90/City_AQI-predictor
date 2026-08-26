from pathlib import Path

from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from src.prediction_store import get_latest_predictions

app = FastAPI(
    title="AQI Prediction API",
    description="AQI forecasting API using Feast and Hugging Face models",
    version="1.0.0"
)


CITIES = [
    "Islamabad",
    "Lahore",
    "Peshawar",
    "Rawalpindi"
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEAST_REPO = PROJECT_ROOT / "feast_st"

store = FeatureStore(
    repo_path=str(FEAST_REPO)
)


FEATURES = [
    "aqi_feature:latitude",
    "aqi_feature:longitude",
    "aqi_feature:pm2_5",
    "aqi_feature:pm10",
    "aqi_feature:co",
    "aqi_feature:no2",
    "aqi_feature:so2",
    "aqi_feature:o3",
    "aqi_feature:temperature",
    "aqi_feature:humidity",
    "aqi_feature:pressure",
    "aqi_feature:wind_speed",
    "aqi_feature:wind_direction",
    "aqi_feature:precipitation",
    "aqi_feature:cloud_cover",
    "aqi_feature:aqi_lag_1",
    "aqi_feature:aqi_lag_3",
    "aqi_feature:aqi_lag_6",
    "aqi_feature:aqi_lag_12",
    "aqi_feature:aqi_lag_24",
    "aqi_feature:aqi_rolling_mean_3h",
    "aqi_feature:aqi_rolling_mean_24h",
    "aqi_feature:aqi_rolling_min_24h",
    "aqi_feature:aqi_rolling_max_24h",
    "aqi_feature:hour",
    "aqi_feature:day",
    "aqi_feature:day_of_week",
    "aqi_feature:month",
    "aqi_feature:hour_sin",
    "aqi_feature:hour_cos",
    "aqi_feature:is_weekend",
    "aqi_feature:is_rush_hour",
    "aqi_feature:aqi_diff_1h",
    "aqi_feature:aqi_diff_24h",
    "aqi_feature:city_encoded",
]


@app.get("/")
def health():
    return {
        "status": "OK"
    }


@app.get("/predict/{city}")
def predict(city: str):

    predictions = get_latest_predictions(city)

    if predictions is None:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {city}"
        )

    return predictions