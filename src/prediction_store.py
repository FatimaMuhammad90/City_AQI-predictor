import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_KEY must be set"
    )

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


def store_predictions(city, prediction_time, predictions):

    prediction_time = pd.Timestamp(
        prediction_time
    ).isoformat()

    rows = [
        {
            "prediction_time": prediction_time,
            "target_time": (
                pd.Timestamp(prediction_time)
                + pd.Timedelta(hours=24)
            ).isoformat(),
            "city": city,
            "horizon": 24,
            "predicted_aqi": float(
                predictions["prediction_24h"]
            ),
            "model": "xgboost"
        },
        {
            "prediction_time": prediction_time,
            "target_time": (
                pd.Timestamp(prediction_time)
                + pd.Timedelta(hours=48)
            ).isoformat(),
            "city": city,
            "horizon": 48,
            "predicted_aqi": float(
                predictions["prediction_48h"]
            ),
            "model": "catboost"
        },
        {
            "prediction_time": prediction_time,
            "target_time": (
                pd.Timestamp(prediction_time)
                + pd.Timedelta(hours=72)
            ).isoformat(),
            "city": city,
            "horizon": 72,
            "predicted_aqi": float(
                predictions["prediction_72h"]
            ),
            "model": "random_forest"
        }
    ]

    response = (
        supabase
        .table("predictions")
        .insert(rows)
        .execute()
    )

    return response.data


def get_latest_predictions(city):

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("city", city)
        .order("prediction_time", desc=True)
        .limit(3)
        .execute()
    )

    rows = response.data

    if not rows:
        return None

    df = pd.DataFrame(rows)

    latest_time = df["prediction_time"].max()

    latest = df[
        df["prediction_time"] == latest_time
    ]

    return {
        "city": city,
        "prediction_time": latest_time,
        "prediction_24h": float(
            latest.loc[
                latest["horizon"] == 24,
                "predicted_aqi"
            ].iloc[0]
        ),
        "prediction_48h": float(
            latest.loc[
                latest["horizon"] == 48,
                "predicted_aqi"
            ].iloc[0]
        ),
        "prediction_72h": float(
            latest.loc[
                latest["horizon"] == 72,
                "predicted_aqi"
            ].iloc[0]
        )
    }