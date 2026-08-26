import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_fetch import fetch_city_data
from feature_engineering import create_aqi_features
from inference import predict
from prediction_store import store_predictions
from feast_update import update_feast


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OBSERVED_AQI_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "observed_aqi_history.csv"
)


def update_observed_history(air_df):

    observed = air_df.copy()

    now = pd.Timestamp.now(tz="UTC")

    observed = observed[
        observed["timestamp_utc"] <= now
    ]

    observed = observed[
        [
            "timestamp_utc",
            "city",
            "latitude",
            "longitude",
            "pm2_5",
            "pm10",
            "co",
            "no2",
            "so2",
            "o3",
            "us_aqi"
        ]
    ]

    if os.path.exists(OBSERVED_AQI_FILE):

        old = pd.read_csv(OBSERVED_AQI_FILE)

        old["timestamp_utc"] = pd.to_datetime(
            old["timestamp_utc"],
            utc=True
        )

        observed = pd.concat(
            [old, observed],
            ignore_index=True
        )

    observed = observed.drop_duplicates(
        subset=["city", "timestamp_utc"],
        keep="last"
    )

    observed = observed.sort_values(
        ["city", "timestamp_utc"]
    ).reset_index(drop=True)

    observed.to_csv(
        OBSERVED_AQI_FILE,
        index=False
    )

    print(
        f"Observed AQI history updated: {observed.shape}"
    )

    return observed


def prepare_features(air_df, weather_df):

    air_df = air_df.copy()
    weather_df = weather_df.copy()

    air_df["timestamp_utc"] = pd.to_datetime(
        air_df["timestamp_utc"],
        utc=True
    )

    weather_df["timestamp_utc"] = pd.to_datetime(
        weather_df["timestamp_utc"],
        utc=True
    )

    # Create lag, rolling, time and difference features
    air_df = create_aqi_features(air_df)

    city_encoding = {
        "Islamabad": 0,
        "Lahore": 1,
        "Peshawar": 2,
        "Rawalpindi": 3
    }

    air_df["city_encoded"] = air_df["city"].map(
        city_encoding
    )

    # Only the latest row is needed for prediction
    latest_rows = (
        air_df
        .sort_values("timestamp_utc")
        .groupby("city")
        .tail(1)
    )

    df = pd.merge(
        latest_rows,
        weather_df,
        on=[
            "city",
            "latitude",
            "longitude",
            "timestamp_utc"
        ],
        how="inner"
    )

    print("After merge:", df.shape)

    return df


def run_city_prediction(city, features):

    city_data = features[
        features["city"] == city
    ].copy()

    if city_data.empty:
        raise ValueError(
            f"No feature data available for {city}"
        )

    city_data = city_data.sort_values(
        "timestamp_utc"
    )

    observed = pd.read_csv(
        OBSERVED_AQI_FILE
    )

    observed["timestamp_utc"] = pd.to_datetime(
        observed["timestamp_utc"],
        utc=True
    )

    city_observed = observed[
        observed["city"] == city
    ].sort_values("timestamp_utc")

    if city_observed.empty:
        raise ValueError(
            f"No observed AQI data available for {city}"
        )

    prediction_time = city_observed.iloc[-1][
        "timestamp_utc"
    ]

    latest = city_data[
        city_data["timestamp_utc"] == prediction_time
    ]

    if latest.empty:
        raise ValueError(
            f"No feature data available for prediction time "
            f"{prediction_time}"
        )

    feature_row = latest.iloc[[0]]

    predictions = predict(feature_row)

    print("\n========================================")
    print(f"{city}")
    print(f"Prediction time: {prediction_time}")
    print("========================================")

    print(
        f"24h XGBoost      : "
        f"{predictions['prediction_24h']:.2f}"
    )

    print(
        f"48h CatBoost     : "
        f"{predictions['prediction_48h']:.2f}"
    )

    print(
        f"72h Random Forest: "
        f"{predictions['prediction_72h']:.2f}"
    )

    store_predictions(
        city=city,
        prediction_time=prediction_time,
        predictions=predictions
    )

    return {
        "city": city,
        "prediction_time": prediction_time.isoformat(),
        "prediction_24h": predictions["prediction_24h"],
        "prediction_48h": predictions["prediction_48h"],
        "prediction_72h": predictions["prediction_72h"]
    }


def main():

    cities = [
        "Islamabad",
        "Lahore",
        "Peshawar",
        "Rawalpindi"
    ]

    print("\n========================================")
    print("AQI PREDICTION PIPELINE")
    print("Running for all cities")
    print("========================================")

    results = []

    for city in cities:

        try:
            print("\n========================================")
            print(f"PROCESSING: {city}")
            print("========================================")

            print("\n[1/4] Fetching data...")

            air_df, weather_df = fetch_city_data(city)

            print(f"AQI data: {air_df.shape}")
            print(f"Weather data: {weather_df.shape}")

            print("\n[2/4] Updating observed AQI history...")

            observed_history = update_observed_history(
                air_df
            )

            print("\n[3/4] Creating features...")

            features = prepare_features(
                observed_history,
                weather_df
            )

            update_feast(features)

            print(
                f"Feature dataset: {features.shape}"
            )

            print("\n[4/4] Running prediction...")

            result = run_city_prediction(
                city,
                features
            )

            results.append(result)

        except Exception as e:
            print(f"\nERROR processing {city}: {e}")

    return results


if __name__ == "__main__":
    main()
