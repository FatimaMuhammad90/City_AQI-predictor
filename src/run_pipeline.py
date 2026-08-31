import os
import sys
import pandas as pd

# This is the Hourly_Aqi_pipeline starting point
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api_fetch import fetch_all_cities
from src.feature_engineering import create_aqi_features
from src.inference import predict
from src.prediction_store import store_predictions
from src.feast_update import update_feast


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

    try:

        print("\n[1/4] Fetching data for all cities...")

        air_df, weather_df = fetch_all_cities()

        print(f"\nCombined AQI data: {air_df.shape}")
        print(f"Combined weather data: {weather_df.shape}")


        print("\n[2/4] Updating observed AQI history...")

        observed_history = update_observed_history(air_df)

        print("\n[3/4] Creating features...")

        features = prepare_features(observed_history,weather_df)

        print(f"Feature dataset: {features.shape}" )
        print("\nUpdating Feast...")
        update_feast(features)
        print("Feast online store updated.")

        print("\n[4/4] Running predictions...")

        results = []

        for city in cities:

            try:

                result = run_city_prediction(
                    city,
                    features
                )

                results.append(result)

            except Exception as e:

                print(
                    f"\nERROR predicting {city}: {e}"
                )



        print("\n========================================")
        print("PIPELINE COMPLETE")
        print("========================================")

        for result in results:

            print(
                f"{result['city']} | "
                f"24h: {result['prediction_24h']:.2f} | "
                f"48h: {result['prediction_48h']:.2f} | "
                f"72h: {result['prediction_72h']:.2f}"
            )

        return results

    except Exception as e:

        print(
            f"\nPIPELINE FAILED: {e}"
        )

        return []


if __name__ == "__main__":
    main()