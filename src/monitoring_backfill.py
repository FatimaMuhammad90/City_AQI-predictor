import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api_fetch import fetch_all_cities
from src.feast_update import update_feast
from src.feature_engineering import create_aqi_features
from src.inference import predict
from src.prediction_store import store_predictions


def prepare_features(air_df, weather_df):
    air_df = air_df.copy()
    weather_df = weather_df.copy()

    air_df["timestamp_utc"] = pd.to_datetime(
        air_df["timestamp_utc"], utc=True
    )
    weather_df["timestamp_utc"] = pd.to_datetime(
        weather_df["timestamp_utc"], utc=True
    )

    # Feature engineering requires the complete recent AQI history.
    air_df = create_aqi_features(air_df)

    city_encoding = {
        "Islamabad": 0,
        "Lahore": 1,
        "Peshawar": 2,
        "Rawalpindi": 3,
    }

    air_df["city_encoded"] = air_df["city"].map(city_encoding)

    latest_rows = (
        air_df.sort_values("timestamp_utc").groupby("city").tail(1)
    )

    df = pd.merge(
        latest_rows,
        weather_df,
        on=["city", "latitude", "longitude", "timestamp_utc"],
        how="inner",
    )

    print(f"After merge: {df.shape}")
    return df


def run_city_prediction(city, features):
    city_data = features[features["city"] == city].copy()

    if city_data.empty:
        raise ValueError(f"No feature data available for {city}")

    city_data = city_data.sort_values("timestamp_utc")

    prediction_time = city_data.iloc[-1]["timestamp_utc"]
    feature_row = city_data.iloc[[-1]]

    predictions = predict(feature_row)

    print(f"\n{city}")
    print(f"Prediction time: {prediction_time}")
    print(f"24h XGBoost      : {predictions['prediction_24h']:.2f}")
    print(f"48h CatBoost     : {predictions['prediction_48h']:.2f}")
    print(f"72h Random Forest: {predictions['prediction_72h']:.2f}")

    store_predictions(
        city=city,
        prediction_time=prediction_time,
        predictions=predictions,
    )

    return {
        "city": city,
        "prediction_time": prediction_time.isoformat(),
        "prediction_24h": predictions["prediction_24h"],
        "prediction_48h": predictions["prediction_48h"],
        "prediction_72h": predictions["prediction_72h"],
    }


def main():
    cities = ["Islamabad", "Lahore", "Peshawar", "Rawalpindi"]

    print("\n========================================")
    print("AQI PREDICTION PIPELINE")
    print("========================================")

    results = []

    try:
        print("\n[1/4] Fetching data for all cities...")
        air_df, weather_df = fetch_all_cities()
        print(f"Combined AQI data: {air_df.shape}")
        print(f"Combined weather data: {weather_df.shape}")

        print("\n[2/4] Creating features...")
        features = prepare_features(air_df, weather_df)
        print(f"Feature dataset: {features.shape}")

        print("\n[3/4] Updating Feast...")
        try:
            update_feast(features)
            print("Feast online store updated.")
        except Exception as e:
            print(f"Feast update failed: {e}")
            print("Continuing to prediction...")

        # 4. PREDICTIONS
        print("\n[4/4] Running predictions...")
        for city in cities:
            try:
                result = run_city_prediction(city, features)
                results.append(result)
            except Exception as e:
                print(f"\nERROR predicting {city}: {e}")
                continue

    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")

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


if __name__ == "__main__":
    main()