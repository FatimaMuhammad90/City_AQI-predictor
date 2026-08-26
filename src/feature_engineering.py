import numpy as np
import pandas as pd

def create_aqi_features(history):
    df = history.copy()

    df = df.sort_values(
        ["city", "timestamp_utc"]
    )

    # Time features
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
    df["month"] = df["timestamp_utc"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (df["hour"].isin([7, 8, 9, 16, 17, 18, 19])).astype(int)
    # AQI lags
    df["aqi_lag_1"] = (df.groupby("city")["us_aqi"].shift(1))
    df["aqi_lag_3"] = (df.groupby("city")["us_aqi"].shift(3))

    df["aqi_lag_6"] = (df.groupby("city")["us_aqi"].shift(6))
    df["aqi_lag_12"] = (
        df.groupby("city")["us_aqi"].shift(12)
    )

    df["aqi_lag_24"] = (
        df.groupby("city")["us_aqi"].shift(24)
    )

    # Rolling features
    grouped = df.groupby("city")["us_aqi"]

    df["aqi_rolling_mean_3h"] = (
        grouped.transform(
            lambda x: x.rolling(3).mean()
        )
    )

    df["aqi_rolling_mean_24h"] = (
        grouped.transform(
            lambda x: x.rolling(24).mean()
        )
    )

    df["aqi_rolling_min_24h"] = (
        grouped.transform(
            lambda x: x.rolling(24).min()
        )
    )

    df["aqi_rolling_max_24h"] = (
        grouped.transform(
            lambda x: x.rolling(24).max()
        )
    )

    # Differences
    df["aqi_diff_1h"] = (
        df["us_aqi"] -
        df["aqi_lag_1"]
    )

    df["aqi_diff_24h"] = (
        df["us_aqi"] -
        df["aqi_lag_24"]
    )

    return df