import pandas as pd
import numpy as np


def add_lag_features(df):
    df = df.sort_values(["city", "timestamp_utc"]).copy()

    df["aqi_lag_1"] = df.groupby("city")["us_aqi"].shift(1)
    df["aqi_lag_3"] = df.groupby("city")["us_aqi"].shift(3)
    df["aqi_lag_6"] = df.groupby("city")["us_aqi"].shift(6)
    df["aqi_lag_12"] = df.groupby("city")["us_aqi"].shift(12)
    df["aqi_lag_24"] = df.groupby("city")["us_aqi"].shift(24)

    return df

def add_rolling_features(df):
    df = df.copy()

    grouped_aqi = df.groupby("city")["us_aqi"]

    df["aqi_rolling_mean_3h"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    df["aqi_rolling_mean_24h"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=24).mean()
    )
    df["aqi_rolling_min_24h"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=24).min()
    )
    df["aqi_rolling_max_24h"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=24).max()
    )

    rolling_columns = [
        "aqi_rolling_mean_3h",
        "aqi_rolling_mean_24h",
        "aqi_rolling_min_24h",
        "aqi_rolling_max_24h",
    ]

    df = df.dropna(subset=rolling_columns).copy()

    return df

def add_time_features(df):
    df = df.copy()

    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
    df["month"] = df["timestamp_utc"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["is_rush_hour"] = (
        df["hour"].isin([7, 8, 9, 17, 18, 19])
    ).astype(int)

    return df


def add_rate_of_change_features(df):
    df = df.copy()

    df["aqi_diff_1h"] = (
        df["us_aqi"] -
        df.groupby("city")["us_aqi"].shift(1)
    )

    df["aqi_diff_24h"] = (
        df["us_aqi"] -
        df.groupby("city")["us_aqi"].shift(24)
    )

    df = df.dropna(
        subset=["aqi_diff_1h", "aqi_diff_24h"]
    ).copy()

    return df


def handle_lag_nulls(df):
    lag_columns = [
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_12",
        "aqi_lag_24",
    ]

    df = df.dropna(subset=lag_columns).copy()

    return df


def create_features(df):

    df = df.copy()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True
    )

    df = df.sort_values(
        ["city", "timestamp_utc"]
    ).reset_index(drop=True)

    df = add_lag_features(df)
    df = handle_lag_nulls(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_rate_of_change_features(df)

    return df