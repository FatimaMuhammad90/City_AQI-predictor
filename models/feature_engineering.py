import pandas as pd
import numpy as np

def add_lag_features(df):
    df = df.sort_values(["city", "timestamp_utc"]).copy()
    df["aqi_lag_1"] = (df.groupby("city")["us_aqi"].shift(1))
    df["aqi_lag_3"] = (df.groupby("city")["us_aqi"].shift(3))
    df["aqi_lag_6"] = (df.groupby("city")["us_aqi"].shift(6))
    df["aqi_lag_12"] = (df.groupby("city")["us_aqi"].shift(12))
    df["aqi_lag_24"] = (df.groupby("city")["us_aqi"].shift(24))
    return df

def add_rolling_features(df):
    df = df.copy()
    grouped_aqi = df.groupby("city")["us_aqi"]
    # Short term trend
    df["aqi_rolling_mean_3h"] = ( grouped_aqi.shift(1).rolling(window=3).mean().reset_index(level=0, drop=True))
    # average AQI over the past day
    df["aqi_rolling_mean_24h"] = (grouped_aqi.shift(1).rolling(window=24).mean().reset_index(level=0, drop=True))
    df["aqi_rolling_min_24h"] = (grouped_aqi.shift(1).rolling(window=24).min().reset_index(level=0, drop=True))
    df["aqi_rolling_max_24h"] = (grouped_aqi.shift(1).rolling(window=24).max().reset_index(level=0, drop=True))
    rolling_columns = [
        "aqi_rolling_mean_3h",
        "aqi_rolling_mean_24h",
        "aqi_rolling_min_24h",
        "aqi_rolling_max_24h"
    ]

    df = df.dropna(subset=rolling_columns).copy()

    print("After removing rolling-feature NaNs:")
    print(df.shape)

    print("\nMissing values:")
    print(df[rolling_columns].isna().sum())

    return df


def add_time_feautures(df):
    df = df.copy()
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
    df["month"] = df["timestamp_utc"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Simple rush-hour indicator
    df["is_rush_hour"] = (df["hour"].isin([7, 8, 9, 17, 18, 19])).astype(int)

    return df
def add_rate_of_change_features(df):
    df["aqi_diff_1h"] = (df["us_aqi"] - df.groupby("city")["us_aqi"].shift(1))
    df["aqi_diff_24h"] = (df["us_aqi"] - df.groupby("city")["us_aqi"].shift(24))

    rate_columns = [
        "aqi_diff_1h",
        "aqi_diff_24h"
    ]

    df = df.dropna(subset=rate_columns).copy()

    print("After removing rolling-feature NaNs:")
    print(df.shape)

    print("\nMissing values:")
    print(df[rate_columns].isna().sum())
    return df

def handle_lag_nulls(df):
    lag_columns = [
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_12",
        "aqi_lag_24"
    ]
    print("Missing values before:")
    print(df[lag_columns].isna().sum())

    df = df.dropna(subset=lag_columns).copy()

    print("\nMissing values after:")
    print(df[lag_columns].isna().sum())

    print("\nNew shape:")
    print(df.shape)

    return df

df = pd.read_csv("../data/combined_air_weather_4_cities.csv")

# Convert timestamp
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

# Add lag features
df = add_lag_features(df)
df = handle_lag_nulls(df)
df = add_rolling_features(df)
df = add_time_feautures(df)
df = add_rate_of_change_features(df)

print(df[["city","timestamp_utc","us_aqi","aqi_lag_1","aqi_lag_3","aqi_lag_6","aqi_lag_12","aqi_lag_24"]].head(30))

df.to_csv("../data/combined_air_weather_5_cities_features.csv",index=False)

print("\nSaved feature-engineered dataset.")