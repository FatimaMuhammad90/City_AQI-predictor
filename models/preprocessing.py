import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(filepath=None, target_column="target_24h"):
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(
            current_dir, "..", "data", "combined_air_weather_5_cities_features.csv")

    df = pd.read_csv(filepath)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    df["target_24h"] = df.groupby("city")["us_aqi"].shift(-24)
    df["target_48h"] = df.groupby("city")["us_aqi"].shift(-48)
    df["target_72h"] = df.groupby("city")["us_aqi"].shift(-72)

    valid_targets = ["target_24h", "target_48h", "target_72h"]

    if target_column not in valid_targets:
        raise ValueError(f"target_column must be one of {valid_targets}")
    df["day"] = df["timestamp_utc"].dt.day
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
    df["month"] = df["timestamp_utc"].dt.month

    encoder = LabelEncoder()
    df["city_encoded"] = encoder.fit_transform(df["city"])

    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    unique_times = df['timestamp_utc'].unique()
    split_idx = int(len(unique_times) * 0.8)
    split_time = unique_times[split_idx]

    train = df[df["timestamp_utc"] < split_time]
    test = df[df["timestamp_utc"] >= split_time]

    columns_to_remove = [
        "city",
        "us_aqi",
        "target_24h",
        "target_48h",
        "target_72h",
        "timestamp_utc",
    ]

    X_train = train.drop(columns=columns_to_remove)
    X_test = test.drop(columns=columns_to_remove)

    y_train = train[target_column]
    y_test = test[target_column]

    test_cities = test["city"].values
    test_origins = test["timestamp_utc"].values

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        train,
        test,
        df,
        test_cities,
        test_origins
    )
