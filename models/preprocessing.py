import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(filepath=None):

    # Automatically locate the dataset if no filepath is given
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(
            current_dir,
            "..",
            "data",
            "combined_air_weather_5_cities.csv"
        )

    # Load Data
    df = pd.read_csv(filepath)

    # Convert timestamp
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    # Sort chronologically
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Time Features
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
    df["month"] = df["timestamp_utc"].dt.month

    # Encode City
    encoder = LabelEncoder()
    df["city"] = encoder.fit_transform(df["city"])

    # Train/Test Split
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Features & Target
    X_train = train.drop(columns=["us_aqi", "timestamp_utc"])
    y_train = train["us_aqi"]

    X_test = test.drop(columns=["us_aqi", "timestamp_utc"])
    y_test = test["us_aqi"]

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test, encoder