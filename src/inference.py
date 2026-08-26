from src.model_registry import AQIModelRegistry

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "pm2_5",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_rolling_mean_3h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_min_24h",
    "aqi_rolling_max_24h",
    "hour",
    "day",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_rush_hour",
    "aqi_diff_1h",
    "aqi_diff_24h",
    "city_encoded"
]
# sarye cached models
registry = None


def get_registry():
    global registry
    if registry is None:
        print("Initializing model registry...")
        registry = AQIModelRegistry()
        registry.load_all_models()
        print("All models loaded from Hugging Face")
    return registry


def predict(features):

    X = features[FEATURE_COLUMNS]

    # Get registry (loads from Hugging Face)
    registry = get_registry()

    # Get models from registry and predict
    print("Running XGBoost (24h)...")
    model_24h = registry.get_model("24h")
    prediction_24h = model_24h.predict(X)[0]

    print("Running CatBoost (48h)...")
    model_48h = registry.get_model("48h")
    prediction_48h = model_48h.predict(X)[0]

    print("Running Random Forest (72h)...")
    model_72h = registry.get_model("72h")
    prediction_72h = model_72h.predict(X)[0]

    return {
        "prediction_24h": float(prediction_24h),
        "prediction_48h": float(prediction_48h),
        "prediction_72h": float(prediction_72h)
    }

