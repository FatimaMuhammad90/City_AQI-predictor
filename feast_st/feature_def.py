from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64


city = Entity(name="city", join_keys=["city"], description="AQI monitoring city")
aqi_source = FileSource(name="aqi_features_source", path="data/aqi_features.parquet", timestamp_field="timestamp_utc")

aqi_features = FeatureView(
    name="aqi_feature",
    entities=[city],
    ttl=timedelta(days=7),

    schema=[
        Field(name="latitude", dtype=Float32),
        Field(name="longitude", dtype=Float32),
        Field(name="pm2_5", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="city_encoded", dtype=Int64),
        Field(name="co", dtype=Float32),
        Field(name="no2", dtype=Float32),
        Field(name="so2", dtype=Float32),
        Field(name="o3", dtype=Float32),

        Field(name="temperature", dtype=Float32),
        Field(name="humidity", dtype=Float32),
        Field(name="pressure", dtype=Float32),
        Field(name="wind_speed", dtype=Float32),
        Field(name="wind_direction", dtype=Float32),
        Field(name="precipitation", dtype=Float32),
        Field(name="cloud_cover", dtype=Float32),

        # Lag features
        Field(name="aqi_lag_1", dtype=Float32),
        Field(name="aqi_lag_3", dtype=Float32),
        Field(name="aqi_lag_6", dtype=Float32),
        Field(name="aqi_lag_12", dtype=Float32),
        Field(name="aqi_lag_24", dtype=Float32),

        # Rolling features
        Field(name="aqi_rolling_mean_3h", dtype=Float32),
        Field(name="aqi_rolling_mean_24h", dtype=Float32),
        Field(name="aqi_rolling_min_24h", dtype=Float32),
        Field(name="aqi_rolling_max_24h", dtype=Float32),

        # Time features
        Field(name="hour", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64),

        Field(name="hour_sin", dtype=Float32),
        Field(name="hour_cos", dtype=Float32),

        Field(name="is_weekend", dtype=Int64),
        Field(name="is_rush_hour", dtype=Int64),

        # Difference features, if you have them
        Field(name="aqi_diff_1h", dtype=Float32),
        Field(name="aqi_diff_24h", dtype=Float32),
    ],

    online=True,

    source=aqi_source,

    tags={
        "project": "aqi_prediction",
        "purpose": "aqi_forecasting"
    }
)