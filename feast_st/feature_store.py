import pandas as pd
df = pd.read_csv("../data/combined_air_weather_5_cities_features.csv")

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"],utc=True)
df.to_parquet("data/aqi_features.parquet",index=False)
print("Saved:", df.shape)
