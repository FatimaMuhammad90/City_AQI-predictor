# import pandas as pd
# df = pd.read_csv("../data/combined_air_weather_5_cities_features.csv")
#
# df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"],utc=True)
# df.to_parquet("data/aqi_features.parquet",index=False)
# print("Saved:", df.shape)
import pandas as pd
df = pd.read_parquet("data/aqi_features.parquet")
print(df.columns.tolist())
print("Shape:", df.shape)
print("timestamp_utc type:",df["timestamp_utc"].dtype)
print(df['timestamp_utc'].tail(10))
print(f"  Min: {df['timestamp_utc'].min()}")
print(f"  Max: {df['timestamp_utc'].max()}")