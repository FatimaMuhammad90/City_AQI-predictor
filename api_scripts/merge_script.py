import pandas as pd

# Load both datasets
air_df = pd.read_csv("../data/historical_air_quality_4_cities.csv")
weather_df = pd.read_csv("../data/historical_weather_4_cities.csv")

# Merge on timestamp and city
combined_df = pd.merge(
    air_df,
    weather_df,
    on=["timestamp_utc", "city", "latitude", "longitude"],
    how="inner"  # Only keep rows that exist in both
)

# Save combined dataset
combined_df.to_csv("combined_air_weather_4_cities.csv", index=False)

print(f"Air data shape: {air_df.shape}")
print(f"Weather data shape: {weather_df.shape}")
print(f"Combined data shape: {combined_df.shape}")
print(combined_df.head())