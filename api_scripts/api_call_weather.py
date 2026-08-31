import requests
import pandas as pd
from datetime import date, timedelta
# for weather
CITIES = [
    {   "city": "Islamabad",
        "lat": 33.6844,
        "lon": 73.0479
    },
    {
        "city": "Rawalpindi",
        "lat": 33.5651,
        "lon": 73.0169
    },
    {
        "city": "Lahore",
        "lat": 31.5204,
        "lon": 74.3587
    },
    {
        "city": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011
    },
    {
        "city": "Peshawar",
        "lat": 34.0151,
        "lon": 71.5249
    }
]

end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=730)

all_weather = []

for city in CITIES:


    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover"
        ]),
        "timezone": "GMT"
    }

    response = requests.get(url, params=params)
    print(response.status_code)

    data = response.json()["hourly"]

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(data["time"], utc=True),
        "city": city["city"],
        "latitude": city["lat"],
        "longitude": city["lon"],
        "temperature": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "pressure": data["surface_pressure"],
        "wind_speed": data["wind_speed_10m"],
        "wind_direction": data["wind_direction_10m"],
        "precipitation": data["precipitation"],
        "cloud_cover": data["cloud_cover"]
    })

    all_weather.append(df)

weather_df = pd.concat(all_weather, ignore_index=True)

weather_df.to_csv(
    "historical_weather_4_cities.csv",
    index=False
)

print(weather_df.head())
print(weather_df.shape)