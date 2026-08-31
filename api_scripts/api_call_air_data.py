import requests
import pandas as pd
from datetime import date, timedelta
# these are the starting scripts to get the data and crete data file
CITIES = [
    {
        "city": "Islamabad",
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
start_date = end_date- timedelta(days=730)

all_air = []

for city in CITIES:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],

        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),

        "hourly": ",".join([
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
            "european_aqi"
        ]),

        "timezone": "UTC"

    }

    response = requests.get(url, params=params)
    print(response.status_code)

    data = response.json()["hourly"]

    df = pd.DataFrame({

        "timestamp_utc": pd.to_datetime(
            data["time"],
            utc=True
        ),

        "city": city["city"],
        "latitude": city["lat"],
        "longitude": city["lon"],
        "pm2_5": data["pm2_5"],
        "pm10": data["pm10"],
        "co": data["carbon_monoxide"],
        "no2": data["nitrogen_dioxide"],
        "so2": data["sulphur_dioxide"],
        "o3": data["ozone"],
        "us_aqi": data["us_aqi"]

    })

    all_air.append(df)

air_df = pd.concat(all_air, ignore_index=True)

air_df.to_csv(
    "historical_air_quality_4_cities.csv",
    index=False
)

print(air_df.head())
print(air_df.shape)