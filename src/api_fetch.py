import requests
import pandas as pd


CITIES = {
    "Islamabad": (33.6844, 73.0479),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Rawalpindi": (33.5651, 73.0169),
}
def get_air_quality(latitude, longitude):

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "past_hours": 48,
        "forecast_days": 0,
        "timezone": "GMT",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        utc=True
    )
    now = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp_utc"] <= now
    ]
    df = (
        df
        .sort_values("timestamp_utc")
        .tail(48)
        .reset_index(drop=True)
    )

    df = df.drop(columns=["time"])

    df = df.rename(columns={
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3"
    })

    return df

def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "precipitation,"
            "cloud_cover"
        ),
        "forecast_days": 7,
        "timezone": "GMT",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])


    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.drop(columns=["time"])
    df = df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction"
    })
    return df


def fetch_city_data(city):

    if city not in CITIES:
        raise ValueError(
            f"Unknown city: {city}"
        )

    latitude, longitude = CITIES[city]

    air_df = get_air_quality(
        latitude,
        longitude
    )

    weather_df = get_weather(
        latitude,
        longitude
    )

    air_df["city"] = city
    air_df["latitude"] = latitude
    air_df["longitude"] = longitude

    weather_df["city"] = city
    weather_df["latitude"] = latitude
    weather_df["longitude"] = longitude

    return air_df, weather_df


def fetch_all_cities():
    all_air = []
    all_weather = []

    for city in CITIES:

        print(f"Fetching data for {city}...")

        air_df, weather_df = fetch_city_data(city)

        all_air.append(air_df)
        all_weather.append(weather_df)

    air_df = pd.concat(all_air,ignore_index=True)
    weather_df = pd.concat(all_weather,ignore_index=True)

    return air_df, weather_df


# if __name__ == "__main__":
#
#     air_df, weather_df = fetch_all_cities()
#     print("\nAQI TIME RANGE:")
#     print(air_df["timestamp_utc"].min())
#     print(air_df["timestamp_utc"].max())
#
#     print(air_df.head())
#     print("Shape:", air_df.shape)
#
#     print("\n==============================")
#     print("WEATHER")
#     print("==============================")
#     print(weather_df.head())
#     print("Shape:", weather_df.shape)