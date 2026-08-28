from datetime import datetime
import time
import pandas as pd
import requests

CITIES = {
    "Islamabad": (33.6844, 73.0479),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Rawalpindi": (33.5651, 73.0169),
}


def make_request(url, params, retries=3, timeout=60):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise


def get_aqi_at_timestamp(city, target_time):
    if city not in CITIES:
        raise ValueError(f"Unknown city: {city}")

    latitude, longitude = CITIES[city]
    target_time = pd.Timestamp(target_time).tz_convert("UTC")
    date_str = target_time.strftime("%Y-%m-%d")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "us_aqi",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "GMT",
    }

    try:
        response = make_request(url, params)
        data = response.json()

        if "hourly" not in data:
            print(f"No hourly data for {city} on {date_str}")
            return None

        df = pd.DataFrame(data["hourly"])
        df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)

        target_hour = target_time.floor("h")
        matched = df[df["timestamp_utc"] == target_hour]

        if matched.empty:
            df["time_diff"] = abs(df["timestamp_utc"] - target_hour)
            closest = df.loc[df["time_diff"].idxmin()]

            if closest["time_diff"] <= pd.Timedelta(hours=1):
                aqi = float(closest["us_aqi"])
                print(
                    f"Fetched AQI for {city} at {target_hour}: {aqi} (closest match)"
                )
                return aqi

            print(f"No close match for {city} at {target_hour}")
            return None

        aqi = float(matched.iloc[0]["us_aqi"])
        print(f"Fetched AQI for {city} at {target_hour}: {aqi}")
        return aqi

    except Exception as e:
        print(f"Error fetching AQI for {city} at {target_time}: {e}")
        return None


def get_aqi_for_timestamps(city, timestamps):
    if not timestamps:
        return {}

    date_groups = {}
    for ts in timestamps:
        ts = pd.Timestamp(ts).tz_convert("UTC")
        date_groups.setdefault(ts.date(), []).append(ts)

    results = {}
    latitude, longitude = CITIES[city]
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    for date_key, ts_list in date_groups.items():
        print(f"\nFetching AQI for {city} on {date_key}...")
        date_str = date_key.strftime("%Y-%m-%d")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "us_aqi",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "GMT",
        }

        try:
            response = make_request(url, params)
            data = response.json()

            if "hourly" not in data:
                print(f"No hourly data for {city} on {date_str}")
                for ts in ts_list:
                    results[ts] = None
                continue

            df = pd.DataFrame(data["hourly"])
            df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)
            aqi_lookup = dict(zip(df["timestamp_utc"], df["us_aqi"]))

            for ts in ts_list:
                target_hour = ts.floor("h")

                if target_hour in aqi_lookup:
                    value = aqi_lookup[target_hour]
                    results[ts] = None if pd.isna(value) else float(value)
                    print(f"  {target_hour}: {results[ts]}")

                elif aqi_lookup:
                    closest_time = min(
                        aqi_lookup.keys(), key=lambda x: abs(x - target_hour)
                    )
                    difference = abs(closest_time - target_hour)

                    if difference <= pd.Timedelta(hours=1):
                        value = aqi_lookup[closest_time]
                        results[ts] = (
                            None if pd.isna(value) else float(value)
                        )
                        print(
                            f"  {target_hour}: {results[ts]} (closest: {closest_time})"
                        )
                    else:
                        results[ts] = None
                        print(f"  {target_hour}: No data found")
                else:
                    results[ts] = None
                    print(f"  {target_hour}: No data")

        except Exception as e:
            print(f"Error fetching data for {city} on {date_str}: {e}")
            for ts in ts_list:
                results[ts] = None

    return results


def get_aqi_batch(city, target_times):
    if not target_times:
        return {}

    if isinstance(target_times, (datetime, pd.Timestamp)):
        target_times = [target_times]

    clean_times = [pd.Timestamp(ts).tz_convert("UTC") for ts in target_times]
    return get_aqi_for_timestamps(city, clean_times)


def get_historical_weather(city, target_times):
    """Fetch actual historical weather for requested timestamps using Open-Meteo archive."""
    if city not in CITIES:
        raise ValueError(f"Unknown city: {city}")

    if not target_times:
        return {}

    latitude, longitude = CITIES[city]
    date_groups = {}

    for ts in target_times:
        ts = pd.Timestamp(ts).tz_convert("UTC")
        date_groups.setdefault(ts.date(), []).append(ts)

    results = {}
    url = "https://archive-api.open-meteo.com/v1/archive"

    for date_key, timestamps in date_groups.items():
        print(f"\nFetching historical weather for {city} on {date_key}...")
        date_str = date_key.strftime("%Y-%m-%d")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": (
                "temperature_2m,relative_humidity_2m,surface_pressure,"
                "wind_speed_10m,wind_direction_10m,precipitation,cloud_cover"
            ),
            "timezone": "GMT",
        }

        try:
            response = make_request(url, params)
            data = response.json()

            if "hourly" not in data:
                print(f"No historical weather for {city} on {date_str}")
                for ts in timestamps:
                    results[ts] = None
                continue

            df = pd.DataFrame(data["hourly"])
            df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)
            df = df.drop(columns=["time"]).rename(
                columns={
                    "temperature_2m": "temperature",
                    "relative_humidity_2m": "humidity",
                    "surface_pressure": "pressure",
                    "wind_speed_10m": "wind_speed",
                    "wind_direction_10m": "wind_direction",
                }
            )

            weather_lookup = df.set_index("timestamp_utc").to_dict("index")

            for ts in timestamps:
                target_hour = ts.floor("h")
                if target_hour in weather_lookup:
                    results[ts] = weather_lookup[target_hour]
                    print(f"  {target_hour}: weather found")
                else:
                    results[ts] = None
                    print(f"  {target_hour}: weather not found")

        except Exception as e:
            print(
                f"Error fetching historical weather for {city} on {date_str}: {e}"
            )
            for ts in timestamps:
                results[ts] = None

    return results