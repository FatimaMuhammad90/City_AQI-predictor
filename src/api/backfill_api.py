from datetime import datetime
import time
import pandas as pd
import requests
# the api helper for daily monitoring pipeline
CITIES = {
    "Islamabad": (33.6844, 73.0479),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Rawalpindi": (33.5651, 73.0169),
}

def validate_city(city):
    if not isinstance(city, str):
        raise TypeError(f"city must be a string, got {type(city).__name__}")
    if city not in CITIES:
        raise ValueError(f"Unknown city: {city}. Valid cities: {list(CITIES.keys())}")

def normalize_timestamp(timestamp):
    try:
        ts = pd.Timestamp(timestamp)
    except Exception as e:
        raise ValueError(f"Invalid timestamp: {timestamp}") from e
    if pd.isna(ts):
        raise ValueError(f"Timestamp cannot be NaT: {timestamp}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

def normalize_timestamps(timestamps):
    if timestamps is None:
        return []
    if isinstance(timestamps, (datetime, pd.Timestamp, str)):
        timestamps = [timestamps]
    if not isinstance(timestamps, (list, tuple, set)):
        raise TypeError("timestamps must be a list, tuple, set, datetime, Timestamp, or string")
    return [normalize_timestamp(ts) for ts in timestamps]

def validate_aqi_value(value, city=None, timestamp=None):
    if value is None or pd.isna(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        print(f"WARNING: Invalid AQI value{f' for {city}' if city else ''}{f' at {timestamp}' if timestamp else ''}: {value}")
        return None
    if not pd.isfinite(value) or value < 0:
        print(f"WARNING: Non-finite or negative AQI value{f' for {city}' if city else ''}{f' at {timestamp}' if timestamp else ''}: {value}")
        return None
    return value

def make_request(url, params, retries=3, timeout=60):
    if not url:
        raise ValueError("URL cannot be empty")
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    if retries < 1 or timeout <= 0:
        raise ValueError("retries must be >= 1 and timeout must be > 0")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"Request failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")

def get_aqi_at_timestamp(city, target_time):
    validate_city(city)
    target_time = normalize_timestamp(target_time)
    latitude, longitude = CITIES[city]
    date_str = target_time.strftime("%Y-%m-%d")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {"latitude": latitude, "longitude": longitude, "hourly": "us_aqi", "start_date": date_str, "end_date": date_str, "timezone": "GMT"}
    try:
        response = make_request(url, params)
        data = response.json()
        if not isinstance(data, dict) or "hourly" not in data or "time" not in data["hourly"] or "us_aqi" not in data["hourly"]:
            print(f"No hourly AQI data or incomplete response for {city} on {date_str}")
            return None
        df = pd.DataFrame(data["hourly"])
        if df.empty:
            print(f"Empty AQI response for {city} on {date_str}")
            return None
        df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])
        if df.empty:
            print(f"No valid timestamps returned for {city} on {date_str}")
            return None
        target_hour = target_time.floor("h")
        matched = df[df["timestamp_utc"] == target_hour]
        if matched.empty:
            df["time_diff"] = (df["timestamp_utc"] - target_hour).abs()
            closest = df.loc[df["time_diff"].idxmin()]
            if closest["time_diff"] <= pd.Timedelta(hours=1):
                aqi = validate_aqi_value(closest["us_aqi"], city, target_hour)
                if aqi is not None:
                    print(f"Fetched AQI for {city} at {target_hour}: {aqi} (closest match)")
                return aqi
            print(f"No close AQI match for {city} at {target_hour}")
            return None
        aqi = validate_aqi_value(matched.iloc[0]["us_aqi"], city, target_hour)
        if aqi is not None:
            print(f"Fetched AQI for {city} at {target_hour}: {aqi}")
        return aqi
    except Exception as e:
        print(f"Error fetching AQI for {city} at {target_time}: {e}")
        return None

def get_aqi_for_timestamps(city, timestamps):
    validate_city(city)
    timestamps = normalize_timestamps(timestamps)
    if not timestamps:
        return {}
    latitude, longitude = CITIES[city]
    date_groups = {}
    for ts in timestamps:
        date_groups.setdefault(ts.date(), []).append(ts)
    results = {}
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    for date_key, ts_list in date_groups.items():
        print(f"\nFetching AQI for {city} on {date_key}...")
        date_str = date_key.strftime("%Y-%m-%d")
        params = {"latitude": latitude, "longitude": longitude, "hourly": "us_aqi", "start_date": date_str, "end_date": date_str, "timezone": "GMT"}
        try:
            response = make_request(url, params)
            data = response.json()
            if not isinstance(data, dict) or "hourly" not in data or "time" not in data["hourly"] or "us_aqi" not in data["hourly"]:
                print(f"Incomplete or missing AQI data for {city} on {date_str}")
                for ts in ts_list:
                    results[ts] = None
                continue
            df = pd.DataFrame(data["hourly"])
            if df.empty:
                print(f"Empty AQI response for {city} on {date_str}")
                for ts in ts_list:
                    results[ts] = None
                continue
            df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp_utc"])
            aqi_lookup = {}
            for _, row in df.iterrows():
                timestamp = row["timestamp_utc"]
                aqi_lookup[timestamp] = validate_aqi_value(row["us_aqi"], city, timestamp)
            for ts in ts_list:
                target_hour = ts.floor("h")
                if target_hour in aqi_lookup:
                    results[ts] = aqi_lookup[target_hour]
                    print(f"  {target_hour}: {results[ts]}")
                elif aqi_lookup:
                    closest_time = min(aqi_lookup.keys(), key=lambda x: abs(x - target_hour))
                    difference = abs(closest_time - target_hour)
                    if difference <= pd.Timedelta(hours=1):
                        results[ts] = aqi_lookup[closest_time]
                        print(f"  {target_hour}: {results[ts]} (closest: {closest_time})")
                    else:
                        results[ts] = None
                        print(f"  {target_hour}: No data found")
                else:
                    results[ts] = None
                    print(f"  {target_hour}: No AQI data")
        except Exception as e:
            print(f"Error fetching AQI for {city} on {date_str}: {e}")
            for ts in ts_list:
                results[ts] = None
    return results

def get_aqi_batch(city, target_times):
    validate_city(city)
    target_times = normalize_timestamps(target_times)
    if not target_times:
        return {}
    return get_aqi_for_timestamps(city, target_times)

def get_historical_weather(city, target_times):
    validate_city(city)
    target_times = normalize_timestamps(target_times)
    if not target_times:
        return {}
    latitude, longitude = CITIES[city]
    date_groups = {}
    for ts in target_times:
        date_groups.setdefault(ts.date(), []).append(ts)
    results = {}
    url = "https://archive-api.open-meteo.com/v1/archive"
    for date_key, timestamps in date_groups.items():
        print(f"\nFetching historical weather for {city} on {date_key}...")
        date_str = date_key.strftime("%Y-%m-%d")
        params = {"latitude": latitude, "longitude": longitude, "start_date": date_str, "end_date": date_str, "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover", "timezone": "GMT"}
        try:
            response = make_request(url, params)
            data = response.json()
            if not isinstance(data, dict) or "hourly" not in data:
                print(f"No historical weather for {city} on {date_str}")
                for ts in timestamps:
                    results[ts] = None
                continue
            required_columns = ["time", "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover"]
            missing = [col for col in required_columns if col not in data["hourly"]]
            if missing:
                print(f"Incomplete weather response for {city} on {date_str}. Missing: {missing}")
                for ts in timestamps:
                    results[ts] = None
                continue
            df = pd.DataFrame(data["hourly"])
            if df.empty:
                print(f"Empty weather response for {city} on {date_str}")
                for ts in timestamps:
                    results[ts] = None
                continue
            df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp_utc"]).drop(columns=["time"]).rename(columns={"temperature_2m": "temperature", "relative_humidity_2m": "humidity", "surface_pressure": "pressure", "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction"})
            weather_lookup = df.set_index("timestamp_utc").to_dict("index")
            for ts in timestamps:
                target_hour = ts.floor("h")
                if target_hour not in weather_lookup:
                    results[ts] = None
                    print(f"  {target_hour}: weather not found")
                    continue
                weather = weather_lookup[target_hour]
                clean_weather = {}
                for field, value in weather.items():
                    if value is None or pd.isna(value):
                        clean_weather[field] = None
                    else:
                        try:
                            value = float(value)
                            clean_weather[field] = value if pd.isfinite(value) else None
                        except (TypeError, ValueError):
                            clean_weather[field] = None
                results[ts] = clean_weather
                print(f"  {target_hour}: weather found")
        except Exception as e:
            print(f"Error fetching historical weather for {city} on {date_str}: {e}")
            for ts in timestamps:
                results[ts] = None
    return results