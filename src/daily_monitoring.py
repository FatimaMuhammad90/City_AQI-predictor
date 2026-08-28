import os
from datetime import datetime, timedelta, date

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_KEY must be set"
    )

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


CITIES = {
    "Islamabad": (33.6844, 73.0479),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Rawalpindi": (33.5651, 73.0169),
}


BASELINE_MAE = {
    24: 12.18,
    48: 15.84,
    72: 16.07,
}


HORIZON_MODEL = {
    24: "xgboost",
    48: "catboost",
    72: "random_forest",
}


ERROR_THRESHOLD = 20
MAE_MARGIN = 1.15
CONSECUTIVE_BAD_DAYS = 5


# ============================================================
# HTTP
# ============================================================

def make_request(url, params, retries=3, timeout=60):

    for attempt in range(1, retries + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=timeout
            )

            response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:

            print(
                f"Request failed "
                f"(attempt {attempt}/{retries}): {e}"
            )

            if attempt < retries:
                import time
                time.sleep(5)

            else:
                raise


# ============================================================
# ACTUAL AQI
# ============================================================

def get_actual_aqi(city, target_times):

    if not target_times:
        return {}

    latitude, longitude = CITIES[city]

    target_times = [
        pd.Timestamp(t).tz_convert("UTC")
        if pd.Timestamp(t).tzinfo
        else pd.Timestamp(t).tz_localize("UTC")
        for t in target_times
    ]

    start_date = min(target_times).strftime("%Y-%m-%d")
    end_date = max(target_times).strftime("%Y-%m-%d")

    url = (
        "https://air-quality-api.open-meteo.com/"
        "v1/air-quality"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "us_aqi,pm2_5,pm10,"
                  "carbon_monoxide,"
                  "nitrogen_dioxide,"
                  "sulphur_dioxide,"
                  "ozone",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }

    response = make_request(url, params)

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"No AQI data returned for {city}"
        )

    df = pd.DataFrame(data["hourly"])

    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.drop(columns=["time"])

    df = df.rename(
        columns={
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "sulphur_dioxide": "so2",
            "ozone": "o3"
        }
    )

    lookup = {}

    for _, row in df.iterrows():

        timestamp = row["timestamp_utc"]

        lookup[timestamp] = {
            "us_aqi": row["us_aqi"],
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "co": row["co"],
            "no2": row["no2"],
            "so2": row["so2"],
            "o3": row["o3"]
        }

    return lookup


# ============================================================
# WEATHER
# ============================================================

def get_weather(city, start_date, end_date):

    latitude, longitude = CITIES[city]

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
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "GMT",
    }

    response = make_request(url, params)

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"No weather data returned for {city}"
        )

    df = pd.DataFrame(data["hourly"])

    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.drop(columns=["time"])

    df = df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "surface_pressure": "pressure",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction"
        }
    )

    return df


# ============================================================
# PENDING PREDICTIONS
# ============================================================

def get_pending_predictions():

    now = datetime.now().astimezone().isoformat()

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("evaluated", False)
        .lt("target_time", now)
        .execute()
    )

    return response.data


# ============================================================
# UPDATE PREDICTION
# ============================================================

def update_prediction(
    prediction_id,
    actual_aqi,
    error_percent
):

    data = {
        "actual_aqi": actual_aqi,
        "error_percent": error_percent,
        "evaluated": True
    }

    return (
        supabase
        .table("predictions")
        .update(data)
        .eq("id", prediction_id)
        .execute()
    )


def calculate_error(predicted, actual):

    if actual is None or actual == 0:
        return None

    return round(
        abs(predicted - actual) / actual * 100,
        2
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

def store_historical_data(rows):

    if not rows:
        return

    try:

        (
            supabase
            .table("historical_data")
            .upsert(
                rows,
                on_conflict="city,timestamp_utc"
            )
            .execute()
        )

        print(
            f"Stored {len(rows)} historical observations."
        )

    except Exception as e:

        print(
            f"WARNING: Historical data storage failed: {e}"
        )


# ============================================================
# BACKFILL PREDICTIONS
# ============================================================

def backfill_predictions():

    print("\n=== DAILY BACKFILL ===")

    pending = get_pending_predictions()

    if not pending:

        print("No pending predictions.")

        return

    print(
        f"Found {len(pending)} pending predictions."
    )

    city_groups = {}

    for prediction in pending:

        city = prediction["city"]

        city_groups.setdefault(
            city,
            []
        ).append(prediction)

    for city, predictions in city_groups.items():

        print(
            f"\nProcessing {city}: "
            f"{len(predictions)} predictions"
        )

        target_times = [
            pd.Timestamp(
                p["target_time"]
            )
            for p in predictions
        ]

        try:

            # ------------------------------------------------
            # Get AQI
            # ------------------------------------------------

            aqi_lookup = get_actual_aqi(
                city,
                target_times
            )

            # ------------------------------------------------
            # Get weather
            # ------------------------------------------------

            min_date = min(target_times).date()
            max_date = max(target_times).date()

            weather_df = get_weather(
                city,
                min_date,
                max_date
            )

            weather_lookup = (
                weather_df
                .set_index("timestamp_utc")
                .to_dict("index")
            )

            historical_rows = []

            # ------------------------------------------------
            # Process predictions
            # ------------------------------------------------

            for prediction in predictions:

                prediction_id = prediction["id"]

                target_time = pd.Timestamp(
                    prediction["target_time"]
                ).tz_convert("UTC")

                actual_data = aqi_lookup.get(
                    target_time
                )

                if not actual_data:

                    print(
                        f"  No AQI available for "
                        f"{target_time}"
                    )

                    continue

                actual_aqi = actual_data["us_aqi"]

                if pd.isna(actual_aqi):

                    print(
                        f"  AQI is null for "
                        f"{target_time}"
                    )

                    continue

                actual_aqi = float(actual_aqi)

                predicted_aqi = float(
                    prediction["predicted_aqi"]
                )

                error = calculate_error(
                    predicted_aqi,
                    actual_aqi
                )

                # --------------------------------------------
                # Update prediction
                # --------------------------------------------

                try:

                    update_prediction(
                        prediction_id,
                        actual_aqi,
                        error
                    )

                    print(
                        f"  ✓ Prediction {prediction_id}: "
                        f"pred={predicted_aqi:.2f}, "
                        f"actual={actual_aqi:.2f}, "
                        f"error={error}%"
                    )

                except Exception as e:

                    print(
                        f"  WARNING: Could not update "
                        f"prediction {prediction_id}: {e}"
                    )

                # --------------------------------------------
                # Historical dataset
                # --------------------------------------------

                weather = weather_lookup.get(
                    target_time
                )

                if weather is None:

                    print(
                        f"  WARNING: No weather for "
                        f"{target_time}"
                    )

                    continue

                historical_rows.append({

                    "timestamp_utc":
                        target_time.isoformat(),

                    "city":
                        city,

                    "latitude":
                        CITIES[city][0],

                    "longitude":
                        CITIES[city][1],

                    "pm2_5":
                        float(actual_data["pm2_5"])
                        if pd.notna(actual_data["pm2_5"])
                        else None,

                    "pm10":
                        float(actual_data["pm10"])
                        if pd.notna(actual_data["pm10"])
                        else None,

                    "co":
                        float(actual_data["co"])
                        if pd.notna(actual_data["co"])
                        else None,

                    "no2":
                        float(actual_data["no2"])
                        if pd.notna(actual_data["no2"])
                        else None,

                    "so2":
                        float(actual_data["so2"])
                        if pd.notna(actual_data["so2"])
                        else None,

                    "o3":
                        float(actual_data["o3"])
                        if pd.notna(actual_data["o3"])
                        else None,

                    "us_aqi":
                        actual_aqi,

                    "temperature":
                        weather.get("temperature"),

                    "humidity":
                        weather.get("humidity"),

                    "pressure":
                        weather.get("pressure"),

                    "wind_speed":
                        weather.get("wind_speed"),

                    "wind_direction":
                        weather.get("wind_direction"),

                    "precipitation":
                        weather.get("precipitation"),

                    "cloud_cover":
                        weather.get("cloud_cover"),
                })

            # --------------------------------------------
            # Save historical observations
            # --------------------------------------------

            store_historical_data(
                historical_rows
            )

        except Exception as e:

            print(
                f"ERROR processing {city}: {e}"
            )

            # Important:
            # one city failing does NOT kill the
            # entire monitoring pipeline.


# ============================================================
# WEEKLY MONITORING
# ============================================================

def get_previous_week():

    today = date.today()

    this_monday = (
        today - timedelta(
            days=today.weekday()
        )
    )

    previous_monday = (
        this_monday - timedelta(days=7)
    )

    previous_sunday = (
        this_monday - timedelta(days=1)
    )

    return previous_monday, previous_sunday


def get_week_predictions(
    city,
    horizon,
    start_date,
    end_date
):

    start = (
        pd.Timestamp(start_date)
        .tz_localize("UTC")
        .isoformat()
    )

    end = (
        pd.Timestamp(end_date)
        .tz_localize("UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    ).isoformat()

    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("city", city)
        .eq("horizon", horizon)
        .eq("evaluated", True)
        .gte("target_time", start)
        .lte("target_time", end)
        .order(
            "target_time",
            desc=False
        )
        .execute()
    )

    return response.data


def calculate_mae(predictions):

    if not predictions:
        return None

    df = pd.DataFrame(predictions)

    df = df.dropna(
        subset=[
            "predicted_aqi",
            "actual_aqi"
        ]
    )

    if df.empty:
        return None

    return round(
        (
            df["predicted_aqi"]
            - df["actual_aqi"]
        ).abs().mean(),
        2
    )


def get_daily_status(predictions):

    if not predictions:
        return {}

    df = pd.DataFrame(predictions)

    df["target_time"] = pd.to_datetime(
        df["target_time"],
        utc=True
    )

    df["date"] = (
        df["target_time"]
        .dt.date
    )

    daily = {}

    for day, group in df.groupby("date"):

        # Day is bad if prediction error
        # exceeds 20%.

        daily[day] = (
            group["error_percent"]
            .gt(ERROR_THRESHOLD)
            .any()
        )

    return daily


def get_max_consecutive_bad_days(
    daily_status
):

    if not daily_status:
        return 0

    dates = sorted(
        daily_status.keys()
    )

    current = 0
    maximum = 0

    previous_date = None

    for current_date in dates:

        if (
            daily_status[current_date]
            and (
                previous_date is None
                or current_date
                == previous_date
                + timedelta(days=1)
            )
        ):

            current += 1

        elif daily_status[current_date]:

            current = 1

        else:

            current = 0

        maximum = max(
            maximum,
            current
        )

        previous_date = current_date

    return maximum


# ============================================================
# WEEKLY MONITOR
# ============================================================

def run_weekly_monitoring():

    today = date.today()

    if today.weekday() != 0:

        print(
            "Not Monday. "
            "Weekly monitoring skipped."
        )

        return

    print(
        "\n=== MONDAY MODEL MONITORING ==="
    )

    start_date, end_date = get_previous_week()

    print(
        f"Evaluating: "
        f"{start_date} → {end_date}"
    )

    cities = list(CITIES.keys())
    horizons = [24, 48, 72]

    results = []

    for city in cities:

        for horizon in horizons:

            model = HORIZON_MODEL[horizon]

            baseline = BASELINE_MAE[horizon]

            try:

                predictions = get_week_predictions(
                    city,
                    horizon,
                    start_date,
                    end_date
                )

                if not predictions:

                    print(
                        f"{city} {horizon}h: "
                        f"No evaluated predictions"
                    )

                    continue

                mae = calculate_mae(
                    predictions
                )

                daily_status = get_daily_status(
                    predictions
                )

                consecutive_bad = (
                    get_max_consecutive_bad_days(
                        daily_status
                    )
                )

                bad_days = sum(
                    daily_status.values()
                )

                flag_bad_days = (
                    consecutive_bad
                    >= CONSECUTIVE_BAD_DAYS
                )

                flag_mae = (
                    mae is not None
                    and mae
                    > baseline * MAE_MARGIN
                )

                flagged = (
                    flag_bad_days
                    or flag_mae
                )

                print(
                    f"\n{city} | "
                    f"{horizon}h | "
                    f"{model}"
                )

                print(
                    f"Predictions: "
                    f"{len(predictions)}"
                )

                print(
                    f"MAE: {mae}"
                )

                print(
                    f"Baseline: {baseline}"
                )

                print(
                    f"Bad days: "
                    f"{bad_days}"
                )

                print(
                    f"Consecutive bad: "
                    f"{consecutive_bad}"
                )

                print(
                    f"FLAGGED: {flagged}"
                )

                results.append({

                    "check_date":
                        today.isoformat(),

                    "city":
                        city,

                    "model":
                        model,

                    "horizon":
                        horizon,

                    "total_predictions":
                        len(predictions),

                    "bad_predictions":
                        bad_days,

                    "consecutive_bad_days":
                        consecutive_bad,

                    "mae":
                        mae,

                    "baseline_mae":
                        baseline,

                    "flagged":
                        flagged
                })

            except Exception as e:

                print(
                    f"WARNING: Monitoring failed "
                    f"for {city} {horizon}h: {e}"
                )

    if results:

        try:

            (
                supabase
                .table("monitoring")
                .insert(results)
                .execute()
            )

            print(
                "\n✓ Monitoring results saved."
            )

        except Exception as e:

            print(
                f"WARNING: Could not save "
                f"monitoring results: {e}"
            )

    print(
        "\n=== WEEKLY MONITORING COMPLETE ==="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )
    print(
        "DAILY AQI MONITORING PIPELINE"
    )
    print(
        "========================================"
    )

    # Part 1:
    # Always run daily.

    try:

        backfill_predictions()

    except Exception as e:

        print(
            f"CRITICAL WARNING: "
            f"Backfill section failed: {e}"
        )

    # Part 2:
    # Only runs Monday.

    try:

        run_weekly_monitoring()

    except Exception as e:

        print(
            f"CRITICAL WARNING: "
            f"Weekly monitoring failed: {e}"
        )

    print(
        "\n========================================"
    )
    print(
        "DAILY MONITORING COMPLETE"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()