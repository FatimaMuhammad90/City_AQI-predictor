import os

from dotenv import load_dotenv
import pandas as pd
from supabase import create_client

# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is missing from the environment variables.")

if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is missing from the environment variables.")


# ============================================================
# SUPABASE CLIENT
# ============================================================

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to create Supabase client: {e}")


# ============================================================
# REQUIRED SCHEMA
# ============================================================

REQUIRED_MONITORING_COLUMNS = {"horizon", "flagged"}

REQUIRED_HISTORICAL_COLUMNS = {"timestamp_utc", "city", "us_aqi"}


# ============================================================
# MONITORING DATA
# ============================================================

def get_flagged_horizons():
    print("\n[1/4] Checking monitoring table...")

    # --------------------------------------------------------
    # First retrieve a row so we can inspect the actual schema.
    # DO NOT filter by flagged before validating it exists.
    # --------------------------------------------------------
    try:
        response = (
            supabase.table("monitoring")
            .select("*")
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not access Supabase 'monitoring' table: {e}"
        )

    rows = response.data

    # --------------------------------------------------------
    # Empty table
    # --------------------------------------------------------
    if not rows:
        print("Monitoring table is empty.")
        print("No horizons are currently available for retraining.")
        return []

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------
    available_columns = set(rows[0].keys())
    missing_columns = REQUIRED_MONITORING_COLUMNS - available_columns

    if missing_columns:
        raise RuntimeError(
            "\nMonitoring table schema validation failed.\n"
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {sorted(available_columns)}\n\n"
            "The weekly retraining pipeline requires:\n"
            "  - horizon\n"
            "  - flagged\n\n"
            "Add the missing column(s) to Supabase before "
            "running the retraining pipeline."
        )

    print("✓ Monitoring table schema validated.")

    # --------------------------------------------------------
    # Now it is safe to query flagged
    # --------------------------------------------------------
    try:
        response = (
            supabase.table("monitoring")
            .select("horizon, flagged")
            .eq("flagged", True)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve flagged monitoring records: {e}"
        )

    rows = response.data

    if not rows:
        print("✓ No models have been flagged for retraining.")
        return []

    # --------------------------------------------------------
    # Validate returned records
    # --------------------------------------------------------
    flagged_horizons = set()
    valid_horizons = {24, 48, 72}

    for row in rows:
        horizon = row.get("horizon")

        if horizon is None:
            print(
                "WARNING: Found flagged monitoring record "
                "without a horizon. Skipping."
            )
            continue

        if horizon not in valid_horizons:
            print(
                f"WARNING: Unknown horizon '{horizon}'. "
                "Skipping."
            )
            continue

        flagged_horizons.add(horizon)

    flagged_horizons = sorted(flagged_horizons)

    if flagged_horizons:
        print(f"✓ Flagged horizons: {flagged_horizons}")
    else:
        print(
            "WARNING: Monitoring records were flagged, "
            "but none contained a valid horizon."
        )

    return flagged_horizons


# ============================================================
# HISTORICAL DATA
# ============================================================

def get_historical_data():
    print("\n[2/4] Retrieving historical data...")

    try:
        response = (
            supabase.table("historical_data")
            .select("*")
            .execute()
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not access Supabase 'historical_data' table: {e}"
        )

    rows = response.data

    if not rows:
        raise RuntimeError("historical_data table returned no records.")

    df = pd.DataFrame(rows)
    print(f"✓ Retrieved {len(df):,} historical records.")

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------
    missing_columns = REQUIRED_HISTORICAL_COLUMNS - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "\nhistorical_data schema validation failed.\n"
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {sorted(df.columns)}"
        )

    print("✓ Historical data schema validated.")

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )

    invalid_timestamps = df["timestamp_utc"].isna().sum()
    if invalid_timestamps > 0:
        raise RuntimeError(
            f"historical_data contains {invalid_timestamps} invalid timestamps."
        )

    # --------------------------------------------------------
    # City validation
    # --------------------------------------------------------
    if df["city"].isna().any():
        raise RuntimeError(
            "historical_data contains rows with missing city values."
        )

    # --------------------------------------------------------
    # AQI validation
    # --------------------------------------------------------
    df["us_aqi"] = pd.to_numeric(df["us_aqi"], errors="coerce")

    invalid_aqi = df["us_aqi"].isna().sum()
    if invalid_aqi > 0:
        raise RuntimeError(
            f"historical_data contains {invalid_aqi} invalid us_aqi values."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------
    duplicates = df.duplicated(subset=["city", "timestamp_utc"]).sum()
    if duplicates > 0:
        raise RuntimeError(
            f"historical_data contains {duplicates} duplicate city/timestamp records."
        )

    print("✓ Historical data integrity checks passed.")
    print(f"✓ Cities: {sorted(df['city'].unique())}")
    print(
        f"✓ Time range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}"
    )

    return df