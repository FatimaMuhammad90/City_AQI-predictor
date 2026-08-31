import subprocess
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEAST_REPO = os.path.join(PROJECT_ROOT,"feast_st")

FEATURE_FILE = os.path.join( FEAST_REPO, "data","aqi_features.parquet")
def update_feast(features):
    features = features.copy()

    features["timestamp_utc"] = pd.to_datetime(
        features["timestamp_utc"],
        utc=True
    )

    try:
        old = pd.read_parquet(FEATURE_FILE)

        features = pd.concat(  [old, features],ignore_index=True)

    except FileNotFoundError:
        pass

    features = features.drop_duplicates( subset=["city", "timestamp_utc"],keep="last")
    features = features.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    features.to_parquet(
        FEATURE_FILE,
        index=False
    )
    print("WRITING PARQUET TO:")
    print(FEATURE_FILE)

    print(
        "LATEST TIMESTAMP BEING SAVED:",
        features["timestamp_utc"].max()
    )
    subprocess.run(
        ["feast", "apply"],
        cwd=FEAST_REPO,
        check=True
    )

    latest_timestamp = features["timestamp_utc"].max()

    subprocess.run(
        [
            "feast",
            "materialize-incremental",
            latest_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        ],
        cwd=FEAST_REPO,
        check=True
    )

    print("Feast online store updated.")