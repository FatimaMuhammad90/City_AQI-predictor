import os
import time
from datetime import datetime
from huggingface_hub import HfApi
from models.preprocessing import preprocess_data
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from models.supabase_data import get_flagged_horizons, get_historical_data, delete_monitoring_entry
from models.train_ML_models import select_best_ml_model, train_ml_models
from models.LSTM_model import prepare_lstm_data, train_lstm_model
from models.feature_engineering import create_features

HORIZONS = [24, 48 , 72]

load_dotenv()

def compare_final_models(ml_name, ml_model, ml_predictions, ml_stats, lstm_model, lstm_predictions,lstm_mae, lstm_rmse, lstm_r2,):
    ml_mae = ml_stats["MAE"]
    ml_r2 = ml_stats["R2"]

    print("FINAL MODEL COMPARISON")
    print(f"{ml_name}: MAE={ml_mae:.4f}, "
        f"RMSE={ml_stats['RMSE']:.4f}, "
    f"R²={ml_r2:.4f}" )

    print(
        f"LSTM: MAE={lstm_mae:.4f}, "
        f"RMSE={lstm_rmse:.4f}, "
        f"R²={lstm_r2:.4f}"
    )

    # MAE is the primary metric
    if ml_mae < lstm_mae:
        winner_name = ml_name
        winner_model = ml_model
        winner_predictions = ml_predictions
        winner_mae = ml_mae
        winner_rmse = ml_stats["RMSE"]
        winner_r2 = ml_r2

    elif lstm_mae < ml_mae:
        winner_name = "LSTM"
        winner_model = lstm_model
        winner_predictions = lstm_predictions
        winner_mae = lstm_mae
        winner_rmse = lstm_rmse
        winner_r2 = lstm_r2

    else:
        # to prevent ties, higher r2 wins
        if ml_r2 >= lstm_r2:
            winner_name = ml_name
            winner_model = ml_model
            winner_predictions = ml_predictions
            winner_mae = ml_mae
            winner_rmse = ml_stats["RMSE"]
            winner_r2 = ml_r2
        else:
            winner_name = "LSTM"
            winner_model = lstm_model
            winner_predictions = lstm_predictions
            winner_mae = lstm_mae
            winner_rmse = lstm_rmse
            winner_r2 = lstm_r2

    print("\nFINAL WINNER")
    print(
        f"{winner_name} | "
        f"MAE={winner_mae:.4f} | "
        f"RMSE={winner_rmse:.4f} | "
        f"R²={winner_r2:.4f}"
    )

    return {
        "name": winner_name,
        "model": winner_model,
        "predictions": winner_predictions,
        "MAE": winner_mae,
        "RMSE": winner_rmse,
        "R2": winner_r2,
    }

def train_ml_for_horizon(historical_df, horizon):
    print("\n" + "=" * 60)
    print(f"ML RETRAINING — {horizon}")
    print("=" * 60)
    processed = preprocess_data(df=historical_df, target_column=horizon)

    ( X_train, X_test, y_train, y_test, encoder,train,test,df_processed,test_cities,test_origins,) = processed

    trained_models, predictions, results_df = train_ml_models(
        X_train, X_test, y_train, y_test
    )

    print("\nML MODEL RESULTS")
    print(results_df)

    winner_name, winner_model, winner_predictions, winner_stats = (
        select_best_ml_model(trained_models, predictions, results_df)
    )

    print(f"\nBest ML model: {winner_name}")

    return {
        "name": winner_name,
        "model": winner_model,
        "predictions": winner_predictions,
        "MAE": winner_stats["MAE"],
        "RMSE": winner_stats["RMSE"],
        "R2": winner_stats["R2"],
        "processed": processed,
    }

def train_lstm_for_horizon(historical_df, horizon):
    print(f"LSTM RETRAINING — {horizon}")
    # LSTM receives raw historical data.
    # It performs its own target generation, sequence generation, split, scaling.
    lstm_data = prepare_lstm_data(
        df=historical_df,
        target_column=horizon,
        sequence_length=24,
        test_size=0.2,
    )

    model, predictions, mae, rmse, r2, history = train_lstm_model(
        lstm_data,
        lstm_units=64,
        dropout_rate=0.2,
        epochs=50,
        batch_size=64,
    )

    return {
        "name": "LSTM",
        "model": model,
        "predictions": predictions,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "processed": lstm_data,
        "history": history,
    }



def save_winner(winner, horizon):
    os.makedirs("../models/retrained", exist_ok=True)
    model_name = winner["name"]
    upload_successful = False
    current_date = datetime.now().strftime("%Y-%m-%d")

    if model_name == "LSTM":
        filename = f"{current_date}_{horizon}.keras"
        path = f"../models/retrained/{filename}"
        winner["model"].save(path)

    elif model_name == "CatBoost":
        filename = f"{current_date}_{horizon}.cbm"
        path = f"../models/retrained/{filename}"
        winner["model"].save_model(path)

    else:
        filename = f"{current_date}_{horizon}.pkl"
        path = f"../models/retrained/{filename}"
        joblib.dump(winner["model"], path)

    print(f"\nWinner saved locally: {path}")

    repo_id = "flork-18115/AQI_prediciton_models"

    # Grab the token from the environment
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("Warning: HF_TOKEN not found in environment. Upload will likely fail.")

    # Authenticate the API instance directly
    api = HfApi(token=hf_token)
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Uploading {filename} to {repo_id} (Attempt {attempt}/{max_retries})...")
            api.upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=repo_id,repo_type="model", commit_message=f"Automated upload for {horizon}h horizon. Best model: {model_name} (MAE: {winner.get('MAE', 'N/A'):.4f})")
            print("Upload successful!")
            upload_successful = True
            break

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 5 seconds.")
                time.sleep(5)
            else:
                print("Failed to upload after multiple attempts. Moving on.")
    if upload_successful:
        delete_monitoring_entry(horizon=horizon)

    return path

def train_one_horizon(historical_df, horizon):
    ml_result = train_ml_for_horizon(historical_df, horizon)

    lstm_result = train_lstm_for_horizon(historical_df, horizon)
    winner = compare_final_models(
        ml_name=ml_result["name"],
        ml_model=ml_result["model"],
        ml_predictions=ml_result["predictions"],
        ml_stats={
            "MAE": ml_result["MAE"],
            "RMSE": ml_result["RMSE"],
            "R2": ml_result["R2"],
        },
        lstm_model=lstm_result["model"],
        lstm_predictions=lstm_result["predictions"],
        lstm_mae=lstm_result["MAE"],
        lstm_rmse=lstm_result["RMSE"],
        lstm_r2=lstm_result["R2"],
    )
    model_path = save_winner(winner, horizon)
    return {
        "horizon": horizon,
        "winner": winner,
        "model_path": model_path,
        "ml_result": ml_result,
        "lstm_result": lstm_result,
    }

def main():
    print("=" * 60)
    print("WEEKLY MODEL RETRAINING PIPELINE")
    print("=" * 60)

    # 1. Check monitoring table
    flagged_horizons = get_flagged_horizons()

    if not flagged_horizons:
        print("No horizons flagged for retraining.")
        return

    print("\nFlagged horizons:", flagged_horizons)

    # 2. get historical data & Feature Engineer
    raw_historical_df = get_historical_data()
    print(f"\nRaw historical data loaded: {raw_historical_df.shape}")

    print("\nApplying Feature Engineering...")
    historical_df = create_features(raw_historical_df)
    print(f"Engineered historical data shape: {historical_df.shape}")

    results = {}

    for horizon in flagged_horizons:
        if horizon not in HORIZONS:
            print(f"Skipping unknown horizon: {horizon}")
            continue

        results[horizon] = train_one_horizon(historical_df, horizon)

    for horizon, result in results.items():
        winner = result["winner"]
        print(f"{horizon}: {winner['name']} (MAE={winner['MAE']:.4f})")

if __name__ == "__main__":
    main()