import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

# Original LSTM modified to fit the retraining script
def evaluate_predictions(data, predictions, horizon):

    actual = data["y_test"]
    cities = data["test_cities"]
    origins = pd.to_datetime(
        data["test_origins"],
        utc=True
    )

    df = data["df_processed"].copy()
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True
    )
    horizon_hours = {
        "target_24h": 24,
        "target_48h": 48,
        "target_72h": 72
    }[horizon]

    baseline_times = origins - pd.Timedelta(hours=horizon_hours)

    baseline_df = pd.DataFrame({
        "city": cities,
        "baseline_time": baseline_times
    })

    lookup_df = df[
        ["city", "timestamp_utc", "us_aqi"]
    ].rename(
        columns={
            "timestamp_utc": "baseline_time",
            "us_aqi": "baseline_prediction"
        }
    )
    baseline_df = baseline_df.merge(
        lookup_df,
        on=["city", "baseline_time"],
        how="left"
    )

    baseline_predictions = baseline_df[
        "baseline_prediction"
    ].values

    valid = ~np.isnan(baseline_predictions)

    baseline_mae = mean_absolute_error(
        actual[valid],
        baseline_predictions[valid]
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual[valid],
            baseline_predictions[valid]
        )
    )

    lstm_mae = mean_absolute_error(actual,predictions)
    lstm_rmse = np.sqrt(mean_squared_error(actual,predictions))
    improvement = ((baseline_mae - lstm_mae)/ baseline_mae) * 100

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    print(f"LSTM MAE:        {lstm_mae:.2f}")
    print(f"Baseline MAE:    {baseline_mae:.2f}")
    print(f"LSTM RMSE:       {lstm_rmse:.2f}")
    print(f"Baseline RMSE:   {baseline_rmse:.2f}")
    print(f"MAE improvement: {improvement:.2f}%")


    print("\n" + "=" * 60)
    print("PER-CITY PERFORMANCE")
    print("=" * 60)

    city_results = []

    for city in np.unique(cities):

        mask = cities == city

        city_actual = actual[mask]
        city_predictions = predictions[mask]

        city_mae = mean_absolute_error(
            city_actual,
            city_predictions
        )

        city_rmse = np.sqrt(
            mean_squared_error(
                city_actual,
                city_predictions
            )
        )

        city_r2 = r2_score(
            city_actual,
            city_predictions
        )

        city_results.append({
            "city": city,
            "MAE": city_mae,
            "RMSE": city_rmse,
            "R2": city_r2
        })

        print(
            f"{city}: "
            f"MAE={city_mae:.2f}, "
            f"RMSE={city_rmse:.2f}, "
            f"R²={city_r2:.3f}"
        )


    print("\n" + "=" * 60)
    print("PERFORMANCE BY AQI RANGE")
    print("=" * 60)

    bins = [0, 50, 100, 150, 200, 300, np.inf]

    labels = [
        "0-50",
        "51-100",
        "101-150",
        "151-200",
        "201-300",
        "300+"
    ]

    categories = pd.cut(
        actual,
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    for category in labels:

        mask = categories == category

        if mask.sum() == 0:
            continue

        category_mae = mean_absolute_error(
            actual[mask],
            predictions[mask]
        )

        print(
            f"{category}: "
            f"n={mask.sum()}, "
            f"MAE={category_mae:.2f}"
        )

    return {
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "city_results": city_results
    }
def prepare_lstm_data(
    df=None,
    filepath=None,
    target_column="target_24h",
    sequence_length=24,
    test_size=0.2,
):
    # --------------------------------------------------------
    # 1. Input source check
    # --------------------------------------------------------
    if df is None:
        if filepath is None:
            raise ValueError("Either df or filepath must be provided")
        df = pd.read_csv(filepath)
    else:
        df = df.copy()

    # --------------------------------------------------------
    # 2. Map integer horizon to column name (e.g., 24 -> "target_24h")
    # --------------------------------------------------------
    if isinstance(target_column, int):
        target_column = f"target_{target_column}h"

    valid_targets = ["target_24h", "target_48h", "target_72h"]
    if target_column not in valid_targets:
        raise ValueError(f"target_column must be one of {valid_targets} or [24, 48, 72]")

    # --------------------------------------------------------
    # 3. Ensure target columns exist in raw df
    # --------------------------------------------------------
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    if target_column not in df.columns:
        df["target_24h"] = df.groupby("city")["us_aqi"].shift(-24)
        df["target_48h"] = df.groupby("city")["us_aqi"].shift(-48)
        df["target_72h"] = df.groupby("city")["us_aqi"].shift(-72)

    # --------------------------------------------------------
    # 4. Drop missing targets
    # --------------------------------------------------------
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    encoder = LabelEncoder()
    df["city_encoded"] = encoder.fit_transform(df["city"])

    exclude_columns = [
        "timestamp_utc",
        "city",
        "us_aqi",
        "target_24h",
        "target_48h",
        "target_72h",
        "created_at",
        "id"
    ]

    # Select only numeric/boolean features to prevent string conversion errors
    candidate_cols = [col for col in df.columns if col not in exclude_columns]
    numeric_df = df[candidate_cols].select_dtypes(include=["number", "bool"])
    feature_columns = list(numeric_df.columns)

    print(f"Number of features: {len(feature_columns)}")

    X_sequences = []
    y_sequences = []
    origin_times = []
    target_times = []
    city_names = []

    horizon_hours = {
        "target_24h": 24,
        "target_48h": 48,
        "target_72h": 72
    }[target_column]

    for city in df["city"].unique():
        city_df = df[df["city"] == city].sort_values("timestamp_utc")

        features = city_df[feature_columns].values
        targets = city_df[target_column].values
        timestamps = city_df["timestamp_utc"].values
        cities = city_df["city"].values

        for i in range(sequence_length, len(city_df)):
            X_sequences.append(features[i - sequence_length:i])
            y_sequences.append(targets[i])
            origin_times.append(timestamps[i])
            target_times.append(timestamps[i] + pd.Timedelta(hours=horizon_hours))
            city_names.append(cities[i])

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_sequences, dtype=np.float32)
    origin_times = np.array(origin_times)
    target_times = np.array(target_times)
    city_names = np.array(city_names)

    print(f"Total sequences created: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    unique_origins = np.sort(np.unique(origin_times))
    cutoff_idx = int(len(unique_origins) * (1 - test_size))
    cutoff_time = unique_origins[cutoff_idx]

    print(f"\nCutoff time: {cutoff_time}")
    print(f"Training origins: {cutoff_idx}")
    print(f"Test origins: {len(unique_origins) - cutoff_idx}")

    train_mask = origin_times < cutoff_time
    test_mask = origin_times >= cutoff_time

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    test_origins = origin_times[test_mask]
    test_targets = target_times[test_mask]
    test_cities = city_names[test_mask]

    scaler = StandardScaler()
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    X_train = X_train_scaled.reshape(n_train_samples, seq_len, n_features)
    X_test = X_test_scaled.reshape(n_test_samples, seq_len, n_features)

    print(f"\nFeatures scaled using StandardScaler")

    assert (test_origins >= cutoff_time).all(), "ERROR: Some test origins are before cutoff!"
    assert (origin_times[train_mask] < cutoff_time).all(), "ERROR: Some train origins are after cutoff!"
    print("Data integrity check passed - no leakage detected")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_origins": test_origins,
        "test_targets": test_targets,
        "test_cities": test_cities,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "encoder": encoder,
        "cutoff_time": cutoff_time,
        "df_processed": df
    }


def train_lstm_model(data, lstm_units=64, dropout_rate=0.2, epochs=50, batch_size=64):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Build model
    model = Sequential([
        LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    print("\n" + "=" * 60)
    print("TRAINING LSTM MODEL")
    print("=" * 60)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )

    predictions = model.predict(X_test, verbose=0).flatten()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return model, predictions, mae, rmse, r2, history


if __name__ == "__main__":
    filepath = "../data/combined_air_weather_5_cities_features.csv"
    horizons = ["target_24h", "target_48h", "target_72h"]
    results = {}

    for target in horizons:
        print("\n" + "=" * 60)
        print(f"TARGET: {target}")
        print("=" * 60)
        data = prepare_lstm_data(filepath=filepath,target_column=target,sequence_length=24,test_size=0.2)


        model, predictions, mae, rmse, r2, history = train_lstm_model(data,lstm_units=64, dropout_rate=0.2,epochs=50,batch_size=64)

        evaluate_predictions(data, predictions, target)

        results[target] = {
            "data": data,
            "model": model,
            "predictions": predictions,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "history": history
        }

        results_df = pd.DataFrame({
            "origin_time": data["test_origins"],
            "target_time": data["test_targets"],
            "city": data["test_cities"],
            "actual": data["y_test"],
            "predicted": predictions
        })

    print(f"{'Horizon':<15} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
    print("-" * 60)
    for target in horizons:
        print(
            f"{target:<15} {results[target]['mae']:<15.4f} {results[target]['rmse']:<15.4f} {results[target]['r2']:<15.4f}")