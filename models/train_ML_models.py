from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_ml_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            verbose=False,
            random_seed=42,
        ),
    }

    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        print(f"Training {name}")

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        trained_models[name] = model
        predictions[name] = pred

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    results_df = pd.DataFrame(results)

    return trained_models, predictions, results_df


def select_best_ml_model(trained_models, predictions, results_df):
    # Primary criterion: lowest MAE
    best_mae = results_df["MAE"].min()
    candidates = results_df[results_df["MAE"] == best_mae]

    # Tie-breaker: highest R2
    if len(candidates) > 1:
        best_row = candidates.loc[candidates["R2"].idxmax()]
    else:
        best_row = candidates.iloc[0]

    winner_name = best_row["model"]

    return (
        winner_name,
        trained_models[winner_name],
        predictions[winner_name],
        best_row,
    )