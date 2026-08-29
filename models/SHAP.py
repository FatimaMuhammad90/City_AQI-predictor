from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download
import joblib
from models.preprocessing import preprocess_data
from models.supabase_data import get_historical_data
import pandas as pd
import shap
REPO_ID = "flork-18115/AQI_prediciton_models"

historical_df = get_historical_data()
print(f"Historical data shape: {historical_df.shape}")


print("\n" + "=" * 60)
print("SHAP — XGBoost 24h")
print("=" * 60)

try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="xgboost_24h.pkl",
        repo_type="model",
    )

    xg_model = joblib.load(model_path)
    print("XGBoost model loaded successfully from Hugging Face!")

except Exception as e:
    print(f"XGBoost model could not be loaded: {e}")
    xg_model = None


if xg_model is not None:
    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(df=historical_df, target_column="target_24h")

    explainer = shap.TreeExplainer(xg_model)
    shap_values = explainer.shap_values(X_test)

    print("X_test shape:", X_test.shape)
    print("SHAP shape:", shap_values.shape)

    shap_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": abs(shap_values).mean(axis=0),
        }
    )

    shap_importance = shap_importance.sort_values(
        "importance", ascending=False
    )

    print("\nFeature Importance:")
    print(shap_importance)

    print("\nGenerating SHAP Summary Plot...")
    shap.summary_plot(shap_values, X_test)


# ============================================================
# 48 HOUR — CatBoost
# ============================================================

print("\n" + "=" * 60)
print("SHAP — CatBoost 48h")
print("=" * 60)

try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="catboost_48h.cbm",
        repo_type="model",
    )

    cat_model = CatBoostRegressor()
    cat_model.load_model(model_path)

    print("CatBoost model loaded successfully from Hugging Face!")

except Exception as e:
    print(f"CatBoost model could not be loaded: {e}")
    cat_model = None


if cat_model is not None:
    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(df=historical_df, target_column="target_48h")

    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(X_test)

    print("X_test shape:", X_test.shape)
    print("SHAP shape:", shap_values.shape)

    shap_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": abs(shap_values).mean(axis=0),
        }
    )

    shap_importance = shap_importance.sort_values(
        "importance", ascending=False
    )

    print("\nFeature Importance:")
    print(shap_importance)

    print("\nGenerating SHAP Summary Plot...")
    shap.summary_plot(shap_values, X_test)

    print("\nGenerating Dependence Plot — aqi_rolling_mean_1h")
    shap.dependence_plot("aqi_rolling_mean_1h", shap_values, X_test)

    print("\nGenerating Dependence Plot — aqi_lag_24")
    shap.dependence_plot("aqi_lag_24", shap_values, X_test)



try:
    model_path = hf_hub_download(repo_id=REPO_ID,filename="rf_model_72h.pkl",repo_type="model",)

    rf_model = joblib.load(model_path)
    print("Random Forest model loaded successfully from Hugging Face!")

except Exception as e:
    print(f"Random Forest model could not be loaded: {e}")
    rf_model = None


if rf_model is not None:
    X_train, X_test, y_train, y_test, encoder, train, test, df_processed, test_cities, test_origins = preprocess_data(df=historical_df, target_column="target_72h")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    print("X_test shape:", X_test.shape)
    print("SHAP shape:", shap_values.shape)

    shap_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": abs(shap_values).mean(axis=0),
        }
    )

    shap_importance = shap_importance.sort_values(
        "importance", ascending=False
    )

    print("\nFeature Importance:")
    print(shap_importance)

    print("\nGenerating SHAP Summary Plot...")
    shap.summary_plot(shap_values, X_test)