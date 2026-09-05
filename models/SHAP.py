import os

import joblib
import pandas as pd
import shap

from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download

from models.preprocessing import preprocess_data
from models.supabase_data import get_historical_data
from models.feature_engineering import create_features


REPO_ID = "flork-18115/AQI_prediciton_models"



def load_shap_model(target_column):

    if target_column == "target_24h":

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="xgboost_24h.pkl",
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )

        model = joblib.load(model_path)

        return model

    elif target_column == "target_48h":

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="catboost_48h.cbm",
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )

        model = CatBoostRegressor()
        model.load_model(model_path)

        return model

    elif target_column == "target_72h":

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="rf_model_72h.pkl",
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )

        model = joblib.load(model_path)

        return model

    else:
        raise ValueError(
            f"Unsupported target column: {target_column}"
        )



# PREPARE DATA


def prepare_shap_data(historical_df, target_column):

    # Same feature engineering used during training
    featured_df = create_features(historical_df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        train,
        test,
        df_processed,
        test_cities,
        test_origins,
    ) = preprocess_data(
        df=featured_df,
        target_column=target_column,
    )

    # Keep only ML features
    X_test = X_test.select_dtypes(
        include=["number", "bool"]
    )

    X_test = X_test.astype(float)

    return X_test, test_cities

def calculate_shap_results(historical_df, target_column):


    model = load_shap_model(target_column)

    X_test, test_cities = prepare_shap_data(
        historical_df,
        target_column,
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    shap_importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": abs(shap_values).mean(axis=0),
        }
    )

    shap_importance = (
        shap_importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        X_test,
        test_cities,
        shap_values,
        shap_importance,
    )