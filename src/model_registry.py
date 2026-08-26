import joblib
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings("ignore")


class AQIModelRegistry:

    def __init__(
        self,
        repo_id="flork-18115/AQI_prediciton_models"
    ):
        self.repo_id = repo_id

        self.models = {}

        self.model_paths = {
            "24h": {
                "model_type": "xgboost",
                "hf_path": "xgboost_24h.pkl"
            },

            "48h": {
                "model_type": "catboost",
                "hf_path": "catboost_48h.cbm"
            },

            "72h": {
                "model_type": "random_forest",
                "hf_path": "rf_model_72h.pkl"
            }
        }

    def load_model(self, horizon):

        if horizon not in self.model_paths:
            raise ValueError(
                f"Unknown horizon: {horizon}"
            )

        model_info = self.model_paths[horizon]

        print(
            f"Loading {horizon} "
            f"{model_info['model_type']} model..."
        )

        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=model_info["hf_path"],
            repo_type="model"
        )

        # Pickle / Joblib models
        if model_info["hf_path"].endswith(
            (".pkl", ".joblib")
        ):
            model = joblib.load(model_path)

        # CatBoost
        elif model_info["hf_path"].endswith(".cbm"):

            from catboost import CatBoostRegressor

            model = CatBoostRegressor()

            model.load_model(model_path)

        else:
            raise ValueError(
                f"Unsupported model format: "
                f"{model_info['hf_path']}"
            )

        self.models[horizon] = model

        print(
            f"Loaded {horizon} "
            f"{model_info['model_type']} model"
        )

        return model

    def get_model(self, horizon):

        if horizon not in self.models:
            return self.load_model(horizon)

        return self.models[horizon]

    def load_all_models(self):

        for horizon in self.model_paths:
            self.load_model(horizon)

        return self.models


if __name__ == "__main__":

    registry = AQIModelRegistry()

    models = registry.load_all_models()

    print("\nLoaded models:")

    for horizon, model in models.items():
        print(
            f"{horizon}: "
            f"{type(model).__name__}"
        )