# AQI Prediction System

### Multi-Horizon AQI Forecasting for Four Pakistani Cities

An end-to-end machine learning system for predicting **Air Quality Index (AQI) 24, 48, and 72 hours ahead** for:

* Islamabad
* Lahore
* Peshawar
* Rawalpindi

The system includes automated data collection, feature engineering, model inference, monitoring, retraining, and an interactive Streamlit dashboard. Consult the report for understanding of thr project in greater detail. 

## Dashboard

### AQI Forecast Dashboard

<img width="892" height="416" alt="Screenshot 2026-09-06 024715" src="https://github.com/user-attachments/assets/50c390cc-964e-4e68-a771-5442e6a98e6a" />

### Hourly Backfill Status

<img width="375" height="112" alt="Screenshot 2026-09-06 025142" src="https://github.com/user-attachments/assets/7043c25b-d93a-417d-9eaa-c162b6d5e386" />


## Production Models

| Forecast | Model         |       MAE |      RMSE |        R² |
| -------- | ------------- | --------: | --------: | --------: |
| 24 hours | XGBoost       | **12.18** |     17.53 | **0.759** |
| 48 hours | CatBoost      | **15.84** | **20.71** | **0.663** |
| 72 hours | Random Forest | **16.07** |     21.27 | **0.648** |

## Technology Stack

### Programming

* Python 3.11.9

### Data & APIs

* Pandas
* NumPy
* Open-Meteo Air Quality API
* Open-Meteo Historical Weather API


### Machine Learning

* Scikit-learn
* XGBoost
* CatBoost
* TensorFlow / Keras
* LSTM

### MLOps & Infrastructure

* Feast — Feature Store
* Supabase PostgreSQL — Database & Online Store
* Hugging Face Hub — Model Storage
* GitHub Actions — Automation

### Model Monitoring

<img width="960" height="405" alt="Screenshot 2026-09-06 025244" src="https://github.com/user-attachments/assets/ee9a77e6-1294-4d7c-9407-6c91e79d3e4f" />


### Dashboard & Explainability

* Streamlit
* SHAP

### SHAP Explainability

<img width="465" height="405" alt="Screenshot 2026-09-06 025104" src="https://github.com/user-attachments/assets/e3a962ce-3b0b-472e-8c71-31e656d8042d" />

## System Architecture (Simplified)

```text
Open-Meteo APIs
      │
      ▼
Data Collection
      │
      ▼
Feature Engineering
      │
      ▼
Feast
      │
      ▼
Supabase PostgreSQL
      │
      ▼
Production Models
      │
      ├── XGBoost → 24h
      ├── CatBoost → 48h
      └── Random Forest → 72h
      │
      ▼
Predictions
      │
      ▼
Monitoring
      │
      ▼
Retraining
      │
      ▼
Hugging Face
```

## Project Structure (Local Project) 

```text
AQI_prediction_system/
├── .devcontainer/
├── .github/
│   └── workflows/
│       ├── daily_monitoring.yml
│       ├── hourly_pipeline.yml
│       └── weekly_retraining.yml
├── .venv/
├── api_scripts/
│   ├── api_call_air_data.py
│   ├── api_call_weather.py
│   └── merge_script.py
├── data/
├── EDA/
│   └── eda.ipynb
├── feast_st/
│   ├── data/
│   ├── feature_def.py
│   ├── feature_store.py
│   ├── feature_store.yaml
│   └── feature_store.yaml.template
├── models/
│   ├── retrained/
│   ├── feature_engineering.py
│   ├── LSTM_model.py
│   ├── preprocessing.py
│   ├── SHAP.py
│   ├── supabase_data.py
│   ├── train_ML_models.py
│   ├── train_models.ipynb
│   └── weekly_retraining.py
├── scripts/
│   └── create_feast_config.py
├── src/
│   ├── api/
│   ├── api_fetch.py
│   ├── daily_monitoring.py
│   ├── feast_update.py
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── model_registry.py
│   ├── prediction_store.py
│   └── run_pipeline.py
├── streamlit_app/
│   ├── app.py
│   ├── streamlit_pred.py
│   └── streamlit_shap.py
├── .env
├── .gitignore
├── README.md
├── Report.docx
└── requirements.txt
```

> `.venv/` and `.env` are excluded from version control.

## Features

* 24h, 48h and 72h AQI forecasting
* Forecasting for four Pakistani cities
* AQI and weather feature engineering
* Lag and rolling features
* Temporal and cyclical features
* XGBoost, CatBoost, Random Forest and LSTM
* Feast feature serving
* Supabase PostgreSQL integration
* Hugging Face model storage
* Automated prediction pipeline
* Daily model monitoring
* Weekly conditional retraining
* SHAP explainability
* Interactive Streamlit dashboard
* GitHub Actions automation

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/FatimaMuhammad90/City_AQI-predictor.git
cd City_AQI-predictor
```

### 2. Create the virtual environment

The project uses:

```text
Python 3.11.9
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
HF_TOKEN=your_huggingface_token
```

Do not commit `.env` or expose API keys and tokens.

## Run the Dashboard

```bash
streamlit run streamlit_app/app.py
```

## Run the Prediction Pipeline

```bash
python -m src.run_pipeline
```

## Run Daily Monitoring

```bash
python -m src.daily_monitoring
```

## Run Weekly Retraining

```bash
python -m models.weekly_retraining
```

## Project Links

* **Live Dashboard:** https://cityaqi-predictor-ht2syb2qpgdjx8fynsb5ex.streamlit.app/
* **GitHub:** https://github.com/FatimaMuhammad90/City_AQI-predictor
* **Hugging Face Models:** https://huggingface.co/flork-18115/AQI_prediciton_models

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Fatima Muhummad Ali

See the [LICENSE](LICENSE) file for the full license text.

## Author

**Fatima Muhummad Ali**

10P Shine Internee Cohort 9 — Data Sciences
