# GasPrix MTL — Montreal Gas Price Predictor

A machine learning web application that forecasts retail gasoline prices in Montreal up to 4 weeks ahead and helps users plan their monthly gas budget.

**Live app:** [gas-price-mtl.onrender.com](https://gas-price-mtl.onrender.com)

---

## Overview

GasPrix MTL uses Ridge Regression and LightGBM trained on 10 years of daily Montreal retail gas prices (2016–2026) alongside WTI crude oil and CAD/USD exchange rate data. The app provides:

- **4-week price range forecast** with widening confidence bands
- **Best day to fill up** recommendation for the coming week
- **Monthly budget estimator** based on last month's gas spend
- **Bilingual interface** (English / French)
- **Daily automated retraining** via GitHub Actions

---

## Data Sources

| Source | Series | Description |
|---|---|---|
| [Kalibrate](https://charting.kalibrate.com) | — | Montreal daily retail gas prices (¢/L, taxes included) |
| [FRED](https://fred.stlouisfed.org/series/DCOILWTICO) | DCOILWTICO | WTI crude oil spot price (USD/barrel) |
| [FRED](https://fred.stlouisfed.org/series/DEXCAUS) | DEXCAUS | CAD/USD exchange rate |

---

## Models

| Model | MAE | RMSE | Directional Accuracy |
|---|---|---|---|
| Persistence baseline | 1.924 ¢/L | 2.741 ¢/L | — |
| **Ridge Regression** | **1.760 ¢/L** | **2.438 ¢/L** | 60.8% |
| XGBoost | 1.864 ¢/L | 2.668 ¢/L | 61.4% |
| LightGBM | 1.895 ¢/L | 2.782 ¢/L | **63.8%** |

- **Ridge** wins on MAE (most accurate price level)
- **LightGBM** wins on directional accuracy (best up/down signal)
- The ensemble (average of both) powers the forecast

**Train / test split:** Aug 2016 – Dec 2023 (train) | Jan 2024 – present (test)

---

## Features

The model uses 23 engineered features:

- **Gas price lags** — `gas_lag_1/2/3/5/10/21`
- **WTI lags** — `wti_cad_lag_1/2/5`
- **Rolling statistics** — 5/10/21-day rolling mean and std
- **Momentum** — 1-day and 5-day WTI and gas price changes
- **Calendar** — day of week, month

---

## Running Locally

### Prerequisites

```bash
pip install flask pandas numpy scikit-learn lightgbm xgboost joblib requests openpyxl gunicorn
```

### 1. Run the notebooks in order

```
Notebooks/montreal_gas_daily_build.ipynb
Notebooks/montreal_gas_feature_engineering.ipynb
Notebooks/montreal_gas_modelling.ipynb
Notebooks/montreal_gas_inference.ipynb
```

### 3. Start the web app

```bash
cd webapp
python app.py
```

Then open `http://127.0.0.1:5000`

### 4. (Optional) Run the local scheduler

```bash
python scheduler.py
```

Retrains the model daily at 8:00 AM automatically.

---

## Deployment

The app is deployed on [Render](https://render.com) and retrains automatically via GitHub Actions every day at 8AM EST.

**GitHub Actions pipeline** (`.github/workflows/daily_pipeline.yml`):
1. Downloads the latest Kalibrate gas price data
2. Fetches WTI and CAD/USD from FRED
3. Rebuilds the feature matrix
4. Retrains Ridge + LightGBM on all available data
5. Commits updated files back to the repo
6. Render auto-deploys on commit

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, pandas, requests |
| ML models | scikit-learn (Ridge), LightGBM, XGBoost |
| Web backend | Flask, Gunicorn |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Automation | GitHub Actions |
| Hosting | Render |
