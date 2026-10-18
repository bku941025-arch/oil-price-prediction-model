# GasPrix MTL — Québec Gas Price Forecaster

Forecasts retail gasoline prices up to 4 weeks ahead for Montréal, Québec City, Gatineau, Sherbrooke and Trois-Rivières, with calibrated price ranges, a best-day-to-fill-up pick, budget and trip-cost calculators, optional price alerts, and a public track record of how accurate past forecasts were.

**Live app:** [gas-price-mtl.onrender.com](https://gas-price-mtl.onrender.com)

---

## Features

- **4-week forecast** with 80% price ranges calibrated on data the model never saw
- **Fill now / wait signal** when prices are expected to move more than 2 ¢/L within 3 days
- **Best day to fill up** over the next 7 days, with savings for your tank size
- **Monthly budget** and **trip / commute cost** calculators
- **Track record**: every forecast is logged and scored against the actual price
- **5 Québec cities**, **bilingual** (EN / FR), works on mobile
- **Price alerts** by push notification via [ntfy](https://ntfy.sh) (optional)
- **Daily automated retraining** via GitHub Actions; no API keys required

---

## Accuracy

Out-of-sample backtest: trained on data up to Dec 2023, then forecasting every day from Jan 2024 to Oct 2026. Error is mean absolute error in ¢/L. “No change” is the naive guess that the price stays where it is.

| City | 1 day | 7 days | 7 days, “no change” | Avg 1–28 days | Avg, “no change” | 7-day direction right | 80% range coverage* |
|---|---|---|---|---|---|---|---|
| Montréal | 1.05 | **3.07** | 3.58 | **4.56** | 4.85 | 67% | 75% |
| Québec | 0.59 | **2.49** | 2.92 | **4.06** | 4.65 | 70% | 83% |
| Gatineau | 1.33 | **4.04** | 4.90 | **5.74** | 6.50 | 70% | 78% |
| Sherbrooke | 0.56 | **2.32** | 2.52 | **3.77** | 4.10 | 75% | 77% |
| Trois-Rivières | 0.51 | **2.40** | 2.57 | **3.90** | 4.33 | 74% | 72% |

\* Ranges calibrated on 2024, coverage measured on 2025 onward.

At 1 day ahead nothing beats “no change” (Kalibrate publishes weekdays only, so most next-day prices equal today's). The model detects this on its own and defers to persistence there. The previous version of the model (recursive forecast with mis-rolled lags) averaged **7.44 ¢/L** over 1–28 days on the same Montréal test, versus 4.70 for “no change”.

Reproduce with:

```bash
python -m gasprix.backtest
```

---

## How it works

**Data** (`gasprix/data.py`)

| Source | Series | Used for |
|---|---|---|
| [Kalibrate](https://charting.kalibrate.com) | Daily retail, taxes incl. | Target prices for each city (¢/L) |
| [FRED](https://fred.stlouisfed.org/series/DGASNYH) | DGASNYH | NY Harbor wholesale gasoline: retail follows it with a lag |
| [FRED](https://fred.stlouisfed.org/series/DCOILWTICO) | DCOILWTICO | WTI crude |
| [FRED](https://fred.stlouisfed.org/series/DEXCAUS) | DEXCAUS | CAD per USD |

FRED is read from its public CSV endpoint, so no API key is needed. Kalibrate reports weekdays only, so weekends and holidays carry the last price forward.

**Features** (`gasprix/features.py`): today's price; price changes over 1–28 days; deviation from 7/28/90-day means; recent volatility; wholesale gasoline in ¢/L CAD and its changes; the **retail–wholesale margin** and how far it sits from normal (the strongest mean-reverting signal); crude, crack spread and CAD moves; day of week, month, and whether a Québec holiday is within a week.

**Model** (`gasprix/model.py`): for each horizon h = 1…28 days, a Ridge + LightGBM pair predicts the *change* in price, so forecasts never feed on their own predictions. Each horizon's prediction is shrunk toward “no change” by a weight chosen on a validation year; where the models can't beat persistence, the weight goes to 0. Price ranges are split-conformal: quantiles of out-of-sample errors, scaled by recent volatility.

**Pipeline** (`pipeline.py`, daily on GitHub Actions)

1. Refresh Kalibrate and FRED data. **Stop** if either is stale (more than 6 / 14 days old).
2. Retrain without the most recent year and score on it. **Stop** if the error is more than 25% worse than the last published run.
3. Retrain on all data and forecast 28 days for each city.
4. Append the forecast to `Data/predictions_log.csv` (scored later against actual prices).
5. Write `site/data/<city>.json`, send alerts if a signal changed, and commit.

If any check fails, nothing is written, the job fails (GitHub emails you), and yesterday's forecast stays live. Run the workflow manually with “force publish” to override the accuracy gate.

**Website** (`site/`): plain HTML, CSS and JS reading the JSON files. It can be served by `webapp/app.py` (Render) or any static host.

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Refresh data, retrain and regenerate the forecasts (about 3 minutes):

```bash
python pipeline.py
```

Serve the site at `http://127.0.0.1:5000` (set `PORT` to change it):

```bash
python webapp/app.py
```

To run the pipeline on your own machine every day at 8:00 instead of on GitHub Actions:

```bash
python scheduler.py
```

---

## Deployment

**Render** (current): auto-deploys on each commit from the pipeline. `webapp/app.py` still exposes `app` for Gunicorn and reads `PORT`, so existing Render settings keep working; it now only serves files.

**GitHub Pages** (optional, free, no cold starts): in the repo settings, set Pages → Source to **GitHub Actions**, then add a repository variable `DEPLOY_PAGES` = `true`. The daily workflow will then publish `site/` to Pages after each run.

**Price alerts** (optional): add a repository secret `NTFY_TOPIC` with a hard-to-guess name (e.g. `gasprix-7f3k9`). Alerts go to `<topic>-<city>` (e.g. `gasprix-7f3k9-montreal`) when a city's signal changes, and the site shows subscribe instructions. To make alert links open your site, set the repository variable `SITE_URL`.

The `FRED_API_KEY` secret is no longer used.

---

## Repository layout

```
gasprix/        data loading, features, model, backtest
pipeline.py     daily refresh → retrain → publish
site/           static website; site/data/ is written by the pipeline
webapp/app.py   Flask server for site/ (Render)
Data/           Kalibrate workbooks, prediction log, last-run metrics
Notebooks/      original exploration and model selection (2016–2026 data build,
                feature engineering, model comparison); the production code in
                gasprix/ supersedes their feature set
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Data & models | Python, pandas, scikit-learn (Ridge), LightGBM, holidays |
| Automation | GitHub Actions |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Hosting | Render (Flask, Gunicorn) or GitHub Pages |
| Alerts | ntfy |
