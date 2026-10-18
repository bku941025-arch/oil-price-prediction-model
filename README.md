# GasPrix — Canadian Gas Price Forecaster

Forecasts retail gasoline prices up to 4 weeks ahead for 77 Canadian cities (every city Kalibrate tracks), explains what's driving each forecast, and helps people decide when and where to fill up.

**Live app:** [gas-price-mtl.onrender.com](https://gas-price-mtl.onrender.com)

---

## Features

**Forecasts**
- **4-week forecast** with 80% price ranges calibrated on data the model never saw, for regular, premium and diesel
- **Chance of a rise**, e.g. "31% chance the price is higher next Friday"
- **Why it's moving**: each forecast split into drivers (wholesale gasoline, station margin, crude and dollar, recent trend, scheduled tax changes)
- **What-if slider**: how the forecast responds if wholesale gasoline and crude jump ±10–20% today
- **Fill now / wait signal** when prices are expected to move more than 2 ¢/L within 3 days

**Tools**
- **Best day to fill up** in the next week (weekdays only, since Kalibrate has no weekend data)
- **Worth the drive?** Compares two cities, e.g. Gatineau vs Ottawa, including the cost of the detour
- **Monthly budget**, **trip / commute cost**, and **EV vs gas** calculators

**Insights**
- **Map** of all 77 cities: today's price or the expected 7-day move
- **What's in the price**: crude, refining, station margin and each tax, plus a gauge of whether stations are charging more than usual
- **Price history** since 2016, annotated with major events, with city comparison and a weekday-pattern check
- **Track record**: every forecast is logged and scored against the actual price

**Everywhere**
- Bilingual (EN / FR), mobile-friendly, installable as an app (works offline with the last data)
- Public JSON API, RSS feeds and an embeddable widget
- Optional push alerts (ntfy) and daily Bluesky / Mastodon posts
- Retrains daily on GitHub Actions; no API keys required

---

## Accuracy

Out-of-sample backtest: one model trained on data up to Dec 2023, then forecasting every day from Jan 2024 to Oct 2026 for all 77 cities. **Market data is used only as it would have been published at the time**, and **tax changes only once they were announced**. Error is mean absolute error in ¢/L. "No change" is the naive guess that the price stays where it is.

| Horizon | Model | "No change" | Improvement |
|---|---|---|---|
| 1 day | 0.94 | 0.96 | 2% |
| 7 days | **3.49** | 3.73 | 6% |
| 14 days | **5.36** | 5.58 | 4% |
| 28 days | **7.46** | 7.92 | 6% |
| Average, 1–28 days | **5.07** | 5.34 | 5% |

The model beats "no change" on average error in **all 77 cities**. Its 80% ranges covered 77% of 2025–26 outcomes (calibrated on 2024).

| City | 7 days | 7 days, "no change" | Avg 1–28 | Avg, "no change" |
|---|---|---|---|---|
| Montréal | 3.53 | 3.58 | 4.83 | 4.85 |
| Québec | 2.71 | 2.92 | 4.48 | 4.65 |
| Gatineau | 4.61 | 4.90 | 6.38 | 6.50 |
| Toronto | 3.66 | 3.83 | 5.03 | 5.31 |
| Vancouver | 5.28 | 5.34 | 6.95 | 7.04 |
| Calgary | 5.70 | 6.02 | 6.81 | 7.13 |
| Winnipeg | 4.01 | 4.31 | 5.65 | 6.14 |
| Halifax | 3.19 | 3.45 | 5.08 | 5.31 |

**About these numbers.** An earlier version of this README reported larger gains (e.g. Montréal 4.56 vs 4.85). Those came from a backtest that let the model see each day's wholesale price that same day; in reality FRED publishes it 2–3 business days later. With honest publication lags the edge is smaller but real. Gasoline prices are close to a random walk, and 5% better than "no change" across 77 cities is a meaningful result. The original 2026 production model (recursive, with mis-rolled lags) was far *worse* than "no change": 7.44 vs 4.70 ¢/L on Montréal.

Reproduce with:

```bash
python -m gasprix.backtest
```

---

## How it works

**Data** (`gasprix/data.py`, `gasprix/cities.py`)

| Source | Series | Used for | Known after |
|---|---|---|---|
| [Kalibrate](https://charting.kalibrate.com) | Daily retail (regular, premium, diesel) | Target prices, 77 cities | next morning |
| [FRED](https://fred.stlouisfed.org/series/DGASNYH) | DGASNYH | NY Harbor wholesale gasoline | 3 days |
| [FRED](https://fred.stlouisfed.org/series/DCOILWTICO) | DCOILWTICO | WTI crude | 3 days |
| [Bank of Canada](https://www.bankofcanada.ca/valet/docs) | FXUSDCAD | USD/CAD | 1 day |

Each market value dated *d* is treated as known only on *d* + lag, in training as well as live, so the backtest can't use information the live pipeline wouldn't have. Kalibrate reports weekdays only; weekends carry the last price forward.

**Taxes** (`gasprix/taxes.py`): federal excise (including the 2026 suspension and the scheduled 2027 return), federal and BC carbon charges (removed April 2025), provincial fuel taxes, Montréal / Vancouver / Victoria transit levies, and GST/HST/QST. Prices are modelled **net of fixed taxes**, so a tax cut doesn't look like a market move. Forecasts then add back the tax changes **that had been announced** by the forecast date. Rates are approximate and cover only the large changes; see the sources in the file.

**Features** (`gasprix/features.py`): ex-tax price changes over 1–28 days and deviations from 7/28/90-day means; recent volatility; wholesale gasoline changes; the **retail-over-wholesale margin** relative to its normal (the strongest signal); crude and USD/CAD changes; day of week, provincial holidays, province, city, and whether prices are regulated (Atlantic Canada). Only changes and deviations go into the models. Price *levels* drift between regimes and made LightGBM overfit.

**Model** (`gasprix/model.py`): for each horizon h = 1…28 days, one Ridge + LightGBM pair is trained on **all cities pooled** (city and province as categorical features). It predicts the ex-tax price change, so forecasts never feed on their own predictions. Each horizon's prediction is shrunk toward "no change" by a weight chosen on a validation year. Ranges and probabilities come from out-of-sample error quantiles, scaled by each city's recent volatility.

**Insights** (`gasprix/insights.py`): driver contributions (exact for Ridge, SHAP for LightGBM, combined exactly as the forecast is), what-if scenarios, the price breakdown, the margin gauge, and the weekday analysis.

**Pipeline** (`pipeline.py`, daily on GitHub Actions, about 5 minutes)

1. Refresh Kalibrate and market data. **Stop** if data is stale; skip any single city whose data has stopped updating.
2. Retrain without the most recent year and score on it. **Stop** if the error is more than 25% worse than the last published run.
3. Retrain on all data and forecast every city, with ranges, probabilities, drivers and scenarios.
4. Log forecasts (`Data/predictions/<year>.csv`) and signals (`Data/signals.csv`) for later scoring.
5. Write `site/data/`, `site/feeds/`, send alerts and posts (if configured), and commit.

If a check fails, nothing is published, the job fails (GitHub emails you), and yesterday's forecast stays live. Run the workflow manually with "force publish" to override the accuracy gate.

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Refresh data, retrain and regenerate everything (about 5 minutes):

```bash
python pipeline.py
```

Serve the site at `http://127.0.0.1:5000` (set `PORT` to change it):

```bash
python webapp/app.py
```

Run the tests (no network needed):

```bash
pip install pytest && pytest
```

To run the pipeline on your own machine every day at 8:00 instead of on GitHub Actions:

```bash
python scheduler.py
```

---

## Deployment & configuration

**Render** (current): auto-deploys on each pipeline commit. `webapp/app.py` exposes `app` for Gunicorn and reads `PORT`; it only serves files.

**GitHub Pages** (optional, free, no cold starts): set Pages → Source to **GitHub Actions** in the repo settings, and add a repository variable `DEPLOY_PAGES` = `true`.

All of the following are optional repository secrets and variables:

| Name | Type | Enables |
|---|---|---|
| `NTFY_TOPIC` | secret | Push alerts to `<topic>-<city>` when a city's signal changes; the site shows how to subscribe |
| `BSKY_HANDLE`, `BSKY_APP_PASSWORD` | secrets | A daily Bluesky post (use an app password, not your account password) |
| `MASTODON_URL`, `MASTODON_TOKEN` | secrets | A daily Mastodon post |
| `SITE_URL` | variable | Absolute links in feeds, alerts and posts |
| `DEPLOY_PAGES` | variable | GitHub Pages deploy |

`FRED_API_KEY` is no longer used. **Revoke the key that was committed in earlier versions.**

Pull requests run the test suite (`.github/workflows/tests.yml`).

---

## Repository layout

```
gasprix/          data, cities, taxes, features, model, insights, publish, backtest
pipeline.py       daily refresh → retrain → publish
site/             static website (works on any static host)
  data/           written by the pipeline: index, per-city forecasts, history
  feeds/          RSS
webapp/app.py     Flask server for site/ (Render)
tests/            pytest suite
Data/             Kalibrate workbooks, prediction logs, signals, last-run metrics
Notebooks/        original Montréal-only exploration (archived; see Notebooks/README.md)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Data & models | Python, pandas, scikit-learn (Ridge), LightGBM, holidays |
| Automation & CI | GitHub Actions, pytest |
| Frontend | HTML, CSS, JavaScript, Chart.js, Leaflet + OpenStreetMap |
| Hosting | Render (Flask, Gunicorn) or GitHub Pages |
| Channels | RSS, ntfy, Bluesky, Mastodon |
