# PumpCast — Canadian Gas Price Forecasts

Forecasts retail gasoline prices up to 4 weeks ahead for 77 Canadian cities (every city Kalibrate tracks), explains what's driving each forecast, and helps people decide when and where to fill up.

**Live app:** [gas-price-mtl.onrender.com](https://gas-price-mtl.onrender.com) · **Write-up:** [My gas price model looked 2× better than it was](docs/leakage-writeup.md)

---

## Features

**Forecasts**
- **4-week forecast** for regular, premium and diesel, with 80% ranges calibrated separately for calm, mid and volatile markets
- **Chance of a rise**, e.g. "31% chance the price is higher next Friday"
- **Why it's moving**: each forecast split into drivers (wholesale gasoline, station margin, crude and dollar, recent trend, scheduled tax changes)
- **What-if slider**: how the forecast responds if wholesale gasoline and crude jump ±10–20% today
- **Fill now / wait signal** when prices are expected to move more than 2 ¢/L within 3 days

**Tools**
- **Best day to fill up** in the next week (weekdays only), with a downloadable calendar reminder
- **Road trip planner**: the cheapest city to fill up along a route, on the day you'll pass through
- **Worth the drive?** Compares two cities, e.g. Gatineau vs Ottawa, including the cost of the detour
- **My fill-ups**: log what you pay (stored only in your browser) and see how you compare with the city average
- **Monthly budget**, **trip / commute cost**, and **EV vs gas** calculators

**Insights**
- **Map** of all 77 cities: today's price or the expected 7-day move
- **What's in the price**: crude, refining, station margin and each tax, plus a gauge of whether stations are charging more than usual
- **Price history** for the past year, with major events, city comparison and a weekday-pattern check
- **Track record**: every forecast is logged and scored against the actual price
- **Model card**: version, training window, range reliability, CRPS and known limitations

**Everywhere**
- English and French, mobile-friendly, accessible (skip link, chart text alternatives, AA contrast), installable as an app that works offline
- Public JSON API, CSV export, RSS feeds, an embeddable widget, and link previews for every city
- Optional push alerts (ntfy) and daily Bluesky / Mastodon posts
- Retrains daily on GitHub Actions; no API keys required

---

## Accuracy

Out-of-sample backtest: one model trained on data up to Dec 2023, then forecasting every day from Jan 2024 to Oct 2026 for all 77 cities. **Market data is used only as it would have been published at the time**, and **tax changes only once they were announced**. Error is mean absolute error in ¢/L. "No change" is the naive guess that the price stays where it is.

| Horizon | Model | "No change" | Improvement |
|---|---|---|---|
| 1 day | 0.94 | 0.96 | 2% |
| 7 days | **3.49** | 3.73 | 6% |
| 14 days | **5.35** | 5.58 | 4% |
| 28 days | **7.46** | 7.92 | 6% |
| Average, 1–28 days | **5.06** | 5.34 | 5% |

The model beats "no change" on average error in **all 77 cities**, and in every province except Yukon at 7 days. Probabilistic accuracy (CRPS) is 4.55 vs 4.86 ¢/L, 6% better. Ranges calibrated on 2024 covered 47% / 77% / 88% of 2025–26 outcomes at the 50% / 80% / 90% levels.

| City | 7 days | 7 days, "no change" | Avg 1–28 | Avg, "no change" |
|---|---|---|---|---|
| Montréal | 3.54 | 3.58 | 4.83 | 4.85 |
| Québec | 2.72 | 2.92 | 4.47 | 4.65 |
| Gatineau | 4.64 | 4.90 | 6.36 | 6.50 |
| Toronto | 3.67 | 3.83 | 5.03 | 5.31 |
| Vancouver | 5.27 | 5.34 | 6.92 | 7.04 |
| Calgary | 5.68 | 6.02 | 6.78 | 7.13 |
| Winnipeg | 4.01 | 4.31 | 5.63 | 6.14 |
| Halifax | 3.24 | 3.45 | 5.07 | 5.31 |

**About these numbers.** An earlier version reported larger gains (e.g. Montréal 4.56 vs 4.85). They came from a backtest that let the model see each day's wholesale price that same day, while FRED actually publishes it 2–3 business days later. The [write-up](docs/leakage-writeup.md) tells the story. With honest lags the edge is smaller but real: gasoline prices are close to a random walk.

**What was tried and measured** (`python -m gasprix.backtest --without <group>`):

| Change | Effect on average error |
|---|---|
| Regional wholesale (Gulf Coast for Western Canada) | small gain, kept |
| Price-cycle phase and province-wide moves | small gain, kept |
| Atlantic regulator weekly schedule | no measurable gain, removed |
| Calibrating ranges per volatility group | evens out coverage (calm / mid / volatile 78 / 75 / 78% instead of 75 / 74 / 81%), kept |
| Price levels as LightGBM inputs | made it overfit regimes, removed |

---

## How it works

**Data** (`gasprix/data.py`, `gasprix/cities.py`)

| Source | Series | Used for | Known after |
|---|---|---|---|
| [Kalibrate](https://charting.kalibrate.com) | Daily retail (regular, premium, diesel) | Target prices, 77 cities | next morning |
| [FRED](https://fred.stlouisfed.org/series/DGASNYH) | DGASNYH | NY Harbor wholesale gasoline (east) | 3 days |
| [FRED](https://fred.stlouisfed.org/series/DGASUSGULF) | DGASUSGULF | Gulf Coast wholesale gasoline (west) | 3 days |
| [FRED](https://fred.stlouisfed.org/series/DCOILWTICO) | DCOILWTICO | WTI crude | 3 days |
| [Bank of Canada](https://www.bankofcanada.ca/valet/docs) | FXUSDCAD | USD/CAD | 1 day |

Each market value dated *d* is treated as known only on *d* + lag, in training as well as live. Kalibrate reports weekdays only; weekends carry the last price forward.

**Taxes** (`gasprix/taxes.py`): federal excise (the 2026 suspension and scheduled 2027 return), federal and BC carbon charges (removed April 2025), provincial fuel taxes (including Alberta's oil-price-indexed relief, at 0 from Oct 2026), Montréal / Vancouver / Victoria transit levies, and GST/HST/QST. Prices are modelled **net of fixed taxes**, and forecasts add back only tax changes **announced** by the forecast date. Rates are approximate. A detector (`insights.tax_anomalies`) flags province-wide jumps the table doesn't explain; it found the Alberta change.

**Features** (`gasprix/features.py`): ex-tax price changes and deviations from recent averages; volatility; regional wholesale gasoline changes; the **retail-over-wholesale margin** relative to normal (the strongest signal); crude and USD/CAD changes; price-cycle phase (days since the last jump); province-wide moves; day of week, provincial holidays, province, city, and whether prices are regulated. Only changes and deviations go into the models.

**Model** (`gasprix/model.py`): for each horizon h = 1…28 days, one Ridge + LightGBM pair trained on **all cities pooled** predicts the ex-tax price change. Each horizon is shrunk toward "no change" by a weight chosen on a validation year. Ranges and probabilities come from out-of-sample error quantiles, scaled by each city's recent volatility and calibrated per volatility group.

**Pipeline** (`pipeline.py`, daily on GitHub Actions, about 5 minutes)

1. Refresh Kalibrate and market data. **Stop** if data is stale; skip any single city whose data has stopped updating.
2. Retrain without the most recent year and score on it. **Stop** if the error is more than 25% worse than the last run.
3. Retrain on all data and forecast every city, with ranges, probabilities, drivers and scenarios.
4. Check for unexplained tax-like jumps and range miscalibration; surface them as site warnings.
5. Log forecasts (`Data/predictions/<year>.csv`) and signals (`Data/signals.csv`).
6. Write `site/data/` and `site/feeds/`, send alerts and posts (if configured), and commit or publish.

If a check fails, nothing is published, the job fails (GitHub emails you), and yesterday's forecast stays live.

---

## Data policy

Retail prices are © Kalibrate, whose site posts no terms of use. To limit redistribution, **the site publishes only the past 12 months** of prices (`PUBLISHED_HISTORY_DAYS` in `pipeline.py`). The model still trains on the full history inside the pipeline. Raw workbooks are backed up monthly as **private** workflow artifacts (90 days), not public releases. Forecasts and code are yours to share.

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

| Task | Command |
|---|---|
| Refresh data, retrain, regenerate the site (about 5 min) | `python pipeline.py` |
| Serve the site at http://127.0.0.1:5000 | `python webapp/app.py` |
| Lint and unit tests | `ruff check . && pytest --ignore=tests/e2e` |
| Browser smoke test (after `playwright install chromium`) | `pytest tests/e2e` |
| Backtest, or measure a feature group | `python -m gasprix.backtest [--without cycles]` |
| Regenerate link-preview cards | `python scripts/make_share_cards.py` |
| Run the pipeline daily on your machine | `python scheduler.py` |

See [CONTRIBUTING.md](CONTRIBUTING.md), especially for reporting tax changes.

---

## Deployment & configuration

**Render** (current): `webapp/app.py` exposes `app` for Gunicorn and reads `PORT`; it only serves files. Dependencies are pinned in `requirements.txt`, so training and serving always match, and Dependabot proposes upgrades monthly.

All of the following are optional repository settings:

| Name | Type | Enables |
|---|---|---|
| `PUBLISH_BRANCH` | variable | Pushes the generated site to its own single-commit branch, overwritten daily, so `main` only receives small logs. Point Render (or any host) at that branch. |
| `DEPLOY_PAGES` | variable | GitHub Pages deploy (also set Pages → Source to GitHub Actions) |
| `SITE_URL` | variable | Absolute links in feeds, alerts, posts and freshness monitoring |
| `NTFY_TOPIC` | secret | Push alerts to `<topic>-<city>` when a city's signal changes |
| `BSKY_HANDLE`, `BSKY_APP_PASSWORD` | secrets | A daily Bluesky post (use an app password) |
| `MASTODON_URL`, `MASTODON_TOKEN` | secrets | A daily Mastodon post |

**Workflows:** `daily_pipeline.yml` (forecasts), `tests.yml` (lint, unit tests and browser smoke test on every PR), `monitor.yml` (fails if the live site's data is more than 36 h old), `backup.yml` (monthly private backup of raw data).

`FRED_API_KEY` is no longer used. **Revoke the key committed in earlier versions.**

---

## Repository layout

```
gasprix/          data, cities, taxes, features, model, insights, publish, backtest
pipeline.py       daily refresh → retrain → publish
site/             static website (works on any static host)
  data/  feeds/   written by the pipeline
  c/  share/      per-city link-preview pages and images
webapp/app.py     Flask server for site/ (Render)
tests/            unit tests; tests/e2e has the browser smoke test
scripts/          one-off generators (share cards)
docs/             write-ups
Data/             Kalibrate workbooks, prediction logs, signals, last-run metrics
Notebooks/        original Montréal-only exploration (archived)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Data & models | Python, pandas, scikit-learn (Ridge), LightGBM, holidays |
| Automation & CI | GitHub Actions, pytest, Playwright, ruff, Dependabot |
| Frontend | HTML, CSS, JavaScript, Chart.js, Leaflet + OpenStreetMap |
| Hosting | Render (Flask, Gunicorn) or GitHub Pages |
| Channels | RSS, ntfy, Bluesky, Mastodon |
