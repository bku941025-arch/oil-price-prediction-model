# Canada Regular Gas Price Forecasting

A project to forecast the trend of **regular gasoline prices in Canada** using a small set of accessible inputs and data that can be automatically pulled from fixed online sources. The goal is to help students and other frequent drivers plan budgets and choose better refueling times.

---

## Motivation

Gas prices can change significantly day-to-day and month-to-month. For students who commute or drive often, fuel is a major monthly expense. By forecasting upcoming regular gas price trends, we can:
- make more accurate monthly budgets,
- anticipate price increases/decreases,
- decide *when* to refuel based on expected movement.

To reduce noise and improve consistency, we limit the scope of data and model validity to **Canada**.

---

## What We Predict

- **Target:** Regular gasoline price (Canadian retail / average price, region-based when possible)
- **Output:** Forecasted price trend over a future time window (e.g., next days/weeks/months depending on data frequency)

---

## Key Factors Considered

Regular gasoline prices are influenced by many variables, including:
- international crude oil prices,
- regional / national temperature patterns and extreme weather,
- global supply & demand,
- exchange rates,
- taxes,
- geopolitical events.

Many difficult-to-quantify social and geopolitical effects are reflected in crude prices, so we focus on measurable signals that capture major drivers:

1. **WTI Crude Oil Price (benchmark)**
   - Used as a reference standard for crude oil pricing in North America
2. **CAD–USD Exchange Rate**
   - Because oil is generally priced in USD, exchange rate shifts affect Canadian fuel costs
3. **Regional Temperature / Extreme Weather**
   - Captures seasonal demand changes and disruptions
4. **Regional Fuel Taxes**
   - Taxes differ by Canadian region but are relatively stable long-term; we model them with a mergeable table

---

## Data Sources

### 1) WTI Crude Oil Prices (FRED)
- Dataset: `MCOILWTICO`
- Source: Federal Reserve Bank of St. Louis (FRED)
- Link: https://fred.stlouisfed.org/series/MCOILWTICO

### 2) Canadian Average Gasoline Prices (Statistics Canada)
- Source: Statistics Canada
- Notes: Used for Canada-wide or region-level average gas price series (depending on the table we select)

### 3) CAD to USD Exchange Rate (FRED)
- Dataset: `EXCAUS` (Canadian Dollars to U.S. Dollar Spot Exchange Rate)
- Source: Federal Reserve Bank of St. Louis (FRED)
- Link (FRED series page): https://fred.stlouisfed.org/series/EXCAUS

### 4) Regional Fuel Tax Table (Self-made)
- A manually created table with regional fuel taxes
- Designed to be stable and easy to merge with other datasets

### 5) Extreme Weather Events (Optional / Self-made)
- A low-frequency event table derived from available weather data
- Included as indicators (e.g., binary flags) because extreme events are rare but impactful

---

## Approach (High Level)

1. **Collect & update data**
   - Pull WTI and exchange rate series from FRED
   - Pull Canadian gas price series from Statistics Canada
   - Merge with tax and weather/event tables

2. **Clean & align time series**
   - Normalize units and timestamps (daily vs monthly, etc.)
   - Handle missing values and outliers

3. **Feature engineering**
   - Lag features (e.g., oil price lagged by 1–4 weeks)
   - Rolling averages / volatility measures
   - Seasonal indicators (month, week-of-year)
   - Weather and extreme-event flags

4. **Modeling**
   - Start with interpretable baselines (linear regression)
   - Extend to time-series models (ARIMAX / SARIMAX)
   - Consider ML models (Random Forest / XGBoost) if beneficial

5. **Evaluation**
   - Use time-based splits (train/validation/test)
   - Report metrics like MAE/RMSE and trend accuracy
   - Validate by region where data supports it

---

## Project Scope & Limitations

- **Scope:** Canada only (data + output validity)
- **Not included:** real-time station-level price prediction for every gas station
- **Important:** Forecast accuracy depends on data frequency, reporting lag, and sudden geopolitical shocks.

---