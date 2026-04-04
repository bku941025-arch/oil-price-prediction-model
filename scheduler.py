"""
scheduler.py — Full automated pipeline for Montréal Gas Price model
--------------------------------------------------------------------
Run this once in a terminal:
    python scheduler.py

Every day at 08:00 AM it will automatically:
    1. Download the latest 2026 Kalibrate data from the public URL
    2. Fetch the latest WTI crude + CAD/USD rates from FRED
    3. Rebuild the full feature matrix (montreal_gas_ml_ready.csv)
    4. Retrain Ridge + LightGBM on all available data
    5. Save fresh .pkl files + retrain_log.json

Historical files (2016–2025) in /Data are never touched.
Leave the terminal open to keep the scheduler running.
Press Ctrl+C to stop.
"""

import os
import json
import time
import requests
import schedule
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
RETRAIN_TIME   = '08:00'             # Daily run time (24h format)
DATA_DIR       = './Data'             # Folder containing historical .xlsx files
WEBAPP_DIR     = './webapp'           # Folder where .pkl files and retrain_log.json are saved
FRED_API_KEY   = '30e324966282328b1d03b97b95ef9b1f' # Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
START_DATE     = '2016-07-01'
CITY           = 'MONTR'
CURRENT_YEAR   = datetime.now().year

# Kalibrate URL pattern — year is injected automatically
KALIBRATE_URL  = (
    'https://charting.kalibrate.com/WPPS/Unleaded/'
    'Retail%20(Incl.%20Tax)/DAILY/{year}/'
    'Unleaded_Retail%20(Incl.%20Tax)_DAILY_{year}.xlsx'
)
# ──────────────────────────────────────────────────────────────────────────────


# ── STEP 1: Download latest Kalibrate file ────────────────────────────────────
def download_kalibrate(year):
    """Download the Kalibrate Excel file for the given year into /Data."""
    url      = KALIBRATE_URL.format(year=year)
    out_path = os.path.join(DATA_DIR, f'Daily{year}.xlsx')

    print(f'  Downloading Kalibrate {year} from Kalibrate...')
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(out_path, 'wb') as f:
            f.write(response.content)
        print(f'  Saved: {out_path}  ({len(response.content) / 1024:.1f} KB)')
        return True
    except Exception as e:
        print(f'  ERROR downloading Kalibrate {year}: {e}')
        return False


# ── STEP 2: Parse all Kalibrate files → daily gas prices ─────────────────────
def parse_kalibrate():
    """
    Read all Daily{year}.xlsx files from /Data and return a
    daily DataFrame of Montréal gas prices.
    Replicates the logic from Notebook 1.
    """
    print('  Parsing Kalibrate files...')
    records   = []
    prev_date = None
    prev_prices = []

    years = range(2016, CURRENT_YEAR + 1)

    for year in years:
        path = os.path.join(DATA_DIR, f'Daily{year}.xlsx')
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found — skipping {year}')
            continue

        raw = pd.read_excel(path, header=None)

        if year == 2016:
            date_row = raw.iloc[0, 1:].values
            df_raw   = raw.iloc[1:, :].set_index(0)
        else:
            date_row = raw.iloc[2, 1:].values
            df_raw   = raw.iloc[3:, :].set_index(0)

        montreal_row = df_raw[df_raw.index.str.contains(CITY, na=False, case=False)]
        prices       = montreal_row.iloc[0].values

        for d, p in zip(date_row, prices):
            try:
                current_date = pd.to_datetime(f'{d}/{year}', format='%m/%d/%Y')
            except Exception:
                if prev_date is None:
                    continue
                current_date = prev_date + pd.Timedelta(days=1)

            if pd.isna(p):
                if len(prev_prices) == 0:
                    continue
                current_price = float(np.mean(prev_prices))
            else:
                current_price = float(p)

            records.append({'date': current_date, 'gas_price': current_price})
            prev_date = current_date
            prev_prices.append(current_price)
            if len(prev_prices) > 10:
                prev_prices.pop(0)

    gas = pd.DataFrame(records).drop_duplicates('date').set_index('date').sort_index()
    gas = gas[gas.index >= START_DATE]
    print(f'  Gas prices: {len(gas):,} rows  {gas.index.min().date()} → {gas.index.max().date()}')
    return gas


# ── STEP 3: Fetch FRED data (WTI + CAD/USD) ───────────────────────────────────
def fetch_fred():
    """
    Fetch WTI crude oil and CAD/USD exchange rate from FRED API.
    Replicates the logic from Notebook 1.
    """
    print('  Fetching FRED data (WTI + CAD/USD)...')

    def get_series(series_id):
        url = (
            f'https://api.stlouisfed.org/fred/series/observations'
            f'?series_id={series_id}'
            f'&observation_start={START_DATE}'
            f'&api_key={FRED_API_KEY}'
            f'&file_type=json'
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        obs = r.json()['observations']
        df  = pd.DataFrame(obs)[['date', 'value']]
        df['date']  = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('date')['value']

    wti   = get_series('DCOILWTICO').rename('wti_usd')
    cadusd = get_series('DEXCAUS').rename('cadusd')

    print(f'  WTI   : {len(wti):,} rows up to {wti.index.max().date()}')
    print(f'  CADUSD: {len(cadusd):,} rows up to {cadusd.index.max().date()}')
    return wti, cadusd


# ── STEP 4: Merge and build features ─────────────────────────────────────────
def build_features(gas, wti, cadusd):
    """
    Merge gas, WTI, CADUSD and build the full feature matrix.
    Replicates the logic from Notebooks 1 and 2.
    """
    print('  Building feature matrix...')

    # Build full daily index and forward-fill FRED gaps
    full_idx = pd.date_range(gas.index.min(), gas.index.max(), freq='D')
    wti_ff   = wti.reindex(full_idx).ffill(limit=5)
    fx_ff    = cadusd.reindex(full_idx).ffill(limit=5)

    # Merge
    df = gas.copy()
    df = df.reindex(full_idx)
    df['wti_usd'] = wti_ff
    df['cadusd']  = fx_ff
    df['wti_cad'] = df['wti_usd'] / df['cadusd']
    df.dropna(subset=['gas_price', 'wti_usd', 'cadusd'], inplace=True)

    # ── Lag features ──
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f'gas_lag_{lag}'] = df['gas_price'].shift(lag)
    for lag in [1, 2, 5]:
        df[f'wti_cad_lag_{lag}'] = df['wti_cad'].shift(lag)

    # ── Rolling statistics ──
    for window in [5, 10, 21]:
        df[f'gas_roll_mean_{window}'] = df['gas_price'].shift(1).rolling(window, min_periods=1).mean()
        df[f'gas_roll_std_{window}']  = df['gas_price'].shift(1).rolling(window, min_periods=1).std()
    for window in [5, 21]:
        df[f'wti_cad_roll_mean_{window}'] = df['wti_cad'].shift(1).rolling(window, min_periods=1).mean()

    # ── Momentum features ──
    df['wti_cad_chg_1'] = df['wti_cad'].diff(1).shift(1)
    df['wti_cad_chg_5'] = df['wti_cad'].diff(5).shift(1)
    df['gas_chg_1']     = df['gas_price'].diff(1).shift(1)
    df['cadusd_chg_1']  = df['cadusd'].diff(1).shift(1)

    # ── Calendar features ──
    df['day_of_week'] = df.index.dayofweek
    df['month']       = df.index.month

    # ── Target variables ──
    df['target_price']  = df['gas_price'].shift(-1)
    df['target_change'] = df['target_price'] - df['gas_price']

    # Drop NaN rows from lag/target creation
    before = len(df)
    df.dropna(inplace=True)
    print(f'  Features built: {len(df):,} rows  (dropped {before - len(df)} NaN rows)')

    # Save
    df.to_csv(os.path.join(DATA_DIR, 'montreal_gas_ml_ready.csv'), index_label='date')
    print('  Saved: ./Data/montreal_gas_ml_ready.csv')
    return df


# ── STEP 5: Retrain models ────────────────────────────────────────────────────
def retrain_models(df):
    """Retrain Ridge and LightGBM on all available data and save .pkl files."""
    print('  Retraining models...')

    EXCLUDE  = ['gas_price', 'wti_usd', 'cadusd', 'wti_cad', 'target_price', 'target_change']
    FEATURES = [c for c in df.columns if c not in EXCLUDE]
    TARGET   = 'target_price'

    df_train = df.dropna(subset=[TARGET])
    X = df_train[FEATURES]
    y = df_train[TARGET]

    # Ridge
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge    = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, n_jobs=-1, random_state=42, verbose=-1
    )
    lgbm.fit(X, y)

    # Save
    joblib.dump(ridge,  os.path.join(WEBAPP_DIR, 'model_ridge.pkl'))
    joblib.dump(lgbm,   os.path.join(WEBAPP_DIR, 'model_lgbm.pkl'))
    joblib.dump(scaler, os.path.join(WEBAPP_DIR, 'scaler.pkl'))

    print(f'  Models saved: ./webapp/model_ridge.pkl, model_lgbm.pkl, scaler.pkl')
    return len(df_train), df.index.max().date()


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
def run_pipeline():
    """
    Run the full pipeline:
    1. Download latest Kalibrate 2026 file
    2. Parse all Kalibrate files
    3. Fetch FRED data
    4. Build feature matrix
    5. Retrain models
    6. Log results
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'\n{"=" * 60}')
    print(f'[{timestamp}] Pipeline started')
    print(f'{"=" * 60}')

    try:
        # Step 1 — Download latest Kalibrate data
        download_kalibrate(CURRENT_YEAR)

        # Step 2 — Parse all Kalibrate files
        gas = parse_kalibrate()

        # Step 3 — Fetch FRED data
        wti, cadusd = fetch_fred()

        # Step 4 — Build feature matrix
        df = build_features(gas, wti, cadusd)

        # Step 5 — Retrain models
        n_rows, data_end = retrain_models(df)

        # Step 6 — Log
        log = {
            'last_retrain'  : timestamp,
            'training_rows' : n_rows,
            'data_end'      : str(data_end),
            'status'        : 'success'
        }
        with open(os.path.join(WEBAPP_DIR, 'retrain_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

        print(f'\n[{timestamp}] Pipeline complete ✓')
        print(f'  Training rows : {n_rows:,}')
        print(f'  Data up to    : {data_end}')
        print(f'{"=" * 60}\n')

    except Exception as e:
        print(f'\n[{timestamp}] Pipeline ERROR: {e}')
        # Log the failure so you can inspect it
        with open(os.path.join(WEBAPP_DIR, 'retrain_log.json'), 'w') as f:
            json.dump({
                'last_retrain': timestamp,
                'status'      : 'failed',
                'error'       : str(e)
            }, f, indent=2)
        print(f'{"=" * 60}\n')


# ── SCHEDULER SETUP ───────────────────────────────────────────────────────────
print(f'{"=" * 60}')
print('  Montréal Gas Price — Automated Pipeline Scheduler')
print(f'{"=" * 60}')
print(f'  Schedule : Daily at {RETRAIN_TIME}')
print(f'  Data dir : {DATA_DIR}')
print(f'  FRED key : {"set ✓" if FRED_API_KEY != "YOUR_FRED_API_KEY" else "NOT SET ✗ — update FRED_API_KEY"}')
print(f'{"=" * 60}\n')

schedule.every().day.at(RETRAIN_TIME).do(run_pipeline)

# Run once immediately on startup
run_pipeline()

# Keep running
while True:
    schedule.run_pending()
    time.sleep(60)
