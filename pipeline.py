"""
pipeline.py — Runs the full data + retrain pipeline.
Called by GitHub Actions daily. Also callable manually:
    python pipeline.py
"""

import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_DIR     = './Data'
WEBAPP_DIR   = './webapp'
START_DATE   = '2016-07-01'
CITY         = 'MONTR'
CURRENT_YEAR = datetime.now().year
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

KALIBRATE_URL = (
    'https://charting.kalibrate.com/WPPS/Unleaded/'
    'Retail%20(Incl.%20Tax)/DAILY/{year}/'
    'Unleaded_Retail%20(Incl.%20Tax)_DAILY_{year}.xlsx'
)
# ──────────────────────────────────────────────────────────────────────────────


def download_kalibrate(year):
    url      = KALIBRATE_URL.format(year=year)
    out_path = os.path.join(DATA_DIR, f'Daily{year}.xlsx')
    print(f'  Downloading Kalibrate {year}...')
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            f.write(r.content)
        print(f'  Saved: {out_path}  ({len(r.content)/1024:.1f} KB)')
    except Exception as e:
        print(f'  ERROR downloading Kalibrate: {e}')


def parse_kalibrate():
    print('  Parsing Kalibrate files...')
    records, prev_date, prev_prices = [], None, []

    for year in range(2016, CURRENT_YEAR + 1):
        path = os.path.join(DATA_DIR, f'Daily{year}.xlsx')
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found — skipping')
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
                if not prev_prices:
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
    gas = gas[(gas.index >= START_DATE) & (gas.index <= pd.Timestamp.today().normalize())]
    print(f'  Gas prices: {len(gas):,} rows  {gas.index.min().date()} → {gas.index.max().date()}')
    return gas


def fetch_fred():
    print('  Fetching FRED data...')

    def get_series(series_id, col_name):
        url = (
            f'https://api.stlouisfed.org/fred/series/observations'
            f'?series_id={series_id}&observation_start={START_DATE}'
            f'&api_key={FRED_API_KEY}&file_type=json'
        )
        r   = requests.get(url, timeout=30)
        r.raise_for_status()
        obs = r.json()['observations']
        df  = pd.DataFrame(obs)[['date', 'value']]
        df['date']  = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('date')['value'].rename(col_name)

    wti    = get_series('DCOILWTICO', 'wti_usd')
    cadusd = get_series('DEXCAUS',    'cadusd')
    print(f'  WTI   : up to {wti.index.max().date()}')
    print(f'  CADUSD: up to {cadusd.index.max().date()}')
    return wti, cadusd


def build_features(gas, wti, cadusd):
    print('  Building features...')
    full_idx = pd.date_range(gas.index.min(), gas.index.max(), freq='D')
    wti_ff   = wti.reindex(full_idx).ffill(limit=7)
    fx_ff    = cadusd.reindex(full_idx).ffill(limit=7)

    df = gas.reindex(full_idx)
    df['wti_usd'] = wti_ff
    df['cadusd']  = fx_ff
    df['wti_cad'] = df['wti_usd'] / df['cadusd']
    df.dropna(subset=['gas_price', 'wti_usd', 'cadusd'], inplace=True)
    df = df[df.index <= pd.Timestamp.today().normalize()]

    for lag in [1, 2, 3, 5, 10, 21]:
        df[f'gas_lag_{lag}'] = df['gas_price'].shift(lag)
    for lag in [1, 2, 5]:
        df[f'wti_cad_lag_{lag}'] = df['wti_cad'].shift(lag)
    for window in [5, 10, 21]:
        df[f'gas_roll_mean_{window}'] = df['gas_price'].shift(1).rolling(window, min_periods=1).mean()
        df[f'gas_roll_std_{window}']  = df['gas_price'].shift(1).rolling(window, min_periods=1).std()
    for window in [5, 21]:
        df[f'wti_cad_roll_mean_{window}'] = df['wti_cad'].shift(1).rolling(window, min_periods=1).mean()

    df['wti_cad_chg_1'] = df['wti_cad'].diff(1).shift(1)
    df['wti_cad_chg_5'] = df['wti_cad'].diff(5).shift(1)
    df['gas_chg_1']     = df['gas_price'].diff(1).shift(1)
    df['cadusd_chg_1']  = df['cadusd'].diff(1).shift(1)
    df['day_of_week']   = df.index.dayofweek
    df['month']         = df.index.month
    df['target_price']  = df['gas_price'].shift(-1)
    df['target_change'] = df['target_price'] - df['gas_price']

    before = len(df)
    df.dropna(inplace=True)
    print(f'  Features: {len(df):,} rows  (dropped {before - len(df)} NaN rows)')

    df.to_csv(os.path.join(DATA_DIR, 'montreal_gas_ml_ready.csv'), index_label='date')
    print('  Saved: Data/montreal_gas_ml_ready.csv')
    return df


def retrain_models(df):
    print('  Retraining models...')
    EXCLUDE  = ['gas_price', 'wti_usd', 'cadusd', 'wti_cad', 'target_price', 'target_change']
    FEATURES = [c for c in df.columns if c not in EXCLUDE]
    TARGET   = 'target_price'

    df_train = df.dropna(subset=[TARGET])
    X, y     = df_train[FEATURES], df_train[TARGET]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge    = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    lgbm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, n_jobs=-1, random_state=42, verbose=-1
    )
    lgbm.fit(X, y)

    joblib.dump(ridge,  os.path.join(WEBAPP_DIR, 'model_ridge.pkl'))
    joblib.dump(lgbm,   os.path.join(WEBAPP_DIR, 'model_lgbm.pkl'))
    joblib.dump(scaler, os.path.join(WEBAPP_DIR, 'scaler.pkl'))
    print('  Models saved to webapp/')
    return len(df_train), df.index.max().date()


def run():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'\n{"=" * 55}')
    print(f'[{timestamp}] Pipeline started')
    print(f'{"=" * 55}')

    try:
        download_kalibrate(CURRENT_YEAR)
        gas          = parse_kalibrate()
        wti, cadusd  = fetch_fred()
        df           = build_features(gas, wti, cadusd)
        n_rows, data_end = retrain_models(df)

        log = {
            'last_retrain'  : timestamp,
            'training_rows' : n_rows,
            'data_end'      : str(data_end),
            'status'        : 'success'
        }
        with open(os.path.join(WEBAPP_DIR, 'retrain_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

        print(f'\n[{timestamp}] Pipeline complete')
        print(f'  Rows     : {n_rows:,}')
        print(f'  Data up to: {data_end}')
        print(f'{"=" * 55}\n')

    except Exception as e:
        print(f'\n[{timestamp}] Pipeline ERROR: {e}')
        with open(os.path.join(WEBAPP_DIR, 'retrain_log.json'), 'w') as f:
            json.dump({'last_retrain': timestamp, 'status': 'failed', 'error': str(e)}, f, indent=2)
        raise


if __name__ == '__main__':
    run()
