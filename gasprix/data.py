"""
data.py — Loads retail gas prices (Kalibrate) and market data (FRED).
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
START_DATE = '2016-07-01'
FIRST_YEAR = 2016

# slug -> (Kalibrate row name without the trailing '*', display name)
CITIES = {
    'montreal'     : ('MONTRÉAL',       'Montréal'),
    'quebec'       : ('QUÉBEC',         'Québec'),
    'gatineau'     : ('GATINEAU',       'Gatineau'),
    'sherbrooke'   : ('SHERBROOKE',     'Sherbrooke'),
    'trois-rivieres': ('TROIS RIVIÈRES', 'Trois-Rivières'),
}

KALIBRATE_URL = (
    'https://charting.kalibrate.com/WPPS/Unleaded/'
    'Retail%20(Incl.%20Tax)/DAILY/{year}/'
    'Unleaded_Retail%20(Incl.%20Tax)_DAILY_{year}.xlsx'
)

# FRED series -> column name. Fetched from the public CSV endpoint, so no API key is needed.
FRED_SERIES = {
    'DCOILWTICO': 'wti_usd',      # WTI crude, USD/barrel
    'DGASNYH'   : 'rbob_usd',     # NY Harbor conventional gasoline, USD/gallon
    'DEXCAUS'   : 'cadusd',       # CAD per USD
}
FRED_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
LITRES_PER_GALLON = 3.78541


def download_kalibrate(year):
    """Download one year's Kalibrate workbook into Data/. Returns True on success."""
    url      = KALIBRATE_URL.format(year=year)
    out_path = os.path.join(DATA_DIR, f'Daily{year}.xlsx')
    print(f'  Downloading Kalibrate {year}...')
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        pd.read_excel(io.BytesIO(r.content), header=None, nrows=5)  # reject HTML error pages
    except Exception as e:
        print(f'  WARNING: Kalibrate {year} download failed: {e}')
        return False
    with open(out_path, 'wb') as f:
        f.write(r.content)
    print(f'  Saved: {out_path}  ({len(r.content) / 1024:.1f} KB)')
    return True


def _read_year(year):
    """Return a DataFrame (dates x Kalibrate city names) for one workbook."""
    raw = pd.read_excel(os.path.join(DATA_DIR, f'Daily{year}.xlsx'), header=None)
    header_row = 0 if year == 2016 else 2
    labels = raw.iloc[header_row, 1:]
    body   = raw.iloc[header_row + 1:, :]

    dates = pd.to_datetime([f'{d}/{year}' for d in labels], format='%m/%d/%Y', errors='coerce')
    names = body.iloc[:, 0].astype(str).str.strip().str.rstrip('*').str.upper()
    prices = body.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    prices.index, prices.columns = names, dates
    prices = prices.loc[:, prices.columns.notna()]
    return prices.T


def load_gas_prices(current_year=None):
    """
    Return a calendar-daily DataFrame of retail prices (¢/L), one column per city slug.
    Kalibrate publishes weekdays only; weekends and holidays carry the last price forward.
    """
    current_year = current_year or datetime.now().year
    frames = []
    for year in range(FIRST_YEAR, current_year + 1):
        if not os.path.exists(os.path.join(DATA_DIR, f'Daily{year}.xlsx')):
            print(f'  WARNING: Daily{year}.xlsx not found — skipping')
            continue
        frames.append(_read_year(year))

    wide = pd.concat(frames).sort_index()
    wide = wide[~wide.index.duplicated(keep='last')]

    out = pd.DataFrame(index=wide.index)
    for slug, (row_name, _) in CITIES.items():
        if row_name in wide.columns:
            out[slug] = wide[row_name]
        else:
            print(f'  WARNING: {row_name} not found in Kalibrate files')

    out = out[(out.index >= START_DATE) & (out.index <= pd.Timestamp.today().normalize())]
    out = out.dropna(how='all')
    full = pd.date_range(out.index.min(), out.index.max(), freq='D')
    return out.reindex(full).ffill(limit=5)


def fetch_market():
    """Return a daily DataFrame of WTI, wholesale gasoline and CAD/USD (raw, with gaps)."""
    cols = {}
    for series, name in FRED_SERIES.items():
        r = requests.get(FRED_CSV_URL.format(series=series), timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ['date', 'value']
        df['date']  = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        s = df.set_index('date')['value']
        cols[name] = s[s.index >= START_DATE]
        print(f'  {series:<11}: up to {cols[name].dropna().index.max().date()}')
    return pd.DataFrame(cols)


def align_market(market, index, max_staleness_days=14):
    """
    Put market data on the gas price calendar. Weekends/holidays and FRED's publication
    lag are forward-filled, but never by more than max_staleness_days.
    """
    m = market.reindex(market.index.union(index)).ffill(limit=max_staleness_days).reindex(index)
    m['wti_cad']      = m['wti_usd'] * m['cadusd']                                  # CAD/barrel
    m['rbob_cad_l']   = m['rbob_usd'] * m['cadusd'] / LITRES_PER_GALLON * 100        # wholesale, ¢/L CAD
    m['crack_usd']    = m['rbob_usd'] * 42 - m['wti_usd']                            # USD/barrel
    return m


def market_staleness_days(market, as_of):
    """Days between as_of and the oldest 'latest observation' across market series."""
    last = min(market[c].dropna().index.max() for c in FRED_SERIES.values())
    return (pd.Timestamp(as_of) - last).days
