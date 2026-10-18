"""
data.py — Loads retail prices (Kalibrate) and market data (FRED).
"""

import io
import os
from datetime import datetime

import pandas as pd
import requests

from .cities import BY_KALIBRATE, CITIES

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
START_DATE = '2016-07-01'
FIRST_YEAR = 2016

# product -> (Kalibrate product path, local file prefix, first year we keep locally)
PRODUCTS = {
    'regular': ('Unleaded', 'Daily',   FIRST_YEAR),
    'premium': ('Premium',  'Premium', None),   # only the current/previous year: used for spreads
    'diesel' : ('Diesel',   'Diesel',  None),
}
KALIBRATE_URL = (
    'https://charting.kalibrate.com/WPPS/{product}/'
    'Retail%20(Incl.%20Tax)/DAILY/{year}/'
    '{product}_Retail%20(Incl.%20Tax)_DAILY_{year}.xlsx'
)

# Market series -> (column name, publication lag in days). A value dated d is treated as
# known on d + lag in training and backtesting as well as live, so the model never relies
# on data it wouldn't have had. FRED's EIA spot prices post 2–3 business days late; the
# Bank of Canada posts USD/CAD the same afternoon. FRED's weekly-updated DEXCAUS only
# fills in before the Bank of Canada series starts (2017). No API keys are needed.
FRED_SERIES = {
    'DCOILWTICO': ('wti_usd',  3),    # WTI crude, USD/barrel
    'DGASNYH'   : ('rbob_usd', 3),    # NY Harbor conventional gasoline, USD/gallon
    'DGASUSGULF': ('gulf_usd', 3),    # US Gulf Coast conventional gasoline, USD/gallon
    'DEXCAUS'   : ('cadusd_fred', 10),
}
BOC_FX = ('FXUSDCAD', 'cadusd', 1)    # CAD per USD, daily
FRED_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
BOC_CSV_URL  = 'https://www.bankofcanada.ca/valet/observations/{series}/csv?start_date={start}'
LAGS = {name: lag for name, lag in FRED_SERIES.values()} | {BOC_FX[1]: BOC_FX[2]}
LITRES_PER_GALLON = 3.78541


def _path(product, year):
    return os.path.join(DATA_DIR, f'{PRODUCTS[product][1]}{year}.xlsx')


def download_kalibrate(year, product='regular'):
    """Download one year's Kalibrate workbook into Data/. Returns True on success."""
    url = KALIBRATE_URL.format(product=PRODUCTS[product][0], year=year)
    print(f'  Downloading Kalibrate {product} {year}...')
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        pd.read_excel(io.BytesIO(r.content), header=None, nrows=5)  # reject HTML error pages
    except Exception as e:
        print(f'  WARNING: Kalibrate {product} {year} download failed: {e}')
        return False
    with open(_path(product, year), 'wb') as f:
        f.write(r.content)
    return True


def _read_year(product, year):
    """Return a DataFrame (dates x Kalibrate city names) for one workbook."""
    raw = pd.read_excel(_path(product, year), header=None)
    header_row = 0 if (product == 'regular' and year == 2016) else 2
    labels = raw.iloc[header_row, 1:]
    body   = raw.iloc[header_row + 1:, :]

    dates = pd.to_datetime([f'{d}/{year}' for d in labels], format='%m/%d/%Y', errors='coerce')
    names = body.iloc[:, 0].astype(str).str.strip().str.rstrip('*').str.upper()
    prices = body.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    prices.index, prices.columns = names, dates
    prices = prices.loc[prices.index.isin(BY_KALIBRATE.keys()), prices.columns.notna()]
    prices = prices[~prices.index.duplicated()]
    return prices.T.sort_index()


def load_prices(product='regular', current_year=None, first_year=None):
    """
    Calendar-daily DataFrame of retail prices (¢/L), one column per city slug.
    Kalibrate publishes weekdays only; weekends and holidays carry the last price forward.
    """
    current_year = current_year or datetime.now().year
    first_year = first_year or PRODUCTS[product][2] or current_year - 1
    frames = []
    for year in range(first_year, current_year + 1):
        if not os.path.exists(_path(product, year)):
            print(f'  WARNING: {os.path.basename(_path(product, year))} not found — skipping')
            continue
        frames.append(_read_year(product, year))

    wide = pd.concat(frames).sort_index()
    wide = wide[~wide.index.duplicated(keep='last')]
    wide.columns = [BY_KALIBRATE[c].slug for c in wide.columns]
    wide = wide[[s for s in CITIES if s in wide.columns]]

    wide = wide[(wide.index >= START_DATE) & (wide.index <= pd.Timestamp.today().normalize())]
    wide = wide.dropna(how='all')
    full = pd.date_range(wide.index.min(), wide.index.max(), freq='D')
    return wide.reindex(full).ffill(limit=5)


def load_gas_prices(current_year=None):
    return load_prices('regular', current_year)


def fetch_market():
    """Daily DataFrame of WTI, wholesale gasoline and CAD/USD as published (raw, with gaps)."""
    cols = {}
    for series, (name, _) in FRED_SERIES.items():
        r = requests.get(FRED_CSV_URL.format(series=series), timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ['date', 'value']
        df['date']  = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        s = df.set_index('date')['value']
        cols[name] = s[s.index >= START_DATE]
        print(f'  {series:<11}: up to {cols[name].dropna().index.max().date()}')

    series, name, _ = BOC_FX
    r = requests.get(BOC_CSV_URL.format(series=series, start=START_DATE), timeout=60)
    r.raise_for_status()
    body = r.text[r.text.index('"OBSERVATIONS"'):].split('\n', 1)[1]
    df = pd.read_csv(io.StringIO(body))
    s = pd.to_numeric(df[series], errors='coerce')
    s.index = pd.to_datetime(df['date'])
    cols[name] = s
    print(f'  {series:<11}: up to {s.dropna().index.max().date()}')
    return pd.DataFrame(cols)


def known_market(market):
    """Shift each series by its publication lag so a row dated d holds what was known on d."""
    out = {}
    for name, lag in LAGS.items():
        s = market[name].dropna()
        s.index = s.index + pd.Timedelta(days=lag)
        out[name] = s
    k = pd.DataFrame(out)
    k['cadusd'] = k['cadusd'].combine_first(k.pop('cadusd_fred')[k.index < k['cadusd'].first_valid_index()])
    return k


def align_market(market, index, max_staleness_days=10):
    """
    Put market data, as known on each day, on the price calendar. Weekends and holidays
    are forward-filled, but never by more than max_staleness_days beyond the expected lag.
    """
    k = known_market(market)
    m = k.reindex(k.index.union(index)).ffill(limit=max_staleness_days).reindex(index)
    m['wti_cad']    = m['wti_usd'] * m['cadusd']                                # CAD/barrel
    m['rbob_cad_l'] = m['rbob_usd'] * m['cadusd'] / LITRES_PER_GALLON * 100      # wholesale, ¢/L CAD
    m['gulf_cad_l'] = m['gulf_usd'] * m['cadusd'] / LITRES_PER_GALLON * 100
    m['crack_usd']  = m['rbob_usd'] * 42 - m['wti_usd']                          # USD/barrel
    return m


def market_staleness_days(market, as_of):
    """Days the oldest series is behind what its publication lag allows."""
    as_of = pd.Timestamp(as_of)
    return max((as_of - market[n].dropna().index.max()).days - LAGS[n]
               for n in ('wti_usd', 'rbob_usd', 'gulf_usd', 'cadusd'))
