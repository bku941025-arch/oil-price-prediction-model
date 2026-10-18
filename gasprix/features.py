"""
features.py — Feature engineering shared by training, backtesting and forecasting.

Every feature for origin date t uses only information known at the end of day t:
that day's retail price, and market data as published by then (see data.known_market).

Prices are modelled net of fixed per-litre taxes ("ex-tax"), so tax step changes such
as the 2025 carbon charge removal don't look like market moves. Models predict the
change in the ex-tax price; scheduled tax changes are added back when forecasting.
"""

import holidays
import numpy as np
import pandas as pd

from . import taxes
from .cities import CITIES, PROVINCE_NAMES, REGULATED

HORIZONS = list(range(1, 29))   # calendar days ahead
PROVINCES = sorted(PROVINCE_NAMES)

# Feature groups that can be switched off to measure their value in the backtest.
USE_REGIONAL_WHOLESALE = True   # western provinces use Gulf Coast gasoline instead of NY Harbor
USE_CYCLES             = True   # price-cycle phase and province-wide moves

WEST = {'BC', 'AB', 'SK', 'MB', 'YT', 'NT'}
JUMP_CENTS = 3.0                # ex-tax daily rise that counts as a cycle "jump"

_HOLIDAYS = {}


def _holiday_dates(province):
    if province not in _HOLIDAYS:
        h = holidays.Canada(subdiv=province, years=range(2016, 2036))
        _HOLIDAYS[province] = pd.DatetimeIndex(sorted(h.keys()))
    return _HOLIDAYS[province]


def _holiday_within(index, province, days):
    hol = _holiday_dates(province)
    nxt = hol.searchsorted(index, side='right')
    gap = np.full(len(index), np.inf)
    ok = nxt < len(hol)
    gap[ok] = (hol[nxt[ok]] - index[ok]).days
    return (gap <= days).astype(float)


def ex_tax(gas, slug):
    """Retail price minus fixed per-litre taxes (grossed up by sales tax)."""
    c = CITIES[slug]
    return gas.astype(float) - taxes.fixed_tax_series(c.province, slug, gas.index)


def build_features(gas, market, slug):
    """
    gas    : calendar-daily Series of retail prices (¢/L) for one city
    market : DataFrame from data.align_market() on the same index
    Returns a DataFrame of features indexed by origin date (rows with NaNs dropped).
    """
    city = CITIES[slug]
    g = gas.astype(float)
    x = ex_tax(g, slug)
    f = pd.DataFrame(index=g.index)

    f['gas'] = g
    f['gas_ex'] = x
    for k in [1, 3, 7, 14, 28]:
        f[f'gas_chg_{k}'] = x - x.shift(k)
    for w in [7, 28, 90]:
        f[f'gas_dev_mean_{w}'] = x - x.rolling(w).mean()
    for w in [7, 28]:
        f[f'gas_std_{w}'] = x.diff().rolling(w).std()

    # Retail tracks wholesale gasoline with a lag; the retail–wholesale margin mean-reverts.
    # Western Canada's supply is priced closer to the US Gulf Coast than New York Harbor.
    west = USE_REGIONAL_WHOLESALE and city.province in WEST
    rbob = market['gulf_cad_l'] if west else market['rbob_cad_l']
    margin = x / (1 + taxes.components(city.province, slug, g.index[-1])['sales_rate']) - rbob
    f['rbob']   = rbob
    f['margin'] = margin
    for w in [28, 90, 365]:
        f[f'margin_dev_{w}'] = margin - margin.rolling(w, min_periods=min(w, 60)).mean()
    for k in [1, 3, 7, 14]:
        f[f'rbob_chg_{k}'] = rbob - rbob.shift(k)
    f['rbob_ma7_chg'] = rbob.rolling(7).mean() - rbob.rolling(7).mean().shift(7)
    for k in [1, 7]:
        f[f'wti_cad_chg_{k}'] = market['wti_cad'] - market['wti_cad'].shift(k)
    f['wti_cad']      = market['wti_cad']
    f['crack']        = market['crack_usd']
    f['cadusd_chg_7'] = market['cadusd'] - market['cadusd'].shift(7)

    f['day_of_week']  = g.index.dayofweek
    f['month']        = g.index.month
    f['holiday_in_7'] = _holiday_within(g.index, city.province, 7)
    f['regulated']    = float(city.province in REGULATED)

    if USE_CYCLES:
        # Sawtooth cycles: sharp jumps, then gradual declines. Track the phase.
        d1 = x.diff()
        jump = d1 >= JUMP_CENTS
        last_jump = pd.Series(np.where(jump, np.arange(len(x)), np.nan), index=x.index).ffill()
        f['days_since_jump'] = np.minimum(np.arange(len(x)) - last_jump, 60).fillna(60)
        f['last_jump_size'] = d1.where(jump).ffill().fillna(0)

    f['province']     = PROVINCES.index(city.province)
    f['city']         = list(CITIES).index(slug)

    return f.dropna()


def build_panel(prices, market_aligned):
    """Stack per-city features into one frame with a (city, date) MultiIndex."""
    frames = []
    for slug in prices.columns:
        g = prices[slug].dropna()
        if len(g) < 400:
            continue
        f = build_features(g, market_aligned.reindex(g.index), slug)
        f.index = pd.MultiIndex.from_product([[slug], f.index], names=['city_slug', 'date'])
        frames.append(f)
    panel = pd.concat(frames)
    if USE_CYCLES:
        # Province-wide moves at the origin (other cities' same-day prices are known by then)
        prov = panel['province']
        dates = panel.index.get_level_values(1)
        for k in (1, 3, 7):
            col = f'gas_chg_{k}'
            med = panel.groupby([prov.values, dates])[col].transform('median')
            panel[f'prov_chg_{k}'] = med
        panel['gap_to_prov_7'] = panel['gas_chg_7'] - panel['prov_chg_7']
    return panel


def targets(prices_ex, panel_index, horizons=HORIZONS):
    """DataFrame of ex-tax change x[t+h] - x[t] for each horizon, aligned to the panel index."""
    out = {}
    cities = panel_index.get_level_values(0)
    dates  = panel_index.get_level_values(1)
    for h in horizons:
        vals = np.full(len(panel_index), np.nan)
        for slug in pd.unique(cities):
            m = cities == slug
            x = prices_ex[slug]
            d = dates[m]
            vals[m] = x.reindex(d + pd.Timedelta(days=h)).values - x.reindex(d).values
        out[h] = vals
    return pd.DataFrame(out, index=panel_index)


def scheduled_tax_change(slug, origins, horizon):
    """Change in fixed retail taxes between each origin and origin + horizon, as known at the origin."""
    return taxes.scheduled_change(CITIES[slug].province, slug, origins, horizon)
