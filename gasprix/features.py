"""
features.py — Feature engineering shared by training, backtesting and forecasting.

Every feature for origin date t uses only information known at the end of day t,
including that day's retail price. Models predict the change gas[t+h] - gas[t].
"""

import holidays
import numpy as np
import pandas as pd

HORIZONS = list(range(1, 29))   # calendar days ahead

_QC_HOLIDAYS = holidays.Canada(subdiv='QC', years=range(2016, 2036))


def _holiday_within(index, days):
    hol = pd.DatetimeIndex(sorted(_QC_HOLIDAYS.keys()))
    nxt = hol.searchsorted(index, side='right')
    nxt_date = pd.Series(pd.NaT, index=index, dtype='datetime64[ns]')
    ok = nxt < len(hol)
    nxt_date[ok] = hol[nxt[ok]]
    return ((nxt_date - index.to_series()).dt.days <= days).astype(float).values


def build_features(gas, market):
    """
    gas    : calendar-daily Series of retail prices (¢/L) for one city
    market : DataFrame from data.align_market() on the same index
    Returns a DataFrame of features indexed by origin date (rows with NaNs dropped).
    """
    g = gas.astype(float)
    f = pd.DataFrame(index=g.index)

    f['gas'] = g
    for k in [1, 3, 7, 14, 28]:
        f[f'gas_chg_{k}'] = g - g.shift(k)
    for w in [7, 28, 90]:
        f[f'gas_dev_mean_{w}'] = g - g.rolling(w).mean()
    for w in [7, 28]:
        f[f'gas_std_{w}'] = g.diff().rolling(w).std()

    # Retail tracks wholesale gasoline with a lag; the retail–wholesale margin mean-reverts.
    rbob = market['rbob_cad_l']
    margin = g - rbob
    f['rbob']   = rbob
    f['margin'] = margin
    for w in [28, 90]:
        f[f'margin_dev_{w}'] = margin - margin.rolling(w).mean()
    for k in [1, 3, 7, 14]:
        f[f'rbob_chg_{k}'] = rbob - rbob.shift(k)
    for k in [1, 7]:
        f[f'wti_cad_chg_{k}'] = market['wti_cad'] - market['wti_cad'].shift(k)
    f['crack']        = market['crack_usd']
    f['cadusd_chg_7'] = market['cadusd'] - market['cadusd'].shift(7)

    f['day_of_week']   = g.index.dayofweek
    f['month']         = g.index.month
    f['holiday_in_7']  = _holiday_within(g.index, 7)

    return f.dropna()


def targets(gas, index, horizons=HORIZONS):
    """DataFrame of gas[t+h] - gas[t] for each horizon, aligned to the given origin index."""
    g = gas.astype(float)
    return pd.DataFrame({h: (g.shift(-h) - g).reindex(index) for h in horizons})
