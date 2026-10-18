"""
insights.py — Explanations and analyses published alongside the forecast.

  explain()        which drivers push each forecast up or down
  scenarios()      the forecast if wholesale gasoline and crude moved by ±X% today
  breakdown()      the pump price split into crude, refining, taxes and retail margin
  margin_gauge()   today's retail margin vs its normal range
  weekday_effect() average price by weekday
"""

import numpy as np
import pandas as pd

from . import taxes
from .cities import CITIES

LITRES_PER_BARREL = 158.987

DRIVER_GROUPS = {
    'wholesale': lambda c: c.startswith('rbob_chg'),
    'crude_fx' : lambda c: c.startswith('wti_cad_chg') or c == 'cadusd_chg_7',
    'margin'   : lambda c: c.startswith('margin_dev'),
    'momentum' : lambda c: c.startswith(('gas_chg', 'gas_dev_mean', 'gas_std')),
    'calendar' : lambda c: c in ('day_of_week', 'holiday_in_7'),
    'local'    : lambda c: c in ('city', 'province', 'regulated'),
}


def _group(col):
    for name, test in DRIVER_GROUPS.items():
        if test(col):
            return name
    return 'local'


def explain(model, rows, horizons, tax_change):
    """
    Split each predicted change into driver groups. Ridge contributions are exact
    (coefficient x standardized deviation); LightGBM's are SHAP values. Both are averaged
    and shrunk exactly like the forecast, so the parts sum to the forecast change.
    rows: panel rows (one per city); tax_change: {h: array of scheduled tax change per row}.
    Returns {h: DataFrame (rows x groups + 'baseline' + 'taxes')}.
    """
    X = rows[model.columns]
    out = {}
    for h in horizons:
        ridge, lgbm = model.models[h]
        scaler, lin = ridge.named_steps['standardscaler'], ridge.named_steps['ridge']
        xr = X[model.ridge_columns]
        r_contrib = (xr.values - scaler.mean_) / scaler.scale_ * lin.coef_
        r = pd.DataFrame(r_contrib, columns=model.ridge_columns, index=rows.index)
        l_contrib = lgbm.predict(X, pred_contrib=True)
        lg = pd.DataFrame(l_contrib[:, :-1], columns=model.columns, index=rows.index)

        w = model.shrink[h] / 2
        parts = pd.DataFrame(0.0, index=rows.index, columns=list(DRIVER_GROUPS))
        for col in model.columns:
            v = lg[col] + (r[col] if col in r else 0)
            parts[_group(col)] += w * v
        parts['baseline'] = w * (lin.intercept_ + l_contrib[:, -1])
        parts['taxes'] = tax_change[h]
        out[h] = parts
    return out


def scenarios(model, rows, shocks=(-0.2, -0.1, 0.1, 0.2)):
    """
    Forecast changes if wholesale gasoline and crude jumped by each shock today.
    Shifts the change and margin features the way an overnight jump would.
    Returns {shock: DataFrame (rows x horizons) of predicted ex-tax change}.
    """
    out = {}
    for s in shocks:
        x = rows.copy()
        d_rbob = x['rbob'] * s
        for k in (1, 3, 7, 14):
            x[f'rbob_chg_{k}'] += d_rbob
        for w in (28, 90, 365):
            x[f'margin_dev_{w}'] -= d_rbob
        x['margin'] -= d_rbob
        d_wti = x['wti_cad'] * s       # crude moves with gasoline
        for k in (1, 7):
            x[f'wti_cad_chg_{k}'] += d_wti
        out[s] = model.predict_change(x)
    return out


def breakdown_series(gas, market, slug, weeks=104):
    """Weekly pump price split into crude, refining, wholesale-to-retail margin and taxes."""
    c = CITIES[slug]
    end = gas.dropna().index.max()
    dates = pd.date_range(end - pd.Timedelta(weeks=weeks), end, freq='7D')
    rows = []
    for d in dates:
        price = gas.get(d)
        if price is None or np.isnan(price) or d not in market.index:
            continue
        m = market.loc[d]
        if np.isnan(m['rbob_cad_l']) or np.isnan(m['wti_cad']):
            continue
        b = taxes.breakdown(price, c.province, slug, d)
        crude = m['wti_cad'] / LITRES_PER_BARREL * 100
        rows.append({
            'date'     : d.strftime('%Y-%m-%d'),
            'price'    : round(price, 1),
            'crude'    : round(crude, 1),
            'refining' : round(m['rbob_cad_l'] - crude, 1),
            'retail'   : round(b['pre_tax'] - m['rbob_cad_l'], 1),
            'fuel_tax' : round(b['federal_excise'] + b['provincial'], 1),
            'carbon'   : round(b['carbon'], 1),
            'sales_tax': round(b['sales_tax'], 1),
        })
    return rows


def margin_gauge(feat_row, margin_history):
    """Today's retail-over-wholesale margin vs the past year (percentile and deviation)."""
    hist = margin_history.dropna().iloc[-365:]
    today = float(feat_row['margin'])
    pct = float((hist < today).mean()) if len(hist) else None
    return {
        'margin'    : round(today, 1),
        'normal'    : round(float(hist.mean()), 1) if len(hist) else None,
        'low'       : round(float(hist.quantile(0.1)), 1) if len(hist) else None,
        'high'      : round(float(hist.quantile(0.9)), 1) if len(hist) else None,
        'percentile': round(pct, 2) if pct is not None else None,
    }


def weekday_effect(raw_prices, years=2):
    """
    Average deviation (¢/L) of each weekday's price from its own week's average, using
    only days Kalibrate actually reported. Returns {weekday: {mean, se, n}} for Mon–Fri.
    """
    s = raw_prices.dropna()
    s = s[s.index >= s.index.max() - pd.Timedelta(days=365 * years)]
    s = s[s.index.dayofweek < 5]
    week = s.groupby(s.index.to_period('W')).transform('mean')
    dev = s - week
    out = {}
    for wd, d in dev.groupby(dev.index.dayofweek):
        out[int(wd)] = {'mean': round(float(d.mean()), 2),
                        'se': round(float(d.std() / np.sqrt(len(d))), 2), 'n': int(len(d))}
    return out


def tax_anomalies(prices, market, lookback_days=10, min_cities=3, min_jump=8.0, agree=0.8):
    """
    Province-wide price jumps that neither the tax table nor wholesale moves explain, i.e. a
    tax change that may be missing from taxes.py. Compares consecutive days Kalibrate actually
    reported (weekdays, not holidays) and skips regulated provinces, whose weekly settings
    jump by design. Tuned on 2016–2026: with the table, about one flag a year (large market
    spikes); without it, every major tax change is flagged.
    Returns [{province, date, change}].
    """
    import holidays as hol

    from .cities import REGULATED
    from .features import ex_tax  # avoid a circular import at module load

    end = prices.index.max()
    national = hol.Canada(years=range(end.year - 1, end.year + 1))
    reported = [d for d in prices.index[prices.index > end - pd.Timedelta(days=lookback_days + 7)]
                if d.dayofweek < 5 and d not in national]
    rbob = market['rbob_cad_l']
    whole = (rbob - rbob.shift(7)).abs().reindex(prices.index)
    by_prov = {}
    for slug in prices.columns:
        by_prov.setdefault(CITIES[slug].province, []).append(slug)

    out = []
    for prov, slugs in by_prov.items():
        if prov in REGULATED or len(slugs) < min_cities:
            continue
        ex = pd.DataFrame({s: ex_tax(prices[s].dropna(), s) for s in slugs if prices[s].notna().any()})
        d1 = ex.reindex(reported).diff()
        for day, row in d1[d1.index > end - pd.Timedelta(days=lookback_days)].iterrows():
            row = row.dropna()
            if len(row) < min_cities:
                continue
            med = row.median()
            same = (np.sign(row) == np.sign(med)).mean()
            if abs(med) >= min_jump and same >= agree and (whole.get(day) or 0) < min_jump / 2:
                out.append({'province': prov, 'date': day.strftime('%Y-%m-%d'), 'change': round(float(med), 1)})
    return out
