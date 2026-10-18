"""
taxes.py — Approximate per-litre gasoline taxes by province over time (¢/L, regular).

Used for two things:
  1. Removing tax step changes from training targets, and adding *scheduled* changes
     (e.g. the federal excise returning in 2027) on top of the market forecast.
  2. The price breakdown chart.

Only large, well-documented changes are modelled; small regional variations
(e.g. Québec's reduced rates in some peripheral and border zones) are not.
Sources:
  - Federal excise suspension: pm.gc.ca 2026-04-14; canada.ca (Finance) 2026-09
  - Federal fuel charge rates: canada.ca CRA "Fuel charge rates"; set to 0 on 2025-04-01
  - BC motor fuel & carbon tax: www2.gov.bc.ca "Motor fuel tax and carbon tax rates"
  - Québec: Revenu Québec / CAA-Québec "Composantes du prix de l'essence"
  - Alberta Fuel Tax Relief Program: suspended from 2026-10-01 (announced 2026-09-22)
  - Other provinces: provincial finance ministries; cwf.ca fuel tax summary
"""

import bisect

import pandas as pd

# Each schedule is a list of (start date, ¢/L) sorted by date; value applies until the next entry.
FEDERAL_EXCISE = {
    'gasoline': [('2000-01-01', 10.0), ('2026-04-20', 0.0), ('2027-02-01', 5.0), ('2027-04-01', 10.0)],
    'diesel'  : [('2000-01-01', 4.0),  ('2026-04-20', 0.0), ('2027-02-01', 2.0), ('2027-04-01', 4.0)],
}

_FED_CHARGE = [('2019-04-01', 4.42), ('2020-04-01', 6.63), ('2021-04-01', 8.84), ('2022-04-01', 11.05),
               ('2023-04-01', 14.31), ('2024-04-01', 17.61), ('2025-04-01', 0.0)]


def _from(start, sched, before=0.0):
    """Schedule that is `before` until `start`, then follows `sched` from its first entry on or after start."""
    tail = [(d, v) for d, v in sched if d > start]
    upto = [v for d, v in sched if d <= start]
    first = upto[-1] if upto else sched[0][1]
    return [('2000-01-01', before), (start, first)] + tail


CARBON = {
    'ON': _from('2019-04-01', _FED_CHARGE),
    'SK': _from('2019-04-01', _FED_CHARGE),
    'MB': _from('2019-04-01', _FED_CHARGE),
    'AB': [('2000-01-01', 6.73), ('2019-05-30', 0.0)] + [e for e in _from('2020-01-01', _FED_CHARGE)[1:]],
    'BC': [('2000-01-01', 6.67), ('2018-04-01', 7.78), ('2019-04-01', 8.89), ('2021-04-01', 9.96),
           ('2022-04-01', 11.05), ('2023-04-01', 14.31), ('2024-04-01', 17.61), ('2025-04-01', 0.0)],
    'NB': _from('2020-04-01', _FED_CHARGE),
    'NS': _from('2023-07-01', _FED_CHARGE),
    'PE': _from('2019-04-01', _FED_CHARGE),
    'NL': _from('2019-01-01', _FED_CHARGE),
    'YT': _from('2019-07-01', _FED_CHARGE),
    'NT': _from('2019-09-01', _FED_CHARGE),
    'QC': [('2000-01-01', 0.0)],   # cap-and-trade cost is embedded in wholesale, not a pump tax
}

PROVINCIAL = {
    'QC': [('2000-01-01', 19.2)],
    'ON': [('2000-01-01', 14.7), ('2022-07-01', 9.0)],
    'MB': [('2000-01-01', 14.0), ('2024-01-01', 0.0), ('2025-01-01', 12.5)],
    'SK': [('2000-01-01', 15.0)],
    'AB': [('2000-01-01', 13.0), ('2022-04-01', 0.0), ('2024-01-01', 13.0),
           ('2026-10-01', 0.0)],   # relief program: 0 when WTI averages over US$90 (reviewed quarterly)
    'BC': [('2000-01-01', 14.5)],
    'NB': [('2000-01-01', 15.5), ('2020-04-01', 10.87)],
    'NS': [('2000-01-01', 15.5)],
    'PE': [('2000-01-01', 13.1), ('2023-07-01', 8.47)],
    'NL': [('2000-01-01', 14.5), ('2022-04-01', 7.5)],
    'YT': [('2000-01-01', 6.2)],
    'NT': [('2000-01-01', 10.7)],
}

# Extra city-level fuel taxes, replacing nothing (added on top of PROVINCIAL)
MUNICIPAL = {
    'montreal' : [('2000-01-01', 3.0)],                         # ARTM transit levy
    'vancouver': [('2000-01-01', 11.0), ('2018-07-01', 12.5)],  # TransLink 17→18.5, net of a lower provincial base
    'victoria' : [('2000-01-01', 5.5)],                         # BC Transit levy
}

SALES_TAX = {   # ad valorem rate applied to the whole pump price
    'QC': [('2000-01-01', 0.14975)],
    'ON': [('2000-01-01', 0.13)],
    'NB': [('2000-01-01', 0.15)],
    'NL': [('2000-01-01', 0.15)],
    'PE': [('2000-01-01', 0.14), ('2016-10-01', 0.15)],
    'NS': [('2000-01-01', 0.15), ('2025-04-01', 0.14)],
}
GST_ONLY = 0.05

# When each change became public, for forecasts made before it took effect. Keys are
# (component, start date, province or None for all). Unlisted changes are assumed to be
# known 45 days ahead (budgets and annual carbon increases were legislated well before).
ANNOUNCED = {
    ('carbon', '2025-04-01', None)        : '2025-03-14',   # federal charge and BC carbon tax removed
    ('federal_excise', '2026-04-20', None): '2026-04-14',   # excise suspension
    ('federal_excise', '2027-02-01', None): '2026-09-02',   # suspension extended, then phased back in
    ('federal_excise', '2027-04-01', None): '2026-09-02',
    ('provincial', '2022-07-01', 'ON')    : '2022-04-04',
    ('provincial', '2022-04-01', 'AB')    : '2022-02-24',
    ('provincial', '2024-01-01', 'AB')    : '2023-11-30',
    ('provincial', '2026-10-01', 'AB')    : '2026-09-22',
    ('provincial', '2024-01-01', 'MB')    : '2023-11-21',
    ('provincial', '2025-01-01', 'MB')    : '2024-11-01',
    ('provincial', '2022-04-01', 'NL')    : '2022-04-07',
}
DEFAULT_NOTICE_DAYS = 45

# Announced changes that were later cancelled: (component, start, ¢/L by product, announced, cancelled).
# Forecasts made in between expected them.
CANCELLED = [
    ('federal_excise', '2026-09-08', {'gasoline': 10.0, 'diesel': 4.0}, '2026-04-14', '2026-09-02'),
]


def _value(sched, date):
    dates = [d for d, _ in sched]
    i = bisect.bisect_right(dates, pd.Timestamp(date).strftime('%Y-%m-%d')) - 1
    return sched[max(i, 0)][1]


def components(province, city_slug, date, product='gasoline'):
    """Per-litre taxes (¢/L, before sales tax) and the sales tax rate in effect on `date`."""
    return {
        'federal_excise': _value(FEDERAL_EXCISE[product], date),
        'carbon'        : _value(CARBON.get(province, [('2000-01-01', 0.0)]), date),
        'provincial'    : _value(PROVINCIAL.get(province, [('2000-01-01', 0.0)]), date),
        'municipal'     : _value(MUNICIPAL.get(city_slug, [('2000-01-01', 0.0)]), date),
        'sales_rate'    : _value(SALES_TAX.get(province, [('2000-01-01', GST_ONLY)]), date),
    }


def fixed_taxes_retail(province, city_slug, date, product='gasoline'):
    """Per-litre taxes as they show up in the pump price (fixed taxes grossed up by sales tax)."""
    c = components(province, city_slug, date, product)
    return (c['federal_excise'] + c['carbon'] + c['provincial'] + c['municipal']) * (1 + c['sales_rate'])


def fixed_tax_series(province, city_slug, index, product='gasoline'):
    """Daily Series of fixed_taxes_retail over a DatetimeIndex."""
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    change_dates = sorted({pd.Timestamp(d) for sched in (
        FEDERAL_EXCISE[product], CARBON.get(province, []), PROVINCIAL.get(province, []),
        MUNICIPAL.get(city_slug, []), SALES_TAX.get(province, [])) for d, _ in sched} | {index.min()})
    s = pd.Series({d: fixed_taxes_retail(province, city_slug, d, product) for d in change_dates})
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def breakdown(price, province, city_slug, date):
    """Split a pump price (¢/L) into tax components and the pre-tax remainder."""
    c = components(province, city_slug, date)
    sales = price * c['sales_rate'] / (1 + c['sales_rate'])
    return {
        'federal_excise': c['federal_excise'], 'carbon': c['carbon'],
        'provincial': c['provincial'] + c['municipal'], 'sales_tax': sales,
        'pre_tax': price - sales - c['federal_excise'] - c['carbon'] - c['provincial'] - c['municipal'],
    }


def _schedules(province, city_slug, product='gasoline'):
    return {
        'federal_excise': FEDERAL_EXCISE[product],
        'carbon'        : CARBON.get(province, []),
        'provincial'    : PROVINCIAL.get(province, []),
        'municipal'     : MUNICIPAL.get(city_slug, []),
    }


def _announced(component, start, province):
    a = ANNOUNCED.get((component, start, province)) or ANNOUNCED.get((component, start, None))
    return pd.Timestamp(a) if a else pd.Timestamp(start) - pd.Timedelta(days=DEFAULT_NOTICE_DAYS)


def scheduled_change(province, city_slug, origins, horizon, product='gasoline'):
    """
    Change in fixed retail taxes from each origin to origin + horizon, *as it was known on
    the origin date*: changes not yet announced are left out, and announced changes that
    were later cancelled are included.
    """
    idx = pd.DatetimeIndex(origins)
    tgt = idx + pd.Timedelta(days=horizon)
    span = pd.date_range(idx.min() - pd.Timedelta(days=1), tgt.max(), freq='D')
    actual = fixed_tax_series(province, city_slug, span, product)
    change = actual.reindex(tgt).values - actual.reindex(idx).values

    def window(start):
        s = pd.Timestamp(start)
        return (idx < s) & (tgt >= s)

    for comp, sched in _schedules(province, city_slug, product).items():
        for (_, v0), (d1, v1) in zip(sched, sched[1:]):
            unknown = window(d1) & (idx < _announced(comp, d1, province))
            if unknown.any():
                rate = components(province, city_slug, d1, product)['sales_rate']
                change[unknown] -= (v1 - v0) * (1 + rate)
    for comp, start, values, announced, cancelled in CANCELLED:
        expected = window(start) & (idx >= pd.Timestamp(announced)) & (idx < pd.Timestamp(cancelled))
        if expected.any():
            c = components(province, city_slug, start, product)
            change[expected] += (values[product] - c[comp]) * (1 + c['sales_rate'])
    return change
