"""
backtest.py — Out-of-sample evaluation across every city.

    python -m gasprix.backtest
    python -m gasprix.backtest --without cycles,regional    # measure a feature group

Trains one pooled model on origins up to 2023-12-31 and forecasts every day from
2024-01-01 on, using market data only as it would have been published at the time.
Interval coverage is checked honestly: residual quantiles come from 2024 and coverage
is measured on 2025 onward.
"""

import argparse
import time
import warnings

import pandas as pd

from . import data, features
from .cities import CITIES, REGULATED
from .model import (
    INTERVAL,
    Calibration,
    DirectEnsemble,
    coverage,
    crps,
    long_predictions,
    metrics_by_horizon,
    volatility_groups,
)

SPLIT = pd.Timestamp('2023-12-31')
CAL_END = pd.Timestamp('2024-12-31')
SHOW = [1, 3, 7, 14, 21, 28]
HIGHLIGHT = ['montreal', 'quebec', 'gatineau', 'sherbrooke', 'trois-rivieres',
             'toronto', 'ottawa', 'vancouver', 'calgary', 'winnipeg', 'halifax']


def summarize(long):
    m = metrics_by_horizon(long)
    return {'mae_1': m.loc[1, 'mae'], 'mae_7': m.loc[7, 'mae'], 'pers_7': m.loc[7, 'persistence_mae'],
            'mae_28': m.loc[28, 'mae'], 'pers_28': m.loc[28, 'persistence_mae'],
            'mae_avg': m['mae'].mean(), 'pers_avg': m['persistence_mae'].mean(),
            'dir_14': m.loc[14, 'directional_acc']}


GROUPS = {'regional': 'USE_REGIONAL_WHOLESALE', 'cycles': 'USE_CYCLES'}


def run(without=(), save=None):
    warnings.filterwarnings('ignore')
    for g in without:
        setattr(features, GROUPS[g], False)
    if without:
        print(f'Without: {", ".join(without)}')
    t0 = time.time()
    prices = data.load_prices('regular')
    market = data.fetch_market()
    panel = features.build_panel(prices, data.align_market(market, prices.index))
    ex = pd.DataFrame({s: features.ex_tax(prices[s].dropna(), s) for s in prices.columns})
    Y = features.targets(ex, panel.index)

    d = panel.index.get_level_values(1)
    model = DirectEnsemble().fit(panel[d <= SPLIT], Y[d <= SPLIT], last_date=SPLIT)
    long = long_predictions(model, panel[d > SPLIT], prices).dropna(subset=['actual'])

    m = metrics_by_horizon(long)
    groups = volatility_groups(panel[d <= SPLIT])
    cal_long, ev_long = long[long['origin'] <= CAL_END], long[long['origin'] > CAL_END]
    cal = Calibration(cal_long, groups)
    base_cal = metrics_by_horizon(cal_long.assign(pred=cal_long['base']))
    print(f'\nAll {long["city"].nunique()} cities, test period 2024-01-01 → {long["target_date"].max().date()}')
    print('  shrink weights: ' + ' '.join(f'h{h}={model.shrink[h]:.2f}' for h in SHOW))
    print(m.loc[SHOW, ['mae', 'persistence_mae', 'directional_acc']].round(3).to_string())
    print(f'  {INTERVAL:.0%} interval coverage on 2025+: {coverage(ev_long, cal):.1%}   '
          f'(50%: {coverage(ev_long, cal, 0.5):.1%}, 90%: {coverage(ev_long, cal, 0.9):.1%})')
    print(f'  CRPS on 2025+: model {crps(ev_long, cal):.3f} vs "no change" {crps(ev_long, base_cal, center="base"):.3f} ¢/L')

    rows = {CITIES[s].name: summarize(long[long['city'] == s]) for s in HIGHLIGHT if s in prices.columns}
    rows['All cities'] = summarize(long)
    print('\n── Selected cities (¢/L) ' + '─' * 40)
    print(pd.DataFrame(rows).T.round(3).to_string())

    prov = long['city'].map(lambda s: CITIES[s].province)
    by_prov = pd.DataFrame({p: summarize(long[prov == p]) for p in sorted(prov.unique())}).T
    print('\n── By province (¢/L) ' + '─' * 44)
    print(by_prov[['mae_7', 'pers_7', 'mae_avg', 'pers_avg']].round(3).to_string())
    reg = prov.isin(REGULATED)
    for name, mask in (('Regulated', reg), ('Unregulated', ~reg)):
        r = summarize(long[mask])
        print(f'{name:<12} avg {r["mae_avg"]:.3f} vs {r["pers_avg"]:.3f}   7-day {r["mae_7"]:.3f} vs {r["pers_7"]:.3f}')
    if save:
        long.to_pickle(save)

    per_city = pd.DataFrame({s: summarize(long[long['city'] == s]) for s in long['city'].unique()}).T
    better = (per_city['mae_avg'] < per_city['pers_avg']).sum()
    print(f'\nModel beats "no change" on average error in {better} of {len(per_city)} cities')
    print(f'Runtime: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--without', default='', help=f'comma-separated: {", ".join(GROUPS)}')
    ap.add_argument('--save', help='pickle the long prediction frame here')
    a = ap.parse_args()
    run([g for g in a.without.split(',') if g], a.save)
