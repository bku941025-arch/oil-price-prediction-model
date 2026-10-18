"""
backtest.py — Out-of-sample evaluation across every city.

    python -m gasprix.backtest

Trains one pooled model on origins up to 2023-12-31 and forecasts every day from
2024-01-01 on, using market data only as it would have been published at the time.
Interval coverage is checked honestly: residual quantiles come from 2024 and coverage
is measured on 2025 onward.
"""

import time
import warnings

import pandas as pd

from . import data, features
from .cities import CITIES
from .model import INTERVAL, DirectEnsemble, coverage, long_predictions, metrics_by_horizon

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


def run():
    warnings.filterwarnings('ignore')
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
    cal = metrics_by_horizon(long[long['origin'] <= CAL_END])
    print(f'\nAll {long["city"].nunique()} cities, test period 2024-01-01 → {long["target_date"].max().date()}')
    print('  shrink weights: ' + ' '.join(f'h{h}={model.shrink[h]:.2f}' for h in SHOW))
    print(m.loc[SHOW, ['mae', 'persistence_mae', 'directional_acc']].round(3).to_string())
    print(f'  {INTERVAL:.0%} interval coverage on 2025+: {coverage(long[long["origin"] > CAL_END], cal):.1%}')

    rows = {CITIES[s].name: summarize(long[long['city'] == s]) for s in HIGHLIGHT if s in prices.columns}
    rows['All cities'] = summarize(long)
    print('\n── Selected cities (¢/L) ' + '─' * 40)
    print(pd.DataFrame(rows).T.round(3).to_string())

    per_city = pd.DataFrame({s: summarize(long[long['city'] == s]) for s in long['city'].unique()}).T
    better = (per_city['mae_avg'] < per_city['pers_avg']).sum()
    print(f'\nModel beats "no change" on average error in {better} of {len(per_city)} cities')
    print(f'Runtime: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    run()
