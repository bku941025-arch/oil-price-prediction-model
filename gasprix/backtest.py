"""
backtest.py — Out-of-sample evaluation for every city.

    python -m gasprix.backtest

Trains on origins up to 2023-12-31 and forecasts every day from 2024-01-01 on.
Interval coverage is checked honestly: residual quantiles are taken from 2024 and
coverage is measured on 2025 onward.
"""

import warnings

import pandas as pd

from . import data, features
from .model import INTERVAL, DirectEnsemble, coverage, long_predictions, metrics_by_horizon

SPLIT = pd.Timestamp('2023-12-31')
CAL_END = pd.Timestamp('2024-12-31')
SHOW = [1, 3, 7, 14, 21, 28]


def run():
    warnings.filterwarnings('ignore')
    gas    = data.load_gas_prices()
    market = data.fetch_market()

    summary = []
    for slug in gas.columns:
        g = gas[slug]
        feats = features.build_features(g, data.align_market(market, g.index))
        model = DirectEnsemble().fit(feats[feats.index <= SPLIT], g, last_date=SPLIT)
        long  = long_predictions(model, feats[feats.index > SPLIT], g).dropna(subset=['actual'])
        m = metrics_by_horizon(long)
        cal = metrics_by_horizon(long[long['origin'] <= CAL_END])
        cov = coverage(long[long['origin'] > CAL_END], cal)

        print(f'\n── {data.CITIES[slug][1]} ' + '─' * 40)
        print(f'  shrink weights: ' + ' '.join(f'h{h}={model.shrink[h]:.2f}' for h in SHOW))
        print(m.loc[SHOW, ['mae', 'persistence_mae', 'directional_acc']].round(3).to_string())
        print(f'  {INTERVAL:.0%} interval coverage on 2025+: {cov:.1%}')
        summary.append({'city': data.CITIES[slug][1],
                        'mae_1': m.loc[1, 'mae'], 'mae_7': m.loc[7, 'mae'], 'mae_28': m.loc[28, 'mae'],
                        'pers_7': m.loc[7, 'persistence_mae'],
                        'mae_avg': m['mae'].mean(), 'pers_avg': m['persistence_mae'].mean(),
                        'dir_7': m.loc[7, 'directional_acc'], 'coverage': cov})

    print('\n── Summary (¢/L, test period 2024-01-01 → present) ' + '─' * 10)
    print(pd.DataFrame(summary).set_index('city').round(3).to_string())


if __name__ == '__main__':
    run()
