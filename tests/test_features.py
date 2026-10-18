"""Feature engineering: no look-ahead, correct targets, publication lags respected."""

import numpy as np
import pandas as pd

from gasprix import data, features


def _synthetic(days=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2022-01-01', periods=days, freq='D')
    gas = pd.Series(170 + rng.normal(0, 1, days).cumsum(), index=idx)
    market = pd.DataFrame({
        'wti_usd': 80 + rng.normal(0, 1, days).cumsum(),
        'rbob_usd': 2.5 + rng.normal(0, 0.02, days).cumsum(),
        'cadusd_fred': 1.35 + rng.normal(0, 0.002, days).cumsum(),
        'cadusd': 1.35 + rng.normal(0, 0.002, days).cumsum(),
    }, index=idx)
    return gas, market


def test_features_do_not_use_future_data():
    """Changing prices after day t must not change any feature at or before t."""
    gas, market = _synthetic()
    aligned = data.align_market(market, gas.index)
    base = features.build_features(gas, aligned, 'montreal')

    cut = gas.index[400]
    gas2, market2 = gas.copy(), market.copy()
    gas2[gas2.index > cut] += 50
    market2[market2.index > cut] *= 1.5
    shocked = features.build_features(gas2, data.align_market(market2, gas.index), 'montreal')

    upto = base.index[base.index <= cut]
    pd.testing.assert_frame_equal(base.loc[upto], shocked.loc[upto])


def test_market_data_respects_publication_lag():
    gas, market = _synthetic()
    aligned = data.align_market(market, gas.index)
    day = gas.index[300]
    lag = data.LAGS['rbob_usd']
    assert aligned.loc[day, 'rbob_usd'] == market.loc[day - pd.Timedelta(days=lag), 'rbob_usd']


def test_targets_are_future_ex_tax_changes():
    gas, market = _synthetic()
    panel = features.build_panel(pd.DataFrame({'montreal': gas}), data.align_market(market, gas.index))
    ex = pd.DataFrame({'montreal': features.ex_tax(gas, 'montreal')})
    Y = features.targets(ex, panel.index, horizons=[1, 7])
    city, day = panel.index[10]
    assert np.isclose(Y[7].iloc[10], ex['montreal'][day + pd.Timedelta(days=7)] - ex['montreal'][day])


def test_panel_has_city_and_province_codes():
    gas, market = _synthetic()
    prices = pd.DataFrame({'montreal': gas, 'toronto': gas + 5})
    panel = features.build_panel(prices, data.align_market(market, gas.index))
    assert set(panel.index.get_level_values(0)) == {'montreal', 'toronto'}
    assert panel.groupby(level=0)['province'].nunique().eq(1).all()
