"""
pipeline.py — Daily data refresh, retrain and forecast publish.
Called by GitHub Actions daily. Also callable manually:
    python pipeline.py

For each city it:
  1. refreshes Kalibrate retail prices and FRED market data (no API key needed)
  2. refuses to publish if either source is stale
  3. scores a fresh model on the most recent year it never saw (holdout)
  4. refuses to publish if that score is much worse than the last published run
  5. retrains on all data, forecasts 28 days with calibrated 80% ranges
  6. logs the forecast so it can be scored against actual prices later
  7. writes site/data/<city>.json, which the website reads
"""

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from gasprix import data, features
from gasprix.model import INTERVAL, DirectEnsemble, holdout_evaluation, interval, long_predictions

ROOT         = os.path.dirname(os.path.abspath(__file__))
SITE_DATA    = os.path.join(ROOT, 'site', 'data')
LOG_PATH     = os.path.join(data.DATA_DIR, 'predictions_log.csv')
METRICS_PATH = os.path.join(data.DATA_DIR, 'metrics.json')

MAX_GAS_STALENESS_DAYS    = 6     # Kalibrate is weekdays only, plus holidays
MAX_MARKET_STALENESS_DAYS = 14    # FRED's DEXCAUS lags about a week
MAX_MAE_DEGRADATION       = 1.25  # vs the last published run's holdout MAE
SIGNAL_THRESHOLD          = 2.0   # ¢/L expected move over 3 days that triggers an alert
FORCE_PUBLISH             = os.environ.get('FORCE_PUBLISH') == '1'

NTFY_TOPIC  = os.environ.get('NTFY_TOPIC', '')
NTFY_SERVER = os.environ.get('NTFY_SERVER', 'https://ntfy.sh')
SITE_URL    = os.environ.get('SITE_URL', 'https://gas-price-mtl.onrender.com')

LOG_HORIZONS      = [1, 3, 7, 14, 21, 28]   # horizons kept in the prediction log
SEED_LOG_HORIZONS = [1, 7]
LOG_COLUMNS = ['made_on', 'city', 'horizon', 'target_date', 'base', 'pred', 'low', 'high', 'source']


class PublishBlocked(Exception):
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)


def load_log():
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        if list(log.columns) == LOG_COLUMNS:
            log['made_on']     = pd.to_datetime(log['made_on'])
            log['target_date'] = pd.to_datetime(log['target_date'])
            return log
        print('  Prediction log has an old format — starting a new one')
    return pd.DataFrame(columns=LOG_COLUMNS)


def weekly_summary(forecast):
    weeks = []
    for w in range(4):
        days = forecast[w * 7:(w + 1) * 7]
        weeks.append({
            'week'     : w + 1,
            'date_from': days[0]['date'],
            'date_to'  : days[-1]['date'],
            'low'      : _r(min(d['low'] for d in days), 1),
            'mid'      : _r(np.mean([d['mid'] for d in days]), 1),
            'high'     : _r(max(d['high'] for d in days), 1),
        })
    return weeks


def price_signal(today_price, forecast):
    """'fill_now' if prices are expected to rise over the next 3 days, 'wait' if fall, else 'steady'."""
    change = np.mean([d['mid'] for d in forecast[:3]]) - today_price
    kind = 'fill_now' if change >= SIGNAL_THRESHOLD else 'wait' if change <= -SIGNAL_THRESHOLD else 'steady'
    return {'kind': kind, 'change_3d': _r(change, 1)}


def live_accuracy(log, slug, gas, as_of, days=90):
    """Score logged live forecasts whose target date has an actual price."""
    d = log[(log['city'] == slug) & (log['source'] == 'live')].copy()
    d['actual'] = gas.reindex(d['target_date']).values
    d = d.dropna(subset=['actual'])
    d = d[d['target_date'] > as_of - pd.Timedelta(days=days)]
    out = {'window_days': days, 'since': None, 'n': int(len(d))}
    if d.empty:
        return out
    out['since'] = d['made_on'].min().strftime('%Y-%m-%d')
    for h in [1, 7]:
        dh = d[d['horizon'] == h]
        out[f'mae_{h}']             = _r((dh['actual'] - dh['pred']).abs().mean()) if len(dh) else None
        out[f'persistence_mae_{h}'] = _r((dh['actual'] - dh['base']).abs().mean()) if len(dh) else None
        out[f'n_{h}']               = int(len(dh))
    out['coverage'] = _r(((d['actual'] >= d['low']) & (d['actual'] <= d['high'])).mean(), 3)
    return out


def track_record(log, slug, gas, days=180, horizon=7):
    """Actual prices alongside the forecast made `horizon` days earlier (live, else backtest)."""
    d = log[(log['city'] == slug) & (log['horizon'] == horizon)]
    d = d.sort_values('source').drop_duplicates('target_date', keep='last')  # 'live' sorts after 'backtest'
    d = d.set_index('target_date')
    end = gas.dropna().index.max()
    idx = pd.date_range(end - pd.Timedelta(days=days), end, freq='D')
    rows = []
    for t in idx:
        rec = {'date': t.strftime('%Y-%m-%d'), 'actual': _r(gas.get(t), 1)}
        if t in d.index:
            rec.update(pred=_r(d.at[t, 'pred'], 1), low=_r(d.at[t, 'low'], 1),
                       high=_r(d.at[t, 'high'], 1), source=d.at[t, 'source'])
        rows.append(rec)
    return rows


def notify(slug, name, today_price, signal, previous_signal):
    """Push a notification through ntfy when a city's signal changes to fill_now / wait."""
    if not NTFY_TOPIC or signal['kind'] == 'steady' or signal['kind'] == previous_signal:
        return
    if signal['kind'] == 'fill_now':
        title = f'{name}: fill up soon — prices expected to rise'
        body  = (f'Expected +{signal["change_3d"]:.1f} ¢/L over the next 3 days '
                 f'(now {today_price:.1f} ¢/L). / Hausse prévue : faites le plein bientôt.')
    else:
        title = f'{name}: prices expected to drop — consider waiting'
        body  = (f'Expected {signal["change_3d"]:.1f} ¢/L over the next 3 days '
                 f'(now {today_price:.1f} ¢/L). / Baisse prévue : attendez si possible.')
    try:
        requests.post(NTFY_SERVER, json={
            'topic': f'{NTFY_TOPIC}-{slug}', 'title': title, 'message': body,
            'click': f'{SITE_URL}/?city={slug}', 'tags': ['fuelpump'],
        }, timeout=20).raise_for_status()
        print(f'  Sent alert for {name}: {signal["kind"]}')
    except Exception as e:
        print(f'  WARNING: alert for {name} failed: {e}')


# ── Per-city run ──────────────────────────────────────────────────────────────
def run_city(slug, gas, market, log, prev_metrics, prev_site, today):
    name = data.CITIES[slug][1]
    g = gas[slug].dropna()
    feats = features.build_features(g, data.align_market(market, g.index))

    # Score on the most recent year, then gate against the last published run
    hold_long, hold = holdout_evaluation(feats, g)
    hold_mae = float(hold['mae'].mean())
    prev_mae = prev_metrics.get(slug, {}).get('holdout_mae')
    if prev_mae and hold_mae > prev_mae * MAX_MAE_DEGRADATION and not FORCE_PUBLISH:
        raise PublishBlocked(f'{name}: holdout MAE {hold_mae:.2f} is >{MAX_MAE_DEGRADATION:.0%} '
                             f'of last run ({prev_mae:.2f}). Set FORCE_PUBLISH=1 to override.')

    # Retrain on everything and forecast from the latest observed day
    model  = DirectEnsemble().fit(feats, g)
    origin = feats.iloc[[-1]]
    fc     = long_predictions(model, origin, g)
    lo, hi = interval(fc, hold)
    today_price = float(origin['gas'].iloc[0])
    forecast = [{'date': t.strftime('%Y-%m-%d'), 'horizon': int(h),
                 'low': _r(l, 1), 'mid': _r(m, 1), 'high': _r(u, 1)}
                for t, h, m, l, u in zip(fc['target_date'], fc['horizon'], fc['pred'], lo, hi)]

    # Log today's forecast; seed the log with the holdout backtest the first time
    new_rows = fc.assign(made_on=today, city=slug, low=lo, high=hi, source='live')[LOG_COLUMNS]
    new_rows = new_rows[new_rows['horizon'].isin(LOG_HORIZONS)]
    log = log[~((log['city'] == slug) & (log['made_on'] == today))]
    if not ((log['city'] == slug) & (log['source'] == 'backtest')).any():
        b_lo, b_hi = interval(hold_long, hold)
        seed = hold_long.assign(made_on=hold_long['origin'], city=slug, low=b_lo, high=b_hi,
                                source='backtest')[LOG_COLUMNS]
        log = pd.concat([seed[seed['horizon'].isin(SEED_LOG_HORIZONS)], log], ignore_index=True)
    log = pd.concat([log, new_rows], ignore_index=True)

    signal = price_signal(today_price, forecast)
    notify(slug, name, today_price, signal, prev_site.get('signal', {}).get('kind'))

    site = {
        'city'        : slug,
        'name'        : name,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_end'    : origin.index[0].strftime('%Y-%m-%d'),
        'today_price' : _r(today_price, 1),
        'interval'    : INTERVAL,
        'forecast'    : forecast,
        'weekly'      : weekly_summary(forecast),
        'signal'      : signal,
        'historical'  : [{'date': t.strftime('%Y-%m-%d'), 'price': _r(p, 1)}
                         for t, p in g.iloc[-120:].items()],
        'track_record': track_record(log, slug, g),
        'live_accuracy': live_accuracy(log, slug, g, origin.index[0]),
        'holdout'     : {
            'days'    : int(hold_long['origin'].nunique()),
            'mae'     : {int(h): _r(v) for h, v in hold['mae'].items()},
            'persistence_mae': {int(h): _r(v) for h, v in hold['persistence_mae'].items()},
            'directional_acc': {int(h): _r(v, 3) for h, v in hold['directional_acc'].items()},
            'shrink'  : {int(h): w for h, w in model.shrink.items()},
        },
        'training_rows': int(len(feats)),
    }
    metrics = {'holdout_mae': _r(hold_mae, 3),
               'holdout_persistence_mae': _r(hold['persistence_mae'].mean(), 3),
               'data_end': site['data_end']}
    print(f'  {name:<15} today {today_price:6.1f}  holdout MAE {hold_mae:.2f} '
          f'(persistence {hold["persistence_mae"].mean():.2f})  signal {signal["kind"]}')
    return site, metrics, log


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    warnings.filterwarnings('ignore', category=UserWarning)
    now   = datetime.now()
    today = pd.Timestamp(now.date())
    print(f'\n{"=" * 60}\n[{now:%Y-%m-%d %H:%M}] Pipeline started\n{"=" * 60}')

    if not data.download_kalibrate(now.year) and now.month == 1:
        data.download_kalibrate(now.year - 1)   # pick up late-December prices
    gas    = data.load_gas_prices(now.year)
    market = data.fetch_market()

    gas_age    = (today - gas.dropna(how='all').index.max()).days
    market_age = data.market_staleness_days(market, today)
    print(f'  Gas data age: {gas_age} days   Market data age: {market_age} days')
    if gas_age > MAX_GAS_STALENESS_DAYS:
        raise PublishBlocked(f'Kalibrate data is {gas_age} days old (limit {MAX_GAS_STALENESS_DAYS})')
    if market_age > MAX_MARKET_STALENESS_DAYS:
        raise PublishBlocked(f'FRED data is {market_age} days old (limit {MAX_MARKET_STALENESS_DAYS})')

    prev_metrics = _read_json(METRICS_PATH)
    log = load_log()
    sites, metrics = {}, {}
    for slug in gas.columns:
        prev_site = _read_json(os.path.join(SITE_DATA, f'{slug}.json'))
        sites[slug], metrics[slug], log = run_city(slug, gas, market, log, prev_metrics, prev_site, today)

    # Nothing is written until every city has passed its checks
    for slug, site in sites.items():
        _write_json(os.path.join(SITE_DATA, f'{slug}.json'), site)
    index = {
        'generated_at': now.strftime('%Y-%m-%d %H:%M'),
        'default_city': 'montreal',
        'cities': [{'slug': s, 'name': v['name'], 'today_price': v['today_price'],
                    'data_end': v['data_end'], 'signal': v['signal']} for s, v in sites.items()],
    }
    if NTFY_TOPIC:
        index['alerts'] = {'server': NTFY_SERVER, 'topic_prefix': NTFY_TOPIC, 'threshold': SIGNAL_THRESHOLD}
    _write_json(os.path.join(SITE_DATA, 'index.json'), index)
    _write_json(METRICS_PATH, metrics)
    log['made_on']     = pd.to_datetime(log['made_on']).dt.strftime('%Y-%m-%d')
    log['target_date'] = pd.to_datetime(log['target_date']).dt.strftime('%Y-%m-%d')
    log[['base', 'pred', 'low', 'high']] = log[['base', 'pred', 'low', 'high']].astype(float).round(2)
    log.to_csv(LOG_PATH, index=False)

    print(f'\n[{datetime.now():%H:%M}] Pipeline complete — {len(sites)} cities published\n{"=" * 60}\n')


if __name__ == '__main__':
    try:
        run()
    except PublishBlocked as e:
        print(f'\nPUBLISH BLOCKED: {e}\nThe previously published forecast stays live.')
        sys.exit(1)
