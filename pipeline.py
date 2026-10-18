"""
pipeline.py — Daily data refresh, retrain and forecast publish.
Called by GitHub Actions daily. Also callable manually:
    python pipeline.py

  1. Refresh Kalibrate retail prices (regular, premium, diesel) and market data
     (FRED + Bank of Canada; no API keys needed)
  2. Refuse to publish if the data is stale
  3. Score a fresh pooled model on the most recent year it never saw (holdout)
  4. Refuse to publish if that score is much worse than the last published run
  5. Retrain on all data and forecast 28 days for every city, with calibrated ranges,
     probabilities, driver explanations and what-if scenarios
  6. Log forecasts and signals so they can be scored against actual prices later
  7. Write site/data/*.json and site/feeds/*.xml, send alerts and social posts
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gasprix import data, features, insights, publish
from gasprix.cities import CITIES, PROVINCE_NAMES
from gasprix.model import (INTERVAL, DirectEnsemble, holdout_evaluation, interval,
                           long_predictions, metrics_by_horizon, prob_above)

ROOT         = os.path.dirname(os.path.abspath(__file__))
SITE_DATA    = os.path.join(ROOT, 'site', 'data')
SITE_FEEDS   = os.path.join(ROOT, 'site', 'feeds')
LOG_DIR      = os.path.join(data.DATA_DIR, 'predictions')
SIGNALS_PATH = os.path.join(data.DATA_DIR, 'signals.csv')
METRICS_PATH = os.path.join(data.DATA_DIR, 'metrics.json')

MAX_GAS_STALENESS_DAYS    = 6     # Kalibrate is weekdays only, plus holidays
MAX_MARKET_STALENESS_DAYS = 7     # beyond each series' normal publication lag
MAX_MAE_DEGRADATION       = 1.25  # vs the last published run's holdout MAE
SIGNAL_THRESHOLD          = 2.0   # ¢/L expected move over 3 days that triggers a signal
EXPLAIN_HORIZONS          = [7, 14, 28]
LOG_HORIZONS              = [1, 7, 14, 28]
SCENARIO_SHOCKS           = (-0.2, -0.1, 0.1, 0.2)
COVERAGE_WARN             = (0.70, 0.90)
FORCE_PUBLISH             = os.environ.get('FORCE_PUBLISH') == '1'

LOG_COLUMNS = ['made_on', 'city', 'horizon', 'target_date', 'base', 'pred', 'low', 'high', 'source']


class PublishBlocked(Exception):
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def _r(x, n=1):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, n)


def _read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def load_log():
    """All yearly prediction logs as one DataFrame."""
    frames = []
    if os.path.isdir(LOG_DIR):
        for name in sorted(os.listdir(LOG_DIR)):
            if name.endswith('.csv'):
                frames.append(pd.read_csv(os.path.join(LOG_DIR, name)))
    if not frames:
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.concat(frames, ignore_index=True)
    log['made_on'] = pd.to_datetime(log['made_on'])
    log['target_date'] = pd.to_datetime(log['target_date'])
    return log


def save_log(log):
    os.makedirs(LOG_DIR, exist_ok=True)
    log = log.copy()
    log['made_on'] = pd.to_datetime(log['made_on'])
    log['target_date'] = pd.to_datetime(log['target_date'])
    log[['base', 'pred', 'low', 'high']] = log[['base', 'pred', 'low', 'high']].astype(float).round(2)
    year = log['made_on'].dt.year
    for y, part in log.groupby(year):
        part = part.sort_values(['made_on', 'city', 'horizon'])
        part.assign(made_on=part['made_on'].dt.strftime('%Y-%m-%d'),
                    target_date=part['target_date'].dt.strftime('%Y-%m-%d')
                    ).to_csv(os.path.join(LOG_DIR, f'{y}.csv'), index=False)


def weekly_summary(forecast):
    weeks = []
    for w in range(4):
        days = forecast[w * 7:(w + 1) * 7]
        weeks.append({'week': w + 1, 'date_from': days[0]['date'], 'date_to': days[-1]['date'],
                      'low': _r(min(d['low'] for d in days)), 'mid': _r(np.mean([d['mid'] for d in days])),
                      'high': _r(max(d['high'] for d in days))})
    return weeks


def price_signal(today_price, forecast):
    """'fill_now' if prices are expected to rise over the next 3 days, 'wait' if fall, else 'steady'."""
    change = np.mean([d['mid'] for d in forecast[:3]]) - today_price
    kind = 'fill_now' if change >= SIGNAL_THRESHOLD else 'wait' if change <= -SIGNAL_THRESHOLD else 'steady'
    return {'kind': kind, 'change_3d': _r(change)}


def live_accuracy(log, slug, prices, as_of, days=90):
    """Score logged live forecasts whose target date has an actual price."""
    d = log[(log['source'] == 'live') & ((log['city'] == slug) if slug else True)].copy()
    if d.empty:
        return {'window_days': days, 'since': None, 'n': 0}
    d['actual'] = [prices[c].get(t, np.nan) for c, t in zip(d['city'], d['target_date'])]
    d = d.dropna(subset=['actual'])
    d = d[d['target_date'] > as_of - pd.Timedelta(days=days)]
    out = {'window_days': days, 'since': None, 'n': int(len(d))}
    if d.empty:
        return out
    out['since'] = d['made_on'].min().strftime('%Y-%m-%d')
    for h in [1, 7]:
        dh = d[d['horizon'] == h]
        out[f'mae_{h}'] = _r((dh['actual'] - dh['pred']).abs().mean(), 2) if len(dh) else None
        out[f'persistence_mae_{h}'] = _r((dh['actual'] - dh['base']).abs().mean(), 2) if len(dh) else None
        out[f'n_{h}'] = int(len(dh))
    out['coverage'] = _r(((d['actual'] >= d['low']) & (d['actual'] <= d['high'])).mean(), 3)
    return out


def track_record(log, slug, gas, days=180, horizon=7):
    """Actual prices alongside the forecast made `horizon` days earlier (live, else backtest)."""
    d = log[(log['city'] == slug) & (log['horizon'] == horizon)]
    d = d.sort_values('source').drop_duplicates('target_date', keep='last').set_index('target_date')
    end = gas.dropna().index.max()
    rows = []
    for t in pd.date_range(end - pd.Timedelta(days=days), end, freq='D'):
        rec = {'date': t.strftime('%Y-%m-%d'), 'actual': _r(gas.get(t))}
        if t in d.index:
            rec.update(pred=_r(d.at[t, 'pred']), low=_r(d.at[t, 'low']), high=_r(d.at[t, 'high']),
                       source=d.at[t, 'source'])
        rows.append(rec)
    return rows


def product_forecast(product_prices, slug, origin, reg_forecast, reg_today):
    """
    Premium and diesel follow regular gasoline's market move from today's spread.
    (Diesel's own taxes differ slightly; the excise gap is ignored here.)
    """
    out = {}
    for product, p in product_prices.items():
        if slug not in p.columns:
            continue
        s = p[slug].dropna()
        s = s[s.index <= origin]
        if s.empty or (origin - s.index.max()).days > 7:
            continue
        spread = float(s.iloc[-1]) - reg_today
        out[product] = {'today_price': _r(s.iloc[-1]), 'spread': _r(spread),
                        'forecast': [{'date': f['date'], 'low': _r(f['low'] + spread),
                                      'mid': _r(f['mid'] + spread), 'high': _r(f['high'] + spread)}
                                     for f in reg_forecast]}
    return out


def history_payload(gas):
    s = gas.dropna()
    return {'start': s.index.min().strftime('%Y-%m-%d'),
            'prices': [_r(v) for v in s.reindex(pd.date_range(s.index.min(), s.index.max())).values]}


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    warnings.filterwarnings('ignore', category=UserWarning)
    now   = datetime.now()
    today = pd.Timestamp(now.date())
    print(f'\n{"=" * 60}\n[{now:%Y-%m-%d %H:%M}] Pipeline started\n{"=" * 60}')

    # 1. Data
    years = [now.year - 1, now.year] if now.month == 1 else [now.year]
    for y in years:
        data.download_kalibrate(y, 'regular')
    for product in ('premium', 'diesel'):
        for y in (now.year - 1, now.year):
            data.download_kalibrate(y, product)
    prices = data.load_prices('regular', now.year)
    product_prices = {p: data.load_prices(p, now.year) for p in ('premium', 'diesel')}
    market = data.fetch_market()

    # 2. Staleness
    last_seen = prices.apply(lambda s: s.last_valid_index())
    gas_age = (today - last_seen.median()).days
    market_age = data.market_staleness_days(market, today)
    print(f'  Gas data age: {gas_age} days   Market data age beyond normal lag: {market_age} days')
    if gas_age > MAX_GAS_STALENESS_DAYS:
        raise PublishBlocked(f'Kalibrate data is {gas_age} days old (limit {MAX_GAS_STALENESS_DAYS})')
    if market_age > MAX_MARKET_STALENESS_DAYS:
        raise PublishBlocked(f'Market data is {market_age} days late (limit {MAX_MARKET_STALENESS_DAYS})')
    stale = [s for s, d in last_seen.items() if (today - d).days > MAX_GAS_STALENESS_DAYS]
    if stale:
        print(f'  Skipping {len(stale)} cities with stale data: {", ".join(stale)}')
        prices = prices.drop(columns=stale)

    aligned = data.align_market(market, prices.index)
    panel = features.build_panel(prices, aligned)
    ex = pd.DataFrame({s: features.ex_tax(prices[s].dropna(), s) for s in prices.columns})
    Y = features.targets(ex, panel.index)
    print(f'  Panel: {len(panel):,} rows, {panel.index.get_level_values(0).nunique()} cities')

    # 3–4. Holdout score and gate
    hold_long, _ = holdout_evaluation(panel, Y, prices)
    hold = metrics_by_horizon(hold_long)
    hold_mae = float(hold['mae'].mean())
    prev = _read_json(METRICS_PATH)
    print(f'  Holdout MAE {hold_mae:.3f} (persistence {hold["persistence_mae"].mean():.3f})')
    if prev.get('holdout_mae') and hold_mae > prev['holdout_mae'] * MAX_MAE_DEGRADATION and not FORCE_PUBLISH:
        raise PublishBlocked(f'Holdout MAE {hold_mae:.2f} is >{MAX_MAE_DEGRADATION:.0%} of last run '
                             f'({prev["holdout_mae"]:.2f}). Set FORCE_PUBLISH=1 to override.')

    # 5. Final model and forecasts from each city's latest day
    model = DirectEnsemble().fit(panel, Y, last_date=max(last_seen[prices.columns]))
    latest = panel.groupby(level=0).tail(1)
    fc = long_predictions(model, latest, prices)
    lo, hi = interval(fc, hold)
    fc['low'], fc['high'] = lo, hi
    fc['prob_up'] = prob_above(fc, hold, fc['base'].values)
    base_change = model.predict_change(latest)          # ex-tax market change, rows x horizons
    tax_by_h = {h: fc.loc[fc['horizon'] == h, 'pred'].values - fc.loc[fc['horizon'] == h, 'base'].values
                - base_change[h].values for h in EXPLAIN_HORIZONS}
    explained = insights.explain(model, latest, EXPLAIN_HORIZONS, tax_by_h)
    # What-if: the forecast plus how much the model's market change shifts under each shock
    scen = {s: v - base_change for s, v in insights.scenarios(model, latest, SCENARIO_SHOCKS).items()}

    # 6. Logs
    log = load_log()
    log = log[log['made_on'] != today]
    if not (log['source'] == 'backtest').any():
        b_lo, b_hi = interval(hold_long, hold)
        seed = hold_long.assign(made_on=hold_long['origin'], low=b_lo, high=b_hi, source='backtest')
        log = pd.concat([seed.loc[seed['horizon'] == 7, LOG_COLUMNS], log], ignore_index=True)
    new = fc[fc['horizon'].isin(LOG_HORIZONS)].assign(made_on=today, source='live')[LOG_COLUMNS]
    log = pd.concat([log, new], ignore_index=True)

    # 7. Per-city payloads
    hold_city = {c: metrics_by_horizon(d) for c, d in hold_long.groupby('city')}
    prev_index = {c['slug']: c for c in _read_json(os.path.join(SITE_DATA, 'index.json')).get('cities', [])}
    signals_rows, sites = [], {}
    for slug, f in fc.groupby('city', sort=False):
        city = CITIES[slug]
        f = f.sort_values('horizon')
        g = prices[slug].dropna()
        origin = f['origin'].iloc[0]
        row = latest.loc[slug].iloc[0]
        today_price = float(f['base'].iloc[0])
        forecast = [{'date': t.strftime('%Y-%m-%d'), 'horizon': int(h), 'low': _r(l), 'mid': _r(m),
                     'high': _r(u), 'prob_up': _r(p, 2)}
                    for t, h, m, l, u, p in zip(f['target_date'], f['horizon'], f['pred'], f['low'],
                                                f['high'], f['prob_up'])]
        signal = price_signal(today_price, forecast)
        hc = hold_city.get(slug)
        city_rows = panel.loc[slug]
        site = {
            'city': slug, 'name': city.name, 'province': city.province,
            'generated_at': now.strftime('%Y-%m-%d %H:%M'), 'data_end': origin.strftime('%Y-%m-%d'),
            'today_price': _r(today_price), 'interval': INTERVAL,
            'forecast': forecast, 'weekly': weekly_summary(forecast), 'signal': signal,
            'historical': [{'date': t.strftime('%Y-%m-%d'), 'price': _r(p)} for t, p in g.iloc[-120:].items()],
            'explain': {int(h): {k: _r(v, 2) for k, v in explained[h].loc[(slug, origin)].items()}
                        for h in EXPLAIN_HORIZONS},
            'scenarios': {f'{s:+.0%}': [_r(p + d) for p, d in zip(f['pred'].values, scen[s].loc[(slug, origin)].values)]
                          for s in SCENARIO_SHOCKS},
            'margin_gauge': insights.margin_gauge(row, city_rows['margin']),
            'products': product_forecast(product_prices, slug, origin, forecast, today_price),
            'track_record': track_record(log, slug, g),
            'live_accuracy': live_accuracy(log, slug, prices, origin),
            'holdout': None if hc is None else {
                'mae': {int(h): _r(v, 2) for h, v in hc['mae'].items()},
                'persistence_mae': {int(h): _r(v, 2) for h, v in hc['persistence_mae'].items()},
                'directional_acc': {int(h): _r(v, 3) for h, v in hc['directional_acc'].items()},
            },
        }
        sites[slug] = site
        signals_rows.append({'date': origin.strftime('%Y-%m-%d'), 'city': slug, 'kind': signal['kind'],
                             'change_3d': signal['change_3d'], 'price': site['today_price'],
                             'forecast_7': forecast[6]['mid']})

    # Nothing is written until every check has passed
    for slug, site in sites.items():
        _write_json(os.path.join(SITE_DATA, f'{slug}.json'), site)
        _write_json(os.path.join(SITE_DATA, 'history', f'{slug}.json'), {
            **history_payload(prices[slug]),
            'weekday': insights.weekday_effect(prices[slug]),
            'breakdown': insights.breakdown_series(prices[slug], aligned, slug),
        })

    national = np.mean([s['today_price'] for s in sites.values()])
    live_all = live_accuracy(log, None, prices, today)
    warnings_out = []
    if live_all['n'] >= 200 and live_all.get('coverage') is not None and not (
            COVERAGE_WARN[0] <= live_all['coverage'] <= COVERAGE_WARN[1]):
        warnings_out.append(f"Live 80% range coverage is {live_all['coverage']:.0%} over the last "
                            f"{live_all['window_days']} days; recalibration may be needed.")
        print(f'  WARNING: {warnings_out[-1]}')
    index = {
        'generated_at': now.strftime('%Y-%m-%d %H:%M'),
        'default_city': 'montreal',
        'national_avg': _r(national),
        'provinces': {k: {'en': v[0], 'fr': v[1]} for k, v in PROVINCE_NAMES.items()},
        'cities': [{**CITIES[s].to_dict(), 'today_price': v['today_price'], 'data_end': v['data_end'],
                    'signal': v['signal'], 'change_7d': _r(v['forecast'][6]['mid'] - v['today_price']),
                    'prob_up_7': v['forecast'][6]['prob_up'], 'products': sorted(v['products'])}
                   for s, v in sites.items()],
        'model': {
            'holdout_mae': _r(hold_mae, 3), 'holdout_persistence_mae': _r(hold['persistence_mae'].mean(), 3),
            'holdout_mae_by_h': {int(h): _r(v, 2) for h, v in hold['mae'].items()},
            'holdout_persistence_by_h': {int(h): _r(v, 2) for h, v in hold['persistence_mae'].items()},
            'shrink': {int(h): w for h, w in model.shrink.items()},
            'training_rows': int(len(panel)), 'live_accuracy': live_all,
        },
        'warnings': warnings_out,
    }
    if publish.NTFY_TOPIC:
        index['alerts'] = {'server': publish.NTFY_SERVER, 'topic_prefix': publish.NTFY_TOPIC,
                           'threshold': SIGNAL_THRESHOLD}
    _write_json(os.path.join(SITE_DATA, 'index.json'), index)
    _write_json(METRICS_PATH, {'holdout_mae': index['model']['holdout_mae'],
                               'holdout_persistence_mae': index['model']['holdout_persistence_mae'],
                               'generated_at': index['generated_at']})
    save_log(log)

    signals = pd.read_csv(SIGNALS_PATH) if os.path.exists(SIGNALS_PATH) else pd.DataFrame()
    new_sig = pd.DataFrame(signals_rows)
    if not signals.empty:
        signals = signals[~signals.set_index(['date', 'city']).index.isin(new_sig.set_index(['date', 'city']).index)]
    signals = pd.concat([signals, new_sig], ignore_index=True).sort_values(['date', 'city'])
    signals.to_csv(SIGNALS_PATH, index=False)
    write_feeds(signals, sites)

    # Outbound channels (each is a no-op unless configured)
    sent = sum(publish.ntfy(site, prev_index.get(slug, {}).get('signal', {}).get('kind'))
               for slug, site in sites.items())
    movers = sorted(index['cities'], key=lambda c: -abs(c['change_7d'] or 0))[:3]
    text = publish.social_text(sites.get(index['default_city'], next(iter(sites.values()))), movers)
    posted = [name for name, fn in (('Bluesky', publish.post_bluesky), ('Mastodon', publish.post_mastodon)) if fn(text)]
    print(f'  Alerts sent: {sent}   Social: {", ".join(posted) or "none configured"}')
    print(f'\n[{datetime.now():%H:%M}] Pipeline complete — {len(sites)} cities published\n{"=" * 60}\n')


def write_feeds(signals, sites):
    s = signals.copy()
    s['date'] = pd.to_datetime(s['date'])
    s = s[s['date'] > s['date'].max() - pd.Timedelta(days=30)]

    def items(rows):
        out = []
        for r in rows.sort_values('date', ascending=False).itertuples():
            site = sites.get(r.city)
            name = site['name'] if site else CITIES[r.city].name
            out.append({
                'title': f'{name} — {r.date:%Y-%m-%d}: {publish.SIGNAL_TEXT[r.kind][0]}',
                'link': f'{publish.SITE_URL}/?city={r.city}',
                'description': f'{name}: {r.price:.1f} ¢/L. {publish.SIGNAL_TEXT[r.kind][0]} '
                               f'({r.change_3d:+.1f} ¢/L over 3 days; 7-day forecast {r.forecast_7:.1f}).',
                'guid': f'{r.city}-{r.date:%Y-%m-%d}',
                'date': r.date.to_pydatetime().replace(tzinfo=timezone.utc),
            })
        return out

    for slug in sites:
        _write_text(os.path.join(SITE_FEEDS, f'{slug}.xml'), publish.rss(
            f'GasPrix — {sites[slug]["name"]}', f'{publish.SITE_URL}/?city={slug}',
            f'Daily gas price forecast for {sites[slug]["name"]}', items(s[s['city'] == slug])))
    changes = s[s['kind'] != 'steady']
    _write_text(os.path.join(SITE_FEEDS, 'signals.xml'), publish.rss(
        'GasPrix — price signals', publish.SITE_URL,
        'Cities where prices are expected to move more than 2 ¢/L within 3 days', items(changes)))


if __name__ == '__main__':
    try:
        run()
    except PublishBlocked as e:
        print(f'\nPUBLISH BLOCKED: {e}\nThe previously published forecast stays live.')
        sys.exit(1)
