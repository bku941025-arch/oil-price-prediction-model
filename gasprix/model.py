"""
model.py — Direct multi-horizon Ridge + LightGBM ensemble, pooled across cities.

One model pair per horizon h predicts the ex-tax price change x[t+h] - x[t] from
features at t, trained on every city at once (city and province are categorical
features for LightGBM). Forecasts never feed on their own predictions. Each horizon's
prediction is shrunk toward "no change" by a weight chosen on an inner validation year:
where the models can't beat persistence, the weight falls to 0.

Prediction intervals and probabilities come from out-of-sample residual quantiles per
horizon, scaled by each city's recent volatility (split-conformal). Quantiles are fitted
separately for calm, mid and volatile cities, so each group's ranges are calibrated.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HORIZONS, scheduled_tax_change

INTERVAL = 0.80
HOLDOUT_DAYS = 365
VALIDATION_DAYS = 365
SHRINK_GRID = np.round(np.arange(0, 1.01, 0.05), 2)
MIN_SCALE = 0.5
QUANTILE_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)

CATEGORICAL = ['city', 'province']
# Levels drift from one regime to the next, so the models only see changes and deviations;
# levels stay in the panel for bookkeeping and the explainer.
NOT_FEATURES = ['gas', 'gas_ex', 'rbob', 'wti_cad', 'margin', 'crack', 'month']
RIDGE_EXCLUDE = CATEGORICAL + ['day_of_week', 'month']


def _lgbm():
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=6,
        min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.8,
        cat_smooth=50, n_jobs=-1, random_state=42, verbose=-1,
    )


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))


def _dates(panel):
    return panel.index.get_level_values(1)


class DirectEnsemble:
    def __init__(self, horizons=HORIZONS):
        self.horizons = horizons
        self.models   = {}
        self.shrink   = {h: 1.0 for h in horizons}
        self.columns  = None
        self.ridge_columns = None

    def _fit_raw(self, panel, Y, last_date):
        models = {}
        d = _dates(panel)
        for h in self.horizons:
            mask = Y[h].notna().values & (d <= last_date - pd.Timedelta(days=h))
            X, y = panel.loc[mask, self.columns], Y[h].values[mask]
            ridge = _ridge().fit(X[self.ridge_columns], y)
            lgbm = _lgbm().fit(X, y, categorical_feature=CATEGORICAL)
            models[h] = (ridge, lgbm)
        return models

    def _raw_predict(self, models, panel):
        X = panel[self.columns]
        return {h: (r.predict(X[self.ridge_columns]) + lg.predict(X)) / 2 for h, (r, lg) in models.items()}

    def fit(self, panel, Y, last_date, shrink=True):
        """Fit on origins whose target date is <= last_date."""
        self.columns = [c for c in panel.columns if c not in NOT_FEATURES]
        self.ridge_columns = [c for c in self.columns if c not in RIDGE_EXCLUDE]
        last_date = pd.Timestamp(last_date)

        if shrink:
            val_start = last_date - pd.Timedelta(days=VALIDATION_DAYS)
            inner = self._fit_raw(panel, Y, val_start)
            val_mask = _dates(panel) > val_start
            val = panel[val_mask]
            raw = self._raw_predict(inner, val)
            vd = _dates(val)
            for h in self.horizons:
                y = Y[h].values[val_mask]
                ok = (vd <= last_date - pd.Timedelta(days=h)) & ~np.isnan(y)
                maes = [np.abs(y[ok] - w * raw[h][ok]).mean() for w in SHRINK_GRID]
                self.shrink[h] = float(SHRINK_GRID[int(np.argmin(maes))])

        self.models = self._fit_raw(panel, Y, last_date)
        return self

    def predict_change(self, panel):
        """DataFrame (rows x horizons) of predicted ex-tax price changes."""
        raw = self._raw_predict(self.models, panel)
        return pd.DataFrame({h: self.shrink[h] * raw[h] for h in self.horizons}, index=panel.index)


def interval_scale(panel):
    """Per-row volatility scale for intervals: recent daily-change std, floored."""
    return np.maximum(panel['gas_std_28'].values, MIN_SCALE)


def long_predictions(model, panel, prices):
    """
    Long DataFrame (city, origin, horizon, target_date, base, pred, actual, scale), where
    pred is the retail price forecast: market change plus scheduled tax changes.
    """
    pred = model.predict_change(panel)
    cities, origins = panel.index.get_level_values(0), _dates(panel)
    base, scale = panel['gas'].values, interval_scale(panel)
    rows = []
    for h in model.horizons:
        tgt = origins + pd.Timedelta(days=h)
        tax = np.empty(len(panel))
        actual = np.full(len(panel), np.nan)
        for slug in pd.unique(cities):
            m = cities == slug
            tax[m] = scheduled_tax_change(slug, origins[m], h)
            actual[m] = prices[slug].reindex(tgt[m]).values
        rows.append(pd.DataFrame({
            'city': cities, 'origin': origins, 'horizon': h, 'target_date': tgt, 'base': base,
            'pred': base + pred[h].values + tax, 'actual': actual, 'scale': scale,
        }))
    return pd.concat(rows, ignore_index=True)


def holdout_evaluation(panel, Y, prices, holdout_days=HOLDOUT_DAYS):
    """
    Train on everything before the last `holdout_days` of origins, predict the holdout.
    Returns (long DataFrame of holdout predictions, the holdout model).
    """
    last = max(prices[c].dropna().index.max() for c in prices.columns)
    split = last - pd.Timedelta(days=holdout_days)
    train = _dates(panel) <= split
    model = DirectEnsemble().fit(panel[train], Y[train], last_date=split)
    long  = long_predictions(model, panel[~train], prices).dropna(subset=['actual'])
    return long, model


def metrics_by_horizon(long):
    """
    MAE / RMSE / directional accuracy for the model and persistence, plus quantiles of
    volatility-scaled residuals used for intervals and probabilities.
    """
    out = []
    for h, d in long.groupby('horizon'):
        err  = d['actual'] - d['pred']
        perr = d['actual'] - d['base']
        # Direction is scored only where the model made a call and the price actually moved
        call = ((d['pred'] - d['base']).abs() > 0.05) & (perr.abs() > 0.05)
        dir_ok = np.sign(d['pred'] - d['base'])[call] == np.sign(perr)[call]
        norm = (err / d['scale']).values
        row = {
            'horizon'        : h,
            'n'              : len(d),
            'mae'            : err.abs().mean(),
            'rmse'           : np.sqrt((err ** 2).mean()),
            'persistence_mae': perr.abs().mean(),
            'directional_acc': dir_ok.mean() if call.any() else np.nan,
        }
        for q, v in zip(QUANTILE_LEVELS, np.quantile(norm, QUANTILE_LEVELS)):
            row[f'q{q:.2f}'] = v
        out.append(row)
    return pd.DataFrame(out).set_index('horizon')


def _q(metrics, level):
    return metrics[f'q{level:.2f}']


VOL_GROUPS = ('calm', 'mid', 'volatile')


def volatility_groups(panel):
    """Assign each city to a volatility tercile by its median recent volatility."""
    med = panel.groupby(level=0)['gas_std_28'].median()
    return pd.qcut(med, 3, labels=VOL_GROUPS).astype(str).to_dict()


class Calibration:
    """Residual quantiles per horizon: one set overall and one per volatility group."""

    def __init__(self, long, groups=None):
        self.overall = metrics_by_horizon(long)
        self.groups = groups or {}
        self.by_group = {}
        if groups:
            g = long['city'].map(groups)
            for name in VOL_GROUPS:
                part = long[g == name]
                if part['origin'].nunique() >= 60:   # enough history to estimate tails
                    self.by_group[name] = metrics_by_horizon(part)

    def rows(self, long):
        """Quantile table aligned to each row of `long` (group-specific when available)."""
        out = self.overall.reindex(long['horizon']).reset_index(drop=True)
        if self.by_group:
            g = long['city'].map(self.groups).reset_index(drop=True)
            for name, m in self.by_group.items():
                mask = (g == name).values
                if mask.any():
                    out.loc[mask, :] = m.reindex(long['horizon'].values[mask]).values
        return out


def _rows(long, calib):
    if isinstance(calib, Calibration):
        return calib.rows(long)
    return calib.reindex(long['horizon']).reset_index(drop=True)


def interval(long, calib, level=INTERVAL):
    """(low, high) arrays for each row of `long` from scaled residual quantiles."""
    lo_q, hi_q = round((1 - level) / 2, 2), round(1 - (1 - level) / 2, 2)
    m = _rows(long, calib)
    return (long['pred'].values + _q(m, lo_q).values * long['scale'].values,
            long['pred'].values + _q(m, hi_q).values * long['scale'].values)


def prob_above(long, calib, threshold):
    """P(actual > threshold) per row, from the empirical scaled-residual distribution."""
    m = _rows(long, calib)
    qs = np.stack([_q(m, lv).values for lv in QUANTILE_LEVELS], axis=1)     # rows x levels
    z = (np.asarray(threshold, dtype=float) - long['pred'].values) / long['scale'].values
    cdf = np.array([np.interp(zi, qrow, QUANTILE_LEVELS, left=0.0, right=1.0) for zi, qrow in zip(z, qs)])
    return np.clip(1 - cdf, 0.02, 0.98)


def coverage(long, calib, level=INTERVAL):
    """Share of actuals inside the intervals implied by the calibration."""
    lo, hi = interval(long, calib, level)
    a = long['actual'].values
    return float(((a >= lo) & (a <= hi)).mean())


def crps(long, calib, center='pred'):
    """
    Continuous ranked probability score (approximated by the mean pinball loss over the
    quantile grid, times 2), in ¢/L; lower is better. center='base' scores a "no change"
    forecast with the same calibrated spread.
    """
    m = _rows(long, calib)
    y, c, sc = long['actual'].values, long[center].values, long['scale'].values
    total = 0.0
    for lv in QUANTILE_LEVELS:
        q = c + _q(m, lv).values * sc
        d = y - q
        total += np.mean(np.maximum(lv * d, (lv - 1) * d))
    return float(2 * total / len(QUANTILE_LEVELS))


def reliability(long, groups=None, levels=(0.5, 0.8, 0.9)):
    """Honest coverage check: calibrate on the first half of `long`'s origins, test on the second."""
    mid = long['origin'].min() + (long['origin'].max() - long['origin'].min()) / 2
    cal, ev = long[long['origin'] <= mid], long[long['origin'] > mid]
    c = Calibration(cal, groups)
    base_cal = metrics_by_horizon(cal.assign(pred=cal['base']))
    return {
        'coverage': {f'{lv:.0%}': round(coverage(ev, c, lv), 3) for lv in levels},
        'crps_model': round(crps(ev, c), 3),
        'crps_persistence': round(crps(ev, base_cal, center='base'), 3),
        'n': int(len(ev)),
    }
