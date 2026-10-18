"""
model.py — Direct multi-horizon Ridge + LightGBM ensemble.

One model pair per horizon h predicts gas[t+h] - gas[t] from features at t, so forecasts
never feed on their own predictions. Each horizon's prediction is then shrunk toward
"no change" by a weight chosen on an inner validation year: where the models can't beat
persistence, the weight falls to 0 and the forecast is persistence.

Prediction intervals come from out-of-sample residual quantiles per horizon
(split-conformal), so an "80% range" is calibrated on data the model never saw.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HORIZONS, targets

INTERVAL = 0.80
HOLDOUT_DAYS = 365


VALIDATION_DAYS = 365
SHRINK_GRID = np.round(np.arange(0, 1.01, 0.05), 2)
MIN_SCALE = 0.5


def _lgbm():
    return LGBMRegressor(
        n_estimators=150, learning_rate=0.03, num_leaves=7, max_depth=5,
        min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1,
    )


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))


class DirectEnsemble:
    def __init__(self, horizons=HORIZONS):
        self.horizons = horizons
        self.models   = {}
        self.shrink   = {h: 1.0 for h in horizons}
        self.columns  = None

    def _fit_raw(self, feats, y_all, last_date):
        models = {}
        for h in self.horizons:
            y = y_all[h]
            mask = y.notna() & (feats.index <= last_date - pd.Timedelta(days=h))
            X, yy = feats.loc[mask, self.columns], y[mask]
            models[h] = (_ridge().fit(X, yy), _lgbm().fit(X, yy))
        return models

    @staticmethod
    def _raw_predict(models, X):
        return {h: (r.predict(X) + l.predict(X)) / 2 for h, (r, l) in models.items()}

    def fit(self, feats, gas, last_date=None, shrink=True):
        """Fit on origins whose target date is <= last_date (default: all known targets)."""
        self.columns = list(feats.columns)
        y_all = targets(gas, feats.index, self.horizons)
        last_date = pd.Timestamp(last_date) if last_date is not None else gas.dropna().index.max()

        if shrink:
            val_start = last_date - pd.Timedelta(days=VALIDATION_DAYS)
            inner = self._fit_raw(feats, y_all, val_start)
            val = feats[(feats.index > val_start)]
            raw = self._raw_predict(inner, val[self.columns])
            for h in self.horizons:
                ok = (val.index <= last_date - pd.Timedelta(days=h)) & y_all[h].reindex(val.index).notna()
                y, p = y_all[h].reindex(val.index)[ok].values, raw[h][ok]
                maes = [np.abs(y - w * p).mean() for w in SHRINK_GRID]
                self.shrink[h] = float(SHRINK_GRID[int(np.argmin(maes))])

        self.models = self._fit_raw(feats, y_all, last_date)
        return self

    def predict_change(self, feats):
        """DataFrame (origins x horizons) of predicted price changes."""
        raw = self._raw_predict(self.models, feats[self.columns])
        return pd.DataFrame({h: self.shrink[h] * raw[h] for h in self.horizons}, index=feats.index)


def interval_scale(feats):
    """Per-origin volatility scale for intervals: recent daily-change std, floored."""
    return np.maximum(feats['gas_std_28'].values, MIN_SCALE)


def long_predictions(model, feats, gas):
    """Long DataFrame (origin, horizon, target_date, base, pred, actual, scale) for the given origins."""
    g = gas.astype(float)
    pred = model.predict_change(feats)
    base, scale = feats['gas'].values, interval_scale(feats)
    rows = []
    for h in model.horizons:
        tgt = feats.index + pd.Timedelta(days=h)
        rows.append(pd.DataFrame({
            'origin': feats.index, 'horizon': h, 'target_date': tgt, 'base': base,
            'pred': base + pred[h].values, 'actual': g.reindex(tgt).values, 'scale': scale,
        }))
    return pd.concat(rows, ignore_index=True)


def holdout_evaluation(feats, gas, holdout_days=HOLDOUT_DAYS):
    """
    Train on everything before the last `holdout_days` of origins, predict the holdout.
    Returns (long DataFrame of holdout predictions, per-horizon metrics DataFrame).
    """
    split = gas.dropna().index.max() - pd.Timedelta(days=holdout_days)
    model = DirectEnsemble().fit(feats[feats.index <= split], gas, last_date=split)
    long  = long_predictions(model, feats[feats.index > split], gas).dropna(subset=['actual'])
    return long, metrics_by_horizon(long)


def metrics_by_horizon(long):
    """
    MAE / RMSE / directional accuracy for the model and persistence, plus quantiles of
    volatility-scaled residuals used for prediction intervals.
    """
    lo_q, hi_q = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    out = []
    for h, d in long.groupby('horizon'):
        err  = d['actual'] - d['pred']
        perr = d['actual'] - d['base']
        # Direction is scored only where the model made a call and the price actually moved
        call = ((d['pred'] - d['base']).abs() > 0.05) & (perr.abs() > 0.05)
        dir_ok = np.sign(d['pred'] - d['base'])[call] == np.sign(perr)[call]
        norm = err / d['scale']
        out.append({
            'horizon'        : h,
            'n'              : len(d),
            'mae'            : err.abs().mean(),
            'rmse'           : np.sqrt((err ** 2).mean()),
            'persistence_mae': perr.abs().mean(),
            'directional_acc': dir_ok.mean() if call.any() else np.nan,
            'q_lo'           : norm.quantile(lo_q),
            'q_hi'           : norm.quantile(hi_q),
        })
    return pd.DataFrame(out).set_index('horizon')


def interval(long, metrics):
    """(low, high) arrays for each row of `long` using metrics' scaled residual quantiles."""
    m = metrics.reindex(long['horizon'])
    return (long['pred'].values + m['q_lo'].values * long['scale'].values,
            long['pred'].values + m['q_hi'].values * long['scale'].values)


def coverage(long, metrics):
    """Share of actuals inside the intervals implied by metrics."""
    lo, hi = interval(long, metrics)
    a = long['actual'].values
    return float(((a >= lo) & (a <= hi)).mean())
