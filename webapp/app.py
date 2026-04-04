"""
app.py — Flask backend for Montréal Gas Price web app
------------------------------------------------------
Run with:
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # .../webapp/
PROJECT_DIR   = os.path.dirname(BASE_DIR)                    # .../oil-price-prediction-model/
DATA_PATH     = os.path.join(PROJECT_DIR, 'Data', 'montreal_gas_ml_ready.csv')
RIDGE_PATH    = os.path.join(BASE_DIR, 'model_ridge.pkl')
LGBM_PATH     = os.path.join(BASE_DIR, 'model_lgbm.pkl')
SCALER_PATH   = os.path.join(BASE_DIR, 'scaler.pkl')
LOG_PATH      = os.path.join(BASE_DIR, 'retrain_log.json')

# ── Feature config (must match Notebook 3) ────────────────────────────────────
EXCLUDE  = ['gas_price', 'wti_usd', 'cadusd', 'wti_cad', 'target_price', 'target_change']


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_models():
    ridge  = joblib.load(RIDGE_PATH)
    lgbm   = joblib.load(LGBM_PATH)
    scaler = joblib.load(SCALER_PATH)
    return ridge, lgbm, scaler


def load_data():
    df = pd.read_csv(DATA_PATH)
    # The date column may be named 'date' or 'Unnamed: 0' depending on how it was saved
    date_col = 'date' if 'date' in df.columns else 'Unnamed: 0'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'date'
    return df


def generate_forecast(df, ridge, lgbm, scaler, n_days=28):
    """Generate n_days recursive ensemble forecast with confidence bands."""
    FEATURES   = [c for c in df.columns if c not in EXCLUDE]
    recent_std = df['gas_price'].iloc[-30:].std()

    forecast_rows = []
    current_df    = df.copy()

    for step in range(1, n_days + 1):
        latest = current_df.iloc[[-1]]
        X      = latest[FEATURES]

        pred_ridge    = ridge.predict(scaler.transform(X))[0]
        pred_lgbm     = lgbm.predict(X)[0]
        pred_ensemble = (pred_ridge + pred_lgbm) / 2

        band      = 1.5 * recent_std * np.sqrt(step)
        pred_low  = round(pred_ensemble - band, 2)
        pred_mid  = round(pred_ensemble, 2)
        pred_high = round(pred_ensemble + band, 2)

        forecast_date = latest.index[0] + timedelta(days=1)

        forecast_rows.append({
            'date'     : forecast_date.strftime('%Y-%m-%d'),
            'pred_low' : pred_low,
            'pred_mid' : pred_mid,
            'pred_high': pred_high,
            'week'     : ((step - 1) // 7) + 1
        })

        # Roll lags forward
        new_row = latest.copy()
        new_row.index              = [forecast_date]
        new_row['gas_lag_21']      = new_row['gas_lag_10']
        new_row['gas_lag_10']      = new_row['gas_lag_5']
        new_row['gas_lag_5']       = new_row['gas_lag_3']
        new_row['gas_lag_3']       = new_row['gas_lag_2']
        new_row['gas_lag_2']       = new_row['gas_lag_1']
        new_row['gas_lag_1']       = pred_ensemble
        new_row['gas_chg_1']       = pred_ensemble - float(latest['gas_lag_1'].iloc[0])
        new_row['day_of_week']     = forecast_date.dayofweek
        new_row['month']           = forecast_date.month
        new_row['gas_roll_mean_5']  = (float(latest['gas_roll_mean_5'].iloc[0])  * 4  + pred_ensemble) / 5
        new_row['gas_roll_mean_10'] = (float(latest['gas_roll_mean_10'].iloc[0]) * 9  + pred_ensemble) / 10
        new_row['gas_roll_mean_21'] = (float(latest['gas_roll_mean_21'].iloc[0]) * 20 + pred_ensemble) / 21

        current_df = pd.concat([current_df, new_row])

    return forecast_rows


def get_weekly_summary(forecast_rows):
    """Aggregate daily forecast into 4 weekly summaries."""
    weeks = {}
    for row in forecast_rows:
        w = row['week']
        if w not in weeks:
            weeks[w] = {'dates': [], 'lows': [], 'mids': [], 'highs': []}
        weeks[w]['dates'].append(row['date'])
        weeks[w]['lows'].append(row['pred_low'])
        weeks[w]['mids'].append(row['pred_mid'])
        weeks[w]['highs'].append(row['pred_high'])

    summary = []
    for w, data in weeks.items():
        summary.append({
            'week'     : w,
            'date_from': data['dates'][0],
            'date_to'  : data['dates'][-1],
            'low'      : round(min(data['lows']), 1),
            'mid'      : round(sum(data['mids']) / len(data['mids']), 1),
            'high'     : round(max(data['highs']), 1)
        })
    return summary


def get_retrain_status():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return None





# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    lang = request.args.get('lang', 'en')
    return render_template('index.html', lang=lang)


@app.route('/budget')
def budget():
    lang = request.args.get('lang', 'en')
    return render_template('budget.html', lang=lang)


@app.route('/fillup')
def fillup():
    lang = request.args.get('lang', 'en')
    return render_template('fillup.html', lang=lang)


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.route('/api/forecast')
def api_forecast():
    try:
        df              = load_data()
        ridge, lgbm, scaler = load_models()
        forecast        = generate_forecast(df, ridge, lgbm, scaler)
        weekly          = get_weekly_summary(forecast)
        today_price     = float(df['gas_price'].iloc[-1])
        today_date      = df.index[-1].strftime('%Y-%m-%d')
        retrain_status  = get_retrain_status()

        # Historical prices for chart context (last 30 days)
        historical = [
            {'date': d.strftime('%Y-%m-%d'), 'price': round(float(p), 2)}
            for d, p in df['gas_price'].iloc[-30:].items()
        ]

        return jsonify({
            'today_date'    : today_date,
            'today_price'   : today_price,
            'forecast'      : forecast,
            'weekly'        : weekly,
            'historical'    : historical,
            'retrain_status': retrain_status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/budget', methods=['POST'])
def api_budget():
    try:
        data             = request.get_json()
        last_month_spend = float(data.get('last_month_spend', 0))

        if last_month_spend <= 0:
            return jsonify({'error': 'Invalid spend amount'}), 400

        df                   = load_data()
        ridge, lgbm, scaler  = load_models()
        forecast             = generate_forecast(df, ridge, lgbm, scaler)

        # Last month's average price
        today            = df.index[-1]
        last_month_start = today - pd.DateOffset(months=1)
        last_month_data  = df[df.index >= last_month_start]['gas_price']
        if len(last_month_data) == 0:
            last_month_data = df['gas_price'].iloc[-30:]
        last_month_avg = float(last_month_data.mean())

        # Implied consumption
        implied_litres = last_month_spend / (last_month_avg / 100)

        # Forecast price range
        forecast_low_avg  = sum(r['pred_low']  for r in forecast) / len(forecast)
        forecast_mid_avg  = sum(r['pred_mid']  for r in forecast) / len(forecast)
        forecast_high_avg = sum(r['pred_high'] for r in forecast) / len(forecast)

        budget_low  = round(implied_litres * (forecast_low_avg  / 100), 2)
        budget_mid  = round(implied_litres * (forecast_mid_avg  / 100), 2)
        budget_high = round(implied_litres * (forecast_high_avg / 100), 2)

        return jsonify({
            'last_month_spend'   : last_month_spend,
            'last_month_avg'     : round(last_month_avg, 1),
            'implied_litres'     : round(implied_litres, 1),
            'forecast_low_avg'   : round(forecast_low_avg, 1),
            'forecast_high_avg'  : round(forecast_high_avg, 1),
            'budget_optimistic'  : budget_low,
            'budget_expected'    : budget_mid,
            'budget_conservative': budget_high
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fillup')
def api_fillup():
    try:
        df                  = load_data()
        ridge, lgbm, scaler = load_models()
        forecast            = generate_forecast(df, ridge, lgbm, scaler, n_days=7)

        today_price = float(df['gas_price'].iloc[-1])

        # Find best and worst day in the 7-day forecast
        best_day  = min(forecast, key=lambda x: x['pred_mid'])
        worst_day = max(forecast, key=lambda x: x['pred_mid'])

        # Savings estimate based on a 50L tank
        TANK_SIZE    = 50
        savings_low  = round((worst_day['pred_mid'] - best_day['pred_mid']) / 100 * TANK_SIZE, 2)
        savings_high = round((worst_day['pred_high'] - best_day['pred_low']) / 100 * TANK_SIZE, 2)

        # Add direction vs today to each forecast day
        for day in forecast:
            day['vs_today'] = round(day['pred_mid'] - today_price, 2)

        return app.response_class(
            response=json.dumps({
                'today_price' : today_price,
                'today_date'  : df.index[-1].strftime('%Y-%m-%d'),
                'forecast'    : forecast,
                'best_day'    : best_day,
                'worst_day'   : worst_day,
                'savings_low' : savings_low,
                'savings_high': savings_high,
                'tank_size'   : TANK_SIZE
            }, allow_nan=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrain-status')
def api_retrain_status():
    status = get_retrain_status()
    if status:
        return jsonify(status)
    return jsonify({'error': 'No retrain log found'}), 404


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
