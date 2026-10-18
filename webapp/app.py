"""
app.py — Serves the static site in ../site (used on Render)
------------------------------------------------------------
The daily pipeline precomputes every forecast into site/data/*.json, so this server
only serves files: no models are loaded per request. The same folder can be hosted
on any static host (see .github/workflows/daily_pipeline.yml for GitHub Pages).

Run with:
    python app.py

Then open: http://localhost:5000
"""

import os

from flask import Flask, abort, jsonify, redirect, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'site')
DATA_DIR = os.path.join(SITE_DIR, 'data')

app = Flask(__name__, static_folder=None)

LEGACY_PAGES = {'/budget': 'budget.html', '/fillup': 'fillup.html'}


@app.after_request
def cache_headers(resp):
    # Revalidate every file (cheap with ETags) so pages, scripts and daily data never mismatch
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/')
def index():
    return send_from_directory(SITE_DIR, 'index.html')


@app.route('/budget')
@app.route('/fillup')
def legacy_page():
    query = request.query_string.decode()
    return redirect(LEGACY_PAGES[request.path] + (f'?{query}' if query else ''), code=301)


@app.route('/api/forecast')
def api_forecast():
    """JSON forecast for one city (?city=montreal) — the same file the site reads."""
    city = request.args.get('city', 'montreal')
    if not city.replace('-', '').isalpha() or not os.path.exists(os.path.join(DATA_DIR, f'{city}.json')):
        return jsonify({'error': f'Unknown city: {city}'}), 404
    return send_from_directory(DATA_DIR, f'{city}.json')


@app.route('/<path:path>')
def static_files(path):
    if not os.path.isfile(os.path.join(SITE_DIR, path)):
        abort(404)
    return send_from_directory(SITE_DIR, path)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
