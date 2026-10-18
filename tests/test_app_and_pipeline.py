"""Web server routes and pipeline helpers (no network access)."""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'webapp'))

import app as webapp  # noqa: E402

import pipeline  # noqa: E402
from gasprix import publish  # noqa: E402


@pytest.fixture
def client():
    return webapp.app.test_client()


@pytest.mark.parametrize('path', ['/', '/index.html', '/map.html', '/fillup.html', '/budget.html', '/trip.html',
                                  '/compare.html', '/ev.html', '/breakdown.html', '/history.html',
                                  '/accuracy.html', '/api.html', '/widget.html', '/assets/app.js',
                                  '/roadtrip.html', '/mylog.html', '/model.html', '/c/montreal.html',
                                  '/manifest.webmanifest', '/sw.js'])
def test_pages_are_served(client, path):
    assert client.get(path).status_code == 200


def test_legacy_routes_redirect(client):
    r = client.get('/fillup?lang=fr')
    assert r.status_code == 301 and r.headers['Location'].endswith('fillup.html?lang=fr')


@pytest.mark.parametrize('path', ['/../pipeline.py', '/%2e%2e/pipeline.py', '/nope.html'])
def test_no_files_outside_site(client, path):
    assert client.get(path).status_code == 404


def test_api_rejects_unknown_city(client):
    assert client.get('/api/forecast?city=../pipeline').status_code == 404


def test_price_signal_thresholds():
    def fc(d):
        return [{'mid': 190 + d}] * 3

    assert pipeline.price_signal(190, fc(3))['kind'] == 'fill_now'
    assert pipeline.price_signal(190, fc(-3))['kind'] == 'wait'
    assert pipeline.price_signal(190, fc(1))['kind'] == 'steady'


def test_weekly_summary_covers_four_weeks():
    dates = pd.date_range('2026-01-01', periods=28)
    fc = [{'date': d.strftime('%Y-%m-%d'), 'low': 180.0, 'mid': 185.0 + i * 0.1, 'high': 190.0}
          for i, d in enumerate(dates)]
    weeks = pipeline.weekly_summary(fc)
    assert [w['week'] for w in weeks] == [1, 2, 3, 4]
    assert weeks[0]['date_from'] == '2026-01-01' and weeks[-1]['date_to'] == '2026-01-28'


def test_rss_is_escaped():
    xml = publish.rss('A & B', 'https://x', 'd', [{'title': '<t>', 'link': 'https://x/?a=1&b=2',
                                                   'description': 'x', 'guid': 'g',
                                                   'date': pd.Timestamp('2026-01-01', tz='UTC').to_pydatetime()}])
    assert '&lt;t&gt;' in xml and 'a=1&amp;b=2' in xml and 'A &amp; B' in xml


def test_social_text_fits_in_a_post():
    site = {'name': 'Montréal', 'today_price': 190.0, 'signal': {'kind': 'steady', 'change_3d': 0.4},
            'forecast': [{'mid': 189.0}] * 7}
    movers = [{'name': 'X' * 60, 'change_7d': 4.2}] * 3
    assert len(publish.social_text(site, movers)) <= 300
