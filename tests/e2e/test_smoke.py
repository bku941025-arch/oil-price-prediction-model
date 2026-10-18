"""Browser smoke test: every page loads its data and renders without console errors."""

import os
import socket
import sys
import threading

import pytest

playwright = pytest.importorskip('playwright.sync_api')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'webapp'))

PAGES = ['index.html', 'map.html', 'fillup.html', 'budget.html', 'trip.html', 'compare.html', 'ev.html',
         'roadtrip.html', 'mylog.html', 'breakdown.html', 'history.html', 'accuracy.html', 'model.html',
         'api.html', 'widget.html']
# External resources (fonts, map tiles, CDN scripts) may be unreachable in CI; only our own errors count.
EXTERNAL = ('fonts.googleapis', 'fonts.gstatic', 'tile.openstreetmap', 'unpkg.com', 'cdn.jsdelivr')


@pytest.fixture(scope='module')
def base_url():
    import app as webapp
    from werkzeug.serving import make_server
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    server = make_server('127.0.0.1', port, webapp.app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{port}'
    server.shutdown()


@pytest.fixture(scope='module')
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.mark.parametrize('page_name', PAGES)
@pytest.mark.parametrize('lang', ['en', 'fr'])
def test_page_renders(browser, base_url, page_name, lang):
    page = browser.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: m.type == 'error' and not any(x in m.text for x in EXTERNAL)
            and errors.append(m.text))
    page.goto(f'{base_url}/{page_name}?lang={lang}&city=montreal', wait_until='networkidle')
    page.wait_for_timeout(500)
    assert not errors, errors
    assert page.locator('.error-msg').count() == 0
    if page_name != 'widget.html':
        assert page.locator('nav.navbar').count() == 1
        assert page.locator('html').get_attribute('lang') == lang
    page.close()
