"""
publish.py — Outbound channels: RSS feeds, ntfy alerts, Bluesky and Mastodon posts.

Every channel is off unless its environment variables are set:
  NTFY_TOPIC (+ NTFY_SERVER)            push alerts when a city's signal changes
  BSKY_HANDLE + BSKY_APP_PASSWORD       daily Bluesky post
  MASTODON_URL + MASTODON_TOKEN         daily Mastodon post
  SITE_URL                              absolute links in feeds and posts
"""

import os
from datetime import UTC, datetime
from email.utils import format_datetime
from xml.sax.saxutils import escape

import requests

NTFY_TOPIC  = os.environ.get('NTFY_TOPIC', '')
NTFY_SERVER = os.environ.get('NTFY_SERVER', 'https://ntfy.sh')
SITE_URL    = os.environ.get('SITE_URL', 'https://gas-price-mtl.onrender.com').rstrip('/')
BSKY_HANDLE, BSKY_PASSWORD = os.environ.get('BSKY_HANDLE', ''), os.environ.get('BSKY_APP_PASSWORD', '')
MASTODON_URL, MASTODON_TOKEN = os.environ.get('MASTODON_URL', '').rstrip('/'), os.environ.get('MASTODON_TOKEN', '')

SIGNAL_TEXT = {
    'fill_now': ('Prices expected to rise — fill up soon', 'Hausse prévue — faites le plein bientôt'),
    'wait'    : ('Prices expected to drop — consider waiting', 'Baisse prévue — attendez si possible'),
    'steady'  : ('Prices expected to hold steady', 'Prix stables prévus'),
}


def summary_line(site):
    s = site['signal']
    return (f"{site['name']}: {site['today_price']:.1f} ¢/L. {SIGNAL_TEXT[s['kind']][0]} "
            f"({s['change_3d']:+.1f} ¢/L over 3 days; 7-day forecast {site['forecast'][6]['mid']:.1f}).")


# ── RSS ───────────────────────────────────────────────────────────────────────
def rss(title, link, description, items):
    """items: [{title, link, description, guid, date (datetime)}] → RSS 2.0 XML string."""
    body = ''.join(
        f"<item><title>{escape(i['title'])}</title><link>{escape(i['link'])}</link>"
        f"<description>{escape(i['description'])}</description>"
        f"<guid isPermaLink=\"false\">{escape(i['guid'])}</guid>"
        f"<pubDate>{format_datetime(i['date'])}</pubDate></item>"
        for i in items)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
            f'<title>{escape(title)}</title><link>{escape(link)}</link>'
            f'<description>{escape(description)}</description>'
            f'<lastBuildDate>{format_datetime(datetime.now(UTC))}</lastBuildDate>'
            f'{body}</channel></rss>\n')


def city_feed(site, previous_items=()):
    """Daily item for one city, keeping the last 30 days of items."""
    now = datetime.now(UTC)
    item = {'title': f"{site['name']} — {site['data_end']}: {SIGNAL_TEXT[site['signal']['kind']][0]}",
            'link': f"{SITE_URL}/?city={site['city']}", 'description': summary_line(site),
            'guid': f"{site['city']}-{site['data_end']}", 'date': now}
    items = [item] + [i for i in previous_items if i['guid'] != item['guid']][:29]
    return items


# ── Push alerts ───────────────────────────────────────────────────────────────
def ntfy(site, previous_signal):
    """Push an alert when a city's signal changes to fill_now / wait."""
    s = site['signal']
    if not NTFY_TOPIC or s['kind'] == 'steady' or s['kind'] == previous_signal:
        return False
    en, fr = SIGNAL_TEXT[s['kind']]
    try:
        requests.post(NTFY_SERVER, json={
            'topic': f"{NTFY_TOPIC}-{site['city']}", 'title': f"{site['name']}: {en}",
            'message': f"{s['change_3d']:+.1f} ¢/L expected over 3 days (now {site['today_price']:.1f} ¢/L). / {fr}.",
            'click': f"{SITE_URL}/?city={site['city']}", 'tags': ['fuelpump'],
        }, timeout=20).raise_for_status()
        return True
    except Exception as e:
        print(f"  WARNING: ntfy alert for {site['name']} failed: {e}")
        return False


# ── Social ────────────────────────────────────────────────────────────────────
def social_text(main_site, movers):
    lines = [summary_line(main_site)]
    if movers:
        lines.append('Biggest expected moves this week: ' + ', '.join(
            f"{m['name']} {m['change_7d']:+.1f}¢" for m in movers) + '.')
    lines.append(SITE_URL)
    text = ' '.join(lines)
    return text[:297] + '…' if len(text) > 300 else text


def post_bluesky(text):
    if not (BSKY_HANDLE and BSKY_PASSWORD):
        return False
    try:
        s = requests.post('https://bsky.social/xrpc/com.atproto.server.createSession',
                          json={'identifier': BSKY_HANDLE, 'password': BSKY_PASSWORD}, timeout=20)
        s.raise_for_status()
        sess = s.json()
        url_start = text.find(SITE_URL)
        record = {'$type': 'app.bsky.feed.post', 'text': text,
                  'createdAt': datetime.now(UTC).isoformat().replace('+00:00', 'Z')}
        if url_start >= 0:   # make the link clickable (facets use UTF-8 byte offsets)
            b0 = len(text[:url_start].encode())
            record['facets'] = [{'index': {'byteStart': b0, 'byteEnd': b0 + len(SITE_URL.encode())},
                                 'features': [{'$type': 'app.bsky.richtext.facet#link', 'uri': SITE_URL}]}]
        requests.post('https://bsky.social/xrpc/com.atproto.repo.createRecord',
                      headers={'Authorization': f"Bearer {sess['accessJwt']}"},
                      json={'repo': sess['did'], 'collection': 'app.bsky.feed.post', 'record': record},
                      timeout=20).raise_for_status()
        return True
    except Exception as e:
        print(f'  WARNING: Bluesky post failed: {e}')
        return False


def post_mastodon(text):
    if not (MASTODON_URL and MASTODON_TOKEN):
        return False
    try:
        requests.post(f'{MASTODON_URL}/api/v1/statuses', data={'status': text},
                      headers={'Authorization': f'Bearer {MASTODON_TOKEN}'}, timeout=20).raise_for_status()
        return True
    except Exception as e:
        print(f'  WARNING: Mastodon post failed: {e}')
        return False
