"""
make_share_cards.py — Link-preview pages and images for each city.

    python scripts/make_share_cards.py

Writes site/c/<city>.html (Open Graph tags + redirect to the forecast) and
site/share/<city>.png (1200x630). Cards contain no prices, so they only need
regenerating when cities or branding change. Needs Pillow (requirements-dev.txt).
"""

import os
import sys
from html import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gasprix.cities import CITIES, PROVINCE_NAMES  # noqa: E402

SITE_URL = os.environ.get('SITE_URL', 'https://gas-price-mtl.onrender.com').rstrip('/')
ACCENT, ORANGE, BG = (44, 95, 138), (224, 123, 57), (250, 250, 250)


def font(size, bold=False):
    for name in (['DejaVuSans-Bold.ttf', 'Arial Bold.ttf', 'Helvetica.ttc'] if bold
                 else ['DejaVuSans.ttf', 'Arial.ttf', 'Helvetica.ttc']):
        for d in ('/usr/share/fonts/truetype/dejavu', '/Library/Fonts', '/System/Library/Fonts',
                  '/System/Library/Fonts/Supplemental'):
            path = os.path.join(d, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def card(city):
    im = Image.new('RGB', (1200, 630), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1200, 14], fill=ACCENT)
    icon = Image.open(os.path.join(ROOT, 'site', 'assets', 'icon-192.png')).convert('RGBA').resize((120, 120))
    im.paste(icon, (80, 90), icon)
    d.text((230, 105), 'PumpCast', font=font(64, True), fill=(17, 17, 17))
    d.text((232, 180), 'Gas price forecasts for 77 Canadian cities', font=font(30), fill=(102, 102, 102))
    d.text((80, 300), city.name, font=font(96, True), fill=ACCENT)
    d.text((84, 420), f'{PROVINCE_NAMES[city.province][0]}  ·  28-day forecast  ·  best day to fill up',
           font=font(32), fill=(68, 68, 68))
    # A small rising/falling forecast motif
    pts = [(820, 560), (880, 540), (940, 548), (1000, 515), (1060, 525), (1120, 490)]
    d.line(pts, fill=ORANGE, width=8, joint='curve')
    return im


def page(city):
    title = f'{city.name} gas price forecast — PumpCast'
    desc = (f'28-day gas price forecast for {city.name}, {PROVINCE_NAMES[city.province][0]}: '
            'expected range, chance of a rise, and the best day to fill up.')
    target = f'../index.html?city={city.slug}'
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape(title)}</title>
<meta name="description" content="{escape(desc)}">
<meta property="og:type" content="website">
<meta property="og:title" content="{escape(title)}">
<meta property="og:description" content="{escape(desc)}">
<meta property="og:image" content="{SITE_URL}/share/{city.slug}.png">
<meta property="og:url" content="{SITE_URL}/c/{city.slug}.html">
<meta name="twitter:card" content="summary_large_image">
<link rel="canonical" href="{SITE_URL}/index.html?city={city.slug}">
<meta http-equiv="refresh" content="0; url={target}">
</head>
<body><p><a href="{target}">{escape(title)}</a></p></body>
</html>
'''


def main():
    os.makedirs(os.path.join(ROOT, 'site', 'c'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'site', 'share'), exist_ok=True)
    for city in CITIES.values():
        card(city).save(os.path.join(ROOT, 'site', 'share', f'{city.slug}.png'), optimize=True)
        with open(os.path.join(ROOT, 'site', 'c', f'{city.slug}.html'), 'w') as f:
            f.write(page(city))
    print(f'Wrote {len(CITIES)} share pages and cards')


if __name__ == '__main__':
    main()
