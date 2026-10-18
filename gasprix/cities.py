"""
cities.py — Every Kalibrate city: display name, province and approximate coordinates.

Keys are Kalibrate row names with the trailing '*' removed. Coordinates are city
centres to about 0.05°, used only to place markers on the map.
"""

import re
import unicodedata

# Kalibrate name: (display name, province, lat, lon)
_CITIES = {
    'WHITEHORSE'      : ('Whitehorse',          'YT', 60.72, -135.06),
    'VANCOUVER'       : ('Vancouver',           'BC', 49.28, -123.12),
    'VICTORIA'        : ('Victoria',            'BC', 48.43, -123.37),
    'PRINCE GEORGE'   : ('Prince George',       'BC', 53.92, -122.75),
    'KAMLOOPS'        : ('Kamloops',            'BC', 50.67, -120.33),
    'KELOWNA'         : ('Kelowna',             'BC', 49.89, -119.50),
    'FORT ST. JOHN'   : ('Fort St. John',       'BC', 56.25, -120.85),
    'ABBOTSFORD'      : ('Abbotsford',          'BC', 49.05, -122.31),
    'YELLOWKNIFE'     : ('Yellowknife',         'NT', 62.45, -114.37),
    'CALGARY'         : ('Calgary',             'AB', 51.05, -114.07),
    'RED DEER'        : ('Red Deer',            'AB', 52.27, -113.81),
    'EDMONTON'        : ('Edmonton',            'AB', 53.55, -113.49),
    'LETHBRIDGE'      : ('Lethbridge',          'AB', 49.69, -112.84),
    'LLOYDMINSTER'    : ('Lloydminster',        'AB', 53.28, -110.01),
    'GRANDE PRAIRIE'  : ('Grande Prairie',      'AB', 55.17, -118.80),
    'REGINA'          : ('Regina',              'SK', 50.45, -104.61),
    'SASKATOON'       : ('Saskatoon',           'SK', 52.13, -106.67),
    'PRINCE ALBERT'   : ('Prince Albert',       'SK', 53.20, -105.75),
    'MOOSE JAW'       : ('Moose Jaw',           'SK', 50.39, -105.53),
    'WINNIPEG'        : ('Winnipeg',            'MB', 49.90, -97.14),
    'BRANDON'         : ('Brandon',             'MB', 49.85, -99.95),
    'CITY OF TORONTO' : ('Toronto',             'ON', 43.65, -79.38),
    'BRAMPTON'        : ('Brampton',            'ON', 43.73, -79.76),
    'ETOBICOKE'       : ('Etobicoke',           'ON', 43.64, -79.56),
    'MISSISSAUGA'     : ('Mississauga',         'ON', 43.59, -79.64),
    'NORTH YORK'      : ('North York',          'ON', 43.77, -79.41),
    'SCARBOROUGH'     : ('Scarborough',         'ON', 43.77, -79.26),
    'VAUGHAN/MARKHAM' : ('Vaughan / Markham',   'ON', 43.84, -79.40),
    'OTTAWA'          : ('Ottawa',              'ON', 45.42, -75.70),
    'KINGSTON'        : ('Kingston',            'ON', 44.23, -76.49),
    'PETERBOROUGH'    : ('Peterborough',        'ON', 44.30, -78.32),
    'WINDSOR'         : ('Windsor',             'ON', 42.31, -83.04),
    'LONDON'          : ('London',              'ON', 42.98, -81.25),
    'SUDBURY'         : ('Sudbury',             'ON', 46.49, -80.99),
    'SAULT STE MARIE' : ('Sault Ste. Marie',    'ON', 46.52, -84.33),
    'THUNDER BAY'     : ('Thunder Bay',         'ON', 48.38, -89.25),
    'NORTH BAY'       : ('North Bay',           'ON', 46.31, -79.46),
    'TIMMINS'         : ('Timmins',             'ON', 48.48, -81.33),
    'HAMILTON'        : ('Hamilton',            'ON', 43.26, -79.87),
    'ST. CATHARINES'  : ('St. Catharines',      'ON', 43.16, -79.25),
    'BARRIE'          : ('Barrie',              'ON', 44.39, -79.69),
    'BRANTFORD'       : ('Brantford',           'ON', 43.14, -80.26),
    'GUELPH'          : ('Guelph',              'ON', 43.55, -80.25),
    'KITCHENER'       : ('Kitchener',           'ON', 43.45, -80.49),
    'OSHAWA'          : ('Oshawa',              'ON', 43.90, -78.86),
    'SARNIA'          : ('Sarnia',              'ON', 42.97, -82.40),
    'MONTRÉAL'        : ('Montréal',            'QC', 45.50, -73.57),
    'QUÉBEC'          : ('Québec',              'QC', 46.81, -71.21),
    'SHERBROOKE'      : ('Sherbrooke',          'QC', 45.40, -71.89),
    'GASPÉ'           : ('Gaspé',               'QC', 48.83, -64.48),
    'CHICOUTIMI'      : ('Chicoutimi',          'QC', 48.43, -71.06),
    'RIMOUSKI'        : ('Rimouski',            'QC', 48.45, -68.52),
    'TROIS RIVIÈRES'  : ('Trois-Rivières',      'QC', 46.34, -72.54),
    'DRUMMONDVILLE'   : ('Drummondville',       'QC', 45.88, -72.48),
    "VAL D'OR"        : ("Val-d'Or",            'QC', 48.10, -77.78),
    'GATINEAU'        : ('Gatineau',            'QC', 45.48, -75.70),
    'SAINT JOHN'      : ('Saint John',          'NB', 45.27, -66.06),
    'FREDERICTON'     : ('Fredericton',         'NB', 45.96, -66.64),
    'MONCTON'         : ('Moncton',             'NB', 46.09, -64.78),
    'BATHURST'        : ('Bathurst',            'NB', 47.62, -65.65),
    'EDMUNDSTON'      : ('Edmundston',          'NB', 47.37, -68.33),
    'MIRAMICHI'       : ('Miramichi',           'NB', 47.03, -65.50),
    'CAMPBELLTON'     : ('Campbellton',         'NB', 48.01, -66.67),
    'SUSSEX'          : ('Sussex',              'NB', 45.72, -65.51),
    'WOODSTOCK'       : ('Woodstock',           'NB', 46.15, -67.60),
    'HALIFAX'         : ('Halifax',             'NS', 44.65, -63.57),
    'SYDNEY'          : ('Sydney',              'NS', 46.14, -60.19),
    'YARMOUTH'        : ('Yarmouth',            'NS', 43.84, -66.12),
    'TRURO'           : ('Truro',               'NS', 45.37, -63.27),
    'KENTVILLE'       : ('Kentville',           'NS', 45.08, -64.50),
    'NEW GLASGOW'     : ('New Glasgow',         'NS', 45.59, -62.65),
    'CHARLOTTETOWN'   : ('Charlottetown',       'PE', 46.24, -63.13),
    'ST JOHNS'        : ("St. John's",          'NL', 47.56, -52.71),
    'GANDER'          : ('Gander',              'NL', 48.95, -54.61),
    'LABRADOR CITY'   : ('Labrador City',       'NL', 52.94, -66.91),
    'CORNER BROOK'    : ('Corner Brook',        'NL', 48.95, -57.95),
    'GRAND FALLS'     : ('Grand Falls-Windsor', 'NL', 48.93, -55.66),
}

PROVINCE_NAMES = {
    'BC': ('British Columbia', 'Colombie-Britannique'), 'AB': ('Alberta', 'Alberta'),
    'SK': ('Saskatchewan', 'Saskatchewan'), 'MB': ('Manitoba', 'Manitoba'),
    'ON': ('Ontario', 'Ontario'), 'QC': ('Québec', 'Québec'),
    'NB': ('New Brunswick', 'Nouveau-Brunswick'), 'NS': ('Nova Scotia', 'Nouvelle-Écosse'),
    'PE': ('Prince Edward Island', 'Île-du-Prince-Édouard'),
    'NL': ('Newfoundland and Labrador', 'Terre-Neuve-et-Labrador'),
    'YT': ('Yukon', 'Yukon'), 'NT': ('Northwest Territories', 'Territoires du Nord-Ouest'),
}

# Atlantic provinces set gasoline prices by regulation (weekly maximums)
REGULATED = {'NB', 'NS', 'PE', 'NL'}


def slugify(name):
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')


class City:
    __slots__ = ('slug', 'kalibrate', 'name', 'province', 'lat', 'lon')

    def __init__(self, kalibrate, name, province, lat, lon):
        self.slug, self.kalibrate, self.name = slugify(name), kalibrate, name
        self.province, self.lat, self.lon = province, lat, lon

    def to_dict(self):
        return {'slug': self.slug, 'name': self.name, 'province': self.province,
                'lat': self.lat, 'lon': self.lon, 'regulated': self.province in REGULATED}


CITIES = {c.slug: c for c in (City(k, *v) for k, v in _CITIES.items())}
BY_KALIBRATE = {c.kalibrate: c for c in CITIES.values()}
