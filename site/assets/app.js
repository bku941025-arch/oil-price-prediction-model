/* app.js — Shared chrome, i18n, city selection and data loading for every page. */

const NAV = [
    { file: 'index.html',  en: 'Forecast', fr: 'Prévision' },
    { file: 'map.html',    en: 'Map',      fr: 'Carte' },
    { file: 'fillup.html', en: 'Best Day', fr: 'Meilleur jour' },
    { en: 'Tools', fr: 'Outils', items: [
        { file: 'budget.html',  en: 'Monthly budget',   fr: 'Budget mensuel' },
        { file: 'trip.html',    en: 'Trip cost',        fr: 'Coût du trajet' },
        { file: 'compare.html', en: 'Worth the drive?', fr: 'Ça vaut le détour?' },
        { file: 'ev.html',      en: 'EV vs gas',        fr: 'VÉ ou essence' },
        { file: 'roadtrip.html', en: 'Road trip planner', fr: 'Planificateur de voyage' },
        { file: 'mylog.html',   en: 'My fill-ups',      fr: 'Mes pleins' },
    ] },
    { en: 'Insights', fr: 'Analyses', items: [
        { file: 'breakdown.html', en: "What's in the price", fr: 'Composantes du prix' },
        { file: 'history.html',   en: 'Price history',       fr: 'Historique des prix' },
        { file: 'accuracy.html',  en: 'Track record',        fr: 'Fiabilité' },
        { file: 'model.html',     en: 'How the model works', fr: 'Fonctionnement du modèle' },
        { file: 'api.html',       en: 'Data, API & widget',  fr: 'Données, API et widget' },
    ] },
];

const PRODUCT_NAMES = { regular: ['Regular', 'Ordinaire'], premium: ['Premium', 'Super'], diesel: ['Diesel', 'Diesel'] };

function pref(name, fallback) {
    const fromUrl = new URLSearchParams(location.search).get(name);
    try {
        if (fromUrl) localStorage.setItem('gasprix_' + name, fromUrl);
        return fromUrl || localStorage.getItem('gasprix_' + name) || fallback;
    } catch (e) {
        return fromUrl || fallback;
    }
}

const LANG = pref('lang', 'en') === 'fr' ? 'fr' : 'en';   // English by default; French is remembered once chosen
let CITY = pref('city', 'montreal');

/** Pick the string for the current language: tr('Hello', 'Bonjour'). */
function tr(en, fr) { return LANG === 'fr' ? fr : en; }

const fmt = {
    cents: v => (v == null ? '—' : v.toFixed(1) + ' ¢/L'),
    money: v => (LANG === 'fr' ? v.toFixed(2).replace('.', ',') + ' $' : (v < 0 ? '−$' : '$') + Math.abs(v).toFixed(2)),
    dollars: v => (LANG === 'fr' ? Math.round(v) + ' $' : '$' + Math.round(v)),
    signed: v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1),
    pct: v => Math.round(v * 100) + ' %'.replace(' ', LANG === 'fr' ? ' ' : ''),
    date: (s, opts) => new Date(s + 'T00:00:00').toLocaleDateString(LANG === 'fr' ? 'fr-CA' : 'en-CA',
        opts || { month: 'short', day: 'numeric' }),
};

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function link(file, extra) {
    const p = new URLSearchParams({ lang: LANG, city: CITY, ...(extra || {}) });
    return `${file}?${p}`;
}

function cityOptions(select, index, selected) {
    const byProv = {};
    for (const c of index.cities) (byProv[c.province] = byProv[c.province] || []).push(c);
    const order = ['QC', 'ON', 'BC', 'AB', 'SK', 'MB', 'NB', 'NS', 'PE', 'NL', 'YT', 'NT'];
    for (const p of order.filter(p => byProv[p])) {
        const group = document.createElement('optgroup');
        group.label = index.provinces[p][LANG];
        byProv[p].sort((a, b) => a.name.localeCompare(b.name, 'fr')).forEach(c =>
            group.appendChild(new Option(c.name, c.slug, false, c.slug === selected)));
        select.appendChild(group);
    }
}

function renderChrome(index) {
    const here = location.pathname.split('/').pop() || 'index.html';
    document.documentElement.lang = LANG;

    const item = p => `<a href="${link(p.file)}" class="nav-link ${p.file === here ? 'active' : ''}">${p[LANG]}</a>`;
    const skip = document.createElement('a');
    skip.className = 'skip-link';
    skip.href = '#main';
    skip.textContent = tr('Skip to content', 'Aller au contenu');
    document.body.prepend(skip);
    const main = document.querySelector('main');
    if (main) main.id = 'main';

    const nav = document.createElement('nav');
    nav.className = 'navbar';
    nav.setAttribute('aria-label', tr('Main', 'Principal'));
    nav.innerHTML = `
        <a href="${link('index.html')}" class="nav-brand">
            <span class="brand-icon">⛽</span>
            <span class="brand-name">Pump<span class="brand-mtl">Cast</span></span>
        </a>
        <div class="nav-links">
            ${NAV.map(p => p.items ? `
                <details class="nav-menu ${p.items.some(i => i.file === here) ? 'active' : ''}">
                    <summary class="nav-link">${p[LANG]} ▾</summary>
                    <div class="nav-dropdown">${p.items.map(item).join('')}</div>
                </details>` : item(p)).join('')}
        </div>
        <div class="nav-lang">
            <a href="${link(here, { lang: 'en' })}" class="lang-btn ${LANG === 'en' ? 'active' : ''}">EN</a>
            <span class="lang-sep">·</span>
            <a href="${link(here, { lang: 'fr' })}" class="lang-btn ${LANG === 'fr' ? 'active' : ''}">FR</a>
            <select class="city-select" aria-label="${tr('City', 'Ville')}"></select>
        </div>`;
    skip.after(nav);

    // Close an open menu when clicking elsewhere or opening another
    nav.querySelectorAll('details.nav-menu').forEach(d => d.addEventListener('toggle', () => {
        if (d.open) nav.querySelectorAll('details.nav-menu').forEach(o => { if (o !== d) o.open = false; });
    }));
    document.addEventListener('click', e => {
        if (!e.target.closest('.nav-menu')) nav.querySelectorAll('details.nav-menu').forEach(d => { d.open = false; });
    });

    const select = nav.querySelector('.city-select');
    cityOptions(select, index, CITY);
    select.addEventListener('change', () => { location.href = link(here, { city: select.value }); });

    if (index.warnings && index.warnings.length) {
        const w = document.createElement('div');
        w.className = 'site-warning';
        w.textContent = index.warnings.join(' ');
        nav.after(w);
    }

    const footer = document.createElement('footer');
    footer.className = 'footer';
    footer.innerHTML = `<p>${tr(
        'Retail prices from Kalibrate; crude and wholesale gasoline from FRED; USD/CAD from the Bank of Canada. Updated daily. Forecasts are estimates — not financial advice.',
        'Prix de détail : Kalibrate; brut et essence en gros : FRED; USD/CAD : Banque du Canada. Mise à jour quotidienne. Les prévisions sont des estimations — pas des conseils financiers.')}
        · <a href="${link('api.html')}">${tr('Data & API', 'Données et API')}</a>
        · <a href="feeds/${CITY}.xml">RSS</a></p>`;
    document.body.append(footer);

    document.querySelectorAll('[data-en]').forEach(el => { el.textContent = el.dataset[LANG]; });
    document.querySelectorAll('[data-en-placeholder]').forEach(el => {
        el.placeholder = el.dataset[LANG + 'Placeholder'];
    });
    const title = document.querySelector('title');
    if (title && title.dataset[LANG]) title.textContent = title.dataset[LANG];
}

async function getJSON(path) {
    const res = await fetch(path, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res.json();
}

/** Render nav/footer, then resolve with { index, city } for the selected city. */
async function loadPage(opts) {
    const index = await getJSON('data/index.json');
    if (!index.cities.some(c => c.slug === CITY)) CITY = index.default_city;
    renderChrome(index);
    if (opts && opts.cityData === false) return { index };
    const city = await getJSON(`data/${CITY}.json`);
    return { index, city };
}

function showError(el, err) {
    console.error(err);
    el.innerHTML = `<p class="error-msg">${tr('Failed to load data.', 'Échec du chargement.')}</p>`;
}

function weekdayName(s, style) {
    return new Date(s + 'T00:00:00').toLocaleDateString(LANG === 'fr' ? 'fr-CA' : 'en-CA', { weekday: style || 'short' });
}

function isWeekend(s) {
    const d = new Date(s + 'T00:00:00').getDay();
    return d === 0 || d === 6;
}

/** Forecast rows for a product: regular uses the city forecast, others the derived one. */
function productForecast(city, product) {
    if (product === 'regular' || !city.products[product]) return { today: city.today_price, forecast: city.forecast };
    return { today: city.products[product].today_price, forecast: city.products[product].forecast };
}

/** Product toggle (Regular / Premium / Diesel) that calls onChange(product). */
function productToggle(el, city, onChange) {
    const products = ['regular', ...Object.keys(city.products || {})];
    let current = pref('product', 'regular');
    if (!products.includes(current)) current = 'regular';
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', tr('Fuel type', 'Type de carburant'));
    el.innerHTML = products.map(p => `<button type="button" data-p="${p}" class="${p === current ? 'active' : ''}" aria-pressed="${p === current}">${PRODUCT_NAMES[p][LANG === 'fr' ? 1 : 0]}</button>`).join('');
    el.addEventListener('click', e => {
        const b = e.target.closest('button');
        if (!b) return;
        current = b.dataset.p;
        try { localStorage.setItem('gasprix_product', current); } catch (err) { /* private mode */ }
        setPressed(el, b);
        onChange(current);
    });
    onChange(current);
}

/** Mark one button in a segmented control as active/pressed. */
function setPressed(group, button) {
    group.querySelectorAll('button').forEach(x => {
        x.classList.toggle('active', x === button);
        x.setAttribute('aria-pressed', String(x === button));
    });
}

/** Give a chart canvas a text alternative for screen readers. */
function describeChart(canvas, text) {
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', text);
}

/** Download text as a file (calendar events, CSV). */
function downloadText(filename, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = Object.assign(document.createElement('a'), { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const CHART_FONT = { family: 'DM Sans', size: 11 };
const GRID = { color: 'rgba(0,0,0,0.04)' };

/** Line chart with an optional shaded band: series = [{label, data, color, dash, band:[low,high]}]. */
function bandChart(canvas, labels, series, extra) {
    const datasets = [];
    for (const s of series) {
        if (s.band) {
            datasets.push({ label: '_hi', data: s.band[1], borderColor: 'transparent', backgroundColor: s.bandColor,
                            pointRadius: 0, fill: '+1', tension: 0.25, spanGaps: false });
            datasets.push({ label: '_lo', data: s.band[0], borderColor: 'transparent', pointRadius: 0,
                            fill: false, tension: 0.25 });
        }
        datasets.push({ label: s.label, data: s.data, borderColor: s.color, borderWidth: s.width || 2, borderDash: s.dash || [],
                        pointRadius: 0, tension: 0.25, fill: false, spanGaps: !!s.spanGaps });
    }
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: extra && extra.animation === false ? false : undefined,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { filter: i => !i.text.startsWith('_'), font: { ...CHART_FONT, size: 12 } } },
                tooltip: {
                    filter: i => !i.dataset.label.startsWith('_') && i.parsed.y != null,
                    callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)} ¢/L` },
                },
            },
            scales: {
                x: { ticks: { maxTicksLimit: 10, font: CHART_FONT, callback(v) { return fmt.date(this.getLabelForValue(v), extra && extra.dateFmt); } }, grid: GRID },
                y: { ticks: { callback: v => v + '¢', font: CHART_FONT }, grid: GRID },
            },
        },
    });
}

// Installable app: cache the shell and last data for offline use
if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
