/* app.js — Shared chrome, i18n, city selection and data loading for every page. */

const PAGES = [
    { file: 'index.html',    en: 'Forecast', fr: 'Prévision' },
    { file: 'fillup.html',   en: 'Best Day', fr: 'Meilleur jour' },
    { file: 'budget.html',   en: 'Budget',   fr: 'Budget' },
    { file: 'trip.html',     en: 'Trip Cost', fr: 'Coût du trajet' },
    { file: 'accuracy.html', en: 'Track Record', fr: 'Fiabilité' },
];

function pref(name, fallback) {
    const fromUrl = new URLSearchParams(location.search).get(name);
    try {
        if (fromUrl) localStorage.setItem('gasprix_' + name, fromUrl);
        return fromUrl || localStorage.getItem('gasprix_' + name) || fallback;
    } catch (e) {
        return fromUrl || fallback;
    }
}

const LANG = pref('lang', (navigator.language || 'en').startsWith('fr') ? 'fr' : 'en') === 'fr' ? 'fr' : 'en';
let CITY = pref('city', 'montreal');

/** Pick the string for the current language: tr('Hello', 'Bonjour'). */
function tr(en, fr) { return LANG === 'fr' ? fr : en; }

const fmt = {
    cents: v => (v == null ? '—' : v.toFixed(1) + ' ¢/L'),
    money: v => (LANG === 'fr' ? v.toFixed(2).replace('.', ',') + ' $' : '$' + v.toFixed(2)),
    dollars: v => (LANG === 'fr' ? Math.round(v) + ' $' : '$' + Math.round(v)),
    signed: v => (v > 0 ? '+' : '') + v.toFixed(1),
    date: (s, opts) => new Date(s + 'T00:00:00').toLocaleDateString(LANG === 'fr' ? 'fr-CA' : 'en-CA',
        opts || { month: 'short', day: 'numeric' }),
};

function link(file, extra) {
    const p = new URLSearchParams({ lang: LANG, city: CITY, ...(extra || {}) });
    return `${file}?${p}`;
}

function renderChrome(index) {
    const here = location.pathname.split('/').pop() || 'index.html';
    document.documentElement.lang = LANG;

    const nav = document.createElement('nav');
    nav.className = 'navbar';
    nav.innerHTML = `
        <a href="${link('index.html')}" class="nav-brand">
            <span class="brand-icon">⛽</span>
            <span class="brand-name">GasPrix<span class="brand-mtl">MTL</span></span>
        </a>
        <div class="nav-links">
            ${PAGES.map(p => `<a href="${link(p.file)}" class="nav-link ${p.file === here ? 'active' : ''}">${p[LANG]}</a>`).join('')}
        </div>
        <div class="nav-lang">
            <a href="${link(here, { lang: 'en' })}" class="lang-btn ${LANG === 'en' ? 'active' : ''}">EN</a>
            <span class="lang-sep">·</span>
            <a href="${link(here, { lang: 'fr' })}" class="lang-btn ${LANG === 'fr' ? 'active' : ''}">FR</a>
            <select class="city-select" aria-label="${tr('City', 'Ville')}"></select>
        </div>`;
    document.body.prepend(nav);

    const select = nav.querySelector('.city-select');
    for (const c of index.cities) {
        const opt = new Option(c.name, c.slug, false, c.slug === CITY);
        select.add(opt);
    }
    select.addEventListener('change', () => { location.href = link(here, { city: select.value }); });

    const footer = document.createElement('footer');
    footer.className = 'footer';
    footer.innerHTML = `<p>${tr(
        'Retail prices from Kalibrate; crude, wholesale gasoline and CAD/USD from FRED. Updated daily. Forecasts are estimates — not financial advice.',
        'Prix de détail : Kalibrate; brut, essence en gros et CAD/USD : FRED. Mise à jour quotidienne. Les prévisions sont des estimations — pas des conseils financiers.')}</p>`;
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
async function loadPage() {
    const index = await getJSON('data/index.json');
    if (!index.cities.some(c => c.slug === CITY)) CITY = index.default_city;
    renderChrome(index);
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

const CHART_FONT = { family: 'DM Sans', size: 11 };
const GRID = { color: 'rgba(0,0,0,0.04)' };

/** Line chart with an optional shaded band: series = [{label, data, color, dash, band:[low,high]}]. */
function bandChart(canvas, labels, series) {
    const datasets = [];
    for (const s of series) {
        if (s.band) {
            datasets.push({ label: '_hi', data: s.band[1], borderColor: 'transparent', backgroundColor: s.bandColor,
                            pointRadius: 0, fill: '+1', tension: 0.25, spanGaps: false });
            datasets.push({ label: '_lo', data: s.band[0], borderColor: 'transparent', pointRadius: 0,
                            fill: false, tension: 0.25 });
        }
        datasets.push({ label: s.label, data: s.data, borderColor: s.color, borderWidth: 2, borderDash: s.dash || [],
                        pointRadius: 0, tension: 0.25, fill: false, spanGaps: false });
    }
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { filter: i => !i.text.startsWith('_'), font: { ...CHART_FONT, size: 12 } } },
                tooltip: {
                    filter: i => !i.dataset.label.startsWith('_') && i.parsed.y != null,
                    callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)} ¢/L` },
                },
            },
            scales: {
                x: { ticks: { maxTicksLimit: 10, font: CHART_FONT, callback(v) { return fmt.date(this.getLabelForValue(v)); } }, grid: GRID },
                y: { ticks: { callback: v => v + '¢', font: CHART_FONT }, grid: GRID },
            },
        },
    });
}
