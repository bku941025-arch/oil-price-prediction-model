/* app.js — Shared chrome, i18n, theme, city picker, motion and charts for every page. */

const NAV = [
    { file: 'index.html',  en: 'Forecast', fr: 'Prévision', icon: '📈' },
    { file: 'map.html',    en: 'Map',      fr: 'Carte', icon: '🗺️' },
    { file: 'fillup.html', en: 'Best Day', fr: 'Meilleur jour', icon: '🏆' },
    { en: 'Tools', fr: 'Outils', items: [
        { file: 'budget.html',   en: 'Monthly budget',    fr: 'Budget mensuel', icon: '💰' },
        { file: 'trip.html',     en: 'Trip cost',         fr: 'Coût du trajet', icon: '🚗' },
        { file: 'roadtrip.html', en: 'Road trip planner', fr: 'Planificateur de voyage', icon: '🛣️' },
        { file: 'compare.html',  en: 'Worth the drive?',  fr: 'Ça vaut le détour?', icon: '⚖️' },
        { file: 'ev.html',       en: 'EV vs gas',         fr: 'VÉ ou essence', icon: '🔌' },
        { file: 'mylog.html',    en: 'My fill-ups',       fr: 'Mes pleins', icon: '📒' },
    ] },
    { en: 'Insights', fr: 'Analyses', items: [
        { file: 'breakdown.html', en: "What's in the price", fr: 'Composantes du prix', icon: '🧾' },
        { file: 'history.html',   en: 'Price history',       fr: 'Historique des prix', icon: '🕰️' },
        { file: 'accuracy.html',  en: 'Track record',        fr: 'Fiabilité', icon: '🎯' },
        { file: 'model.html',     en: 'How the model works', fr: 'Fonctionnement du modèle', icon: '🧠' },
        { file: 'api.html',       en: 'Data, API & widget',  fr: 'Données, API et widget', icon: '🔗' },
    ] },
];

const PRODUCT_NAMES = { regular: ['Regular', 'Ordinaire'], premium: ['Premium', 'Super'], diesel: ['Diesel', 'Diesel'] };

function store(key, value) {
    try {
        if (value === undefined) return localStorage.getItem(key);
        localStorage.setItem(key, value);
    } catch (e) { /* storage blocked: preferences just aren't remembered */ }
    return value;
}

function pref(name, fallback) {
    const fromUrl = new URLSearchParams(location.search).get(name);
    if (fromUrl) store('gasprix_' + name, fromUrl);
    return fromUrl || store('gasprix_' + name) || fallback;
}

const LANG = pref('lang', 'en') === 'fr' ? 'fr' : 'en';   // English by default; French is remembered once chosen
let CITY = pref('city', 'montreal');

/** Pick the string for the current language: tr('Hello', 'Bonjour'). */
function tr(en, fr) { return LANG === 'fr' ? fr : en; }

// ── Theme (applied before first paint: this script loads in <head>) ──────────
const THEMES = ['system', 'light', 'dark'];
let THEME = store('pumpcast_theme') || 'system';
const darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function applyTheme() {
    const dark = THEME === 'dark' || (THEME === 'system' && darkQuery && darkQuery.matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#0a0f1d' : '#2b59c3');
    window.dispatchEvent(new Event('themechange'));
}
applyTheme();
darkQuery?.addEventListener?.('change', () => { if (THEME === 'system') applyTheme(); });

function cycleTheme() {
    THEME = THEMES[(THEMES.indexOf(THEME) + 1) % THEMES.length];
    store('pumpcast_theme', THEME);
    applyTheme();
    updateThemeButton();
    toast(tr(`Theme: ${THEME}`, `Thème : ${{ system: 'système', light: 'clair', dark: 'sombre' }[THEME]}`));
}

function updateThemeButton() {
    const b = document.getElementById('theme-btn');
    if (!b) return;
    b.textContent = { system: '🌓', light: '☀️', dark: '🌙' }[THEME];
    b.setAttribute('aria-label', tr(`Theme: ${THEME} (click to change)`, `Thème : ${THEME} (cliquer pour changer)`));
    b.title = b.getAttribute('aria-label');
}

/** Current value of a CSS colour token, e.g. themeColor('chart-actual'). */
function themeColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue('--' + name).trim();
}

// ── Formatting ───────────────────────────────────────────────────────────────
const fmt = {
    cents: v => (v == null ? '—' : v.toFixed(1) + ' ¢/L'),
    money: v => (LANG === 'fr' ? v.toFixed(2).replace('.', ',') + ' $' : (v < 0 ? '−$' : '$') + Math.abs(v).toFixed(2)),
    dollars: v => (LANG === 'fr' ? Math.round(v) + ' $' : '$' + Math.round(v)),
    signed: v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1),
    pct: v => Math.round(v * 100) + (LANG === 'fr' ? ' %' : '%'),
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

/** Great-circle distance in km between two {lat, lon}. */
function distanceKm(a, b) {
    const r = d => d * Math.PI / 180;
    const h = Math.sin(r(b.lat - a.lat) / 2) ** 2 + Math.cos(r(a.lat)) * Math.cos(r(b.lat)) * Math.sin(r(b.lon - a.lon) / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.sqrt(h));
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

// ── Toasts ───────────────────────────────────────────────────────────────────
function toast(message) {
    let host = document.querySelector('.toast-host');
    if (!host) {
        host = Object.assign(document.createElement('div'), { className: 'toast-host' });
        host.setAttribute('role', 'status');
        host.setAttribute('aria-live', 'polite');
        document.body.appendChild(host);
    }
    const t = Object.assign(document.createElement('div'), { className: 'toast', textContent: message });
    host.appendChild(t);
    setTimeout(() => t.remove(), 2600);
}

// ── City picker (searchable, keyboard-driven, with "near me") ───────────────
const fold = s => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();

function recentCities() {
    try { return JSON.parse(store('pumpcast_recent') || '[]'); } catch (e) { return []; }
}

function goToCity(slug) {
    const recent = [slug, ...recentCities().filter(s => s !== slug)].slice(0, 5);
    store('pumpcast_recent', JSON.stringify(recent));
    const here = location.pathname.split('/').pop() || 'index.html';
    location.href = link(here, { city: slug });
}

function openCityPicker(index) {
    if (document.querySelector('.palette-backdrop')) return;
    const opener = document.activeElement;
    const back = document.createElement('div');
    back.className = 'palette-backdrop';
    back.innerHTML = `
        <div class="palette" role="dialog" aria-modal="true" aria-label="${tr('Choose a city', 'Choisir une ville')}">
            <div class="palette-search">
                <span aria-hidden="true">🔍</span>
                <input type="text" role="combobox" aria-expanded="true" aria-controls="palette-list" aria-autocomplete="list"
                       placeholder="${tr('Search 77 cities…', 'Rechercher parmi 77 villes…')}" autocomplete="off" spellcheck="false">
                <kbd>Esc</kbd>
            </div>
            <div class="palette-actions">
                <button type="button" class="chip-btn" data-act="locate">📍 ${tr('Use my location', 'Utiliser ma position')}</button>
                <button type="button" class="chip-btn" data-act="cheapest">💸 ${tr('Cheapest right now', 'La moins chère')}</button>
            </div>
            <div class="palette-list" id="palette-list" role="listbox"></div>
            <div class="palette-foot"><span><kbd>↑</kbd> <kbd>↓</kbd> ${tr('move', 'naviguer')}</span><span><kbd>↵</kbd> ${tr('open', 'ouvrir')}</span><span><kbd>/</kbd> ${tr('search anywhere', 'rechercher partout')}</span></div>
        </div>`;
    document.body.appendChild(back);
    const input = back.querySelector('input');
    const list = back.querySelector('.palette-list');
    let items = [], active = 0;

    const close = () => { back.remove(); opener?.focus?.(); };
    const highlight = (name, q) => {
        if (!q) return esc(name);
        const i = fold(name).indexOf(fold(q));
        return i < 0 ? esc(name) : esc(name.slice(0, i)) + '<mark>' + esc(name.slice(i, i + q.length)) + '</mark>' + esc(name.slice(i + q.length));
    };
    function render() {
        const q = input.value.trim();
        const byScore = index.cities.map(c => {
            const n = fold(c.name), p = fold(index.provinces[c.province][LANG]), f = fold(q);
            const score = !f ? 1 : n.startsWith(f) ? 3 : n.includes(f) ? 2 : (p.startsWith(f) || fold(c.province) === f) ? 1 : 0;
            return { c, score };
        }).filter(x => x.score > 0);
        let groups;
        if (!q) {
            const recent = recentCities().map(s => index.cities.find(c => c.slug === s)).filter(Boolean);
            groups = [[tr('Recent', 'Récentes'), recent], [tr('All cities', 'Toutes les villes'),
                index.cities.slice().sort((a, b) => a.name.localeCompare(b.name, 'fr'))]];
        } else {
            groups = [[tr('Matches', 'Résultats'), byScore.sort((a, b) => b.score - a.score || a.c.name.localeCompare(b.c.name, 'fr')).map(x => x.c)]];
        }
        items = [];
        list.innerHTML = groups.filter(g => g[1].length).map(([label, cities]) =>
            `<div class="palette-group">${label}</div>` + cities.map(c => {
                const i = items.push(c) - 1;
                const ch = c.change_7d || 0;
                return `<button type="button" class="palette-item" role="option" id="opt-${i}" data-i="${i}" aria-selected="false">
                    <span>${highlight(c.name, q)} <span class="muted">· ${c.province}</span></span>
                    <span class="meta">${c.today_price.toFixed(1)}¢ <span class="${ch > 0.5 ? 'worse' : ch < -0.5 ? 'better' : ''}">${ch > 0.5 ? '▲' : ch < -0.5 ? '▼' : '▶'} ${fmt.signed(ch)}</span></span>
                </button>`;
            }).join('')).join('') || `<div class="palette-empty">${tr('No city matches', 'Aucune ville trouvée')}</div>`;
        active = 0;
        select(0);
    }
    function select(i) {
        const opts = list.querySelectorAll('.palette-item');
        if (!opts.length) return;
        active = (i + opts.length) % opts.length;
        opts.forEach((o, n) => o.setAttribute('aria-selected', String(n === active)));
        opts[active].scrollIntoView({ block: 'nearest' });
        input.setAttribute('aria-activedescendant', opts[active].id);
    }
    input.addEventListener('input', render);
    input.addEventListener('keydown', e => {
        if (e.key === 'ArrowDown') { e.preventDefault(); select(active + 1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); select(active - 1); }
        else if (e.key === 'Enter' && items[active]) goToCity(items[active].slug);
        else if (e.key === 'Escape') close();
    });
    list.addEventListener('click', e => { const b = e.target.closest('.palette-item'); if (b) goToCity(items[+b.dataset.i].slug); });
    back.addEventListener('click', e => { if (e.target === back) close(); });
    back.querySelector('[data-act="cheapest"]').addEventListener('click', () => {
        goToCity(index.cities.reduce((a, b) => (b.today_price < a.today_price ? b : a)).slug);
    });
    back.querySelector('[data-act="locate"]').addEventListener('click', e => {
        const btn = e.currentTarget;
        if (!navigator.geolocation) return toast(tr('Location isn’t available in this browser', 'Position non disponible dans ce navigateur'));
        btn.textContent = tr('📍 Locating…', '📍 Localisation…');
        navigator.geolocation.getCurrentPosition(pos => {
            const here = { lat: pos.coords.latitude, lon: pos.coords.longitude };
            goToCity(index.cities.reduce((a, b) => (distanceKm(here, b) < distanceKm(here, a) ? b : a)).slug);
        }, () => {
            btn.textContent = `📍 ${tr('Use my location', 'Utiliser ma position')}`;
            toast(tr('Couldn’t get your location', 'Impossible d’obtenir votre position'));
        }, { timeout: 8000, maximumAge: 600000 });
    });
    render();
    input.focus();
}

// ── Chrome ───────────────────────────────────────────────────────────────────
function renderChrome(index) {
    const here = location.pathname.split('/').pop() || 'index.html';
    document.documentElement.lang = LANG;
    const current = index.cities.find(c => c.slug === CITY) || index.cities[0];

    const item = p => `<a href="${link(p.file)}" class="nav-link ${p.file === here ? 'active' : ''}">${p[LANG]}</a>`;
    const skip = Object.assign(document.createElement('a'), { className: 'skip-link', href: '#main', textContent: tr('Skip to content', 'Aller au contenu') });
    document.body.prepend(skip);
    const main = document.querySelector('main');
    if (main) main.id = 'main';

    const nav = document.createElement('nav');
    nav.className = 'navbar';
    nav.setAttribute('aria-label', tr('Main', 'Principal'));
    nav.innerHTML = `
        <a href="${link('index.html')}" class="nav-brand">
            <span class="brand-icon" aria-hidden="true">⛽</span>
            <span class="brand-name">Pump<span class="brand-mtl">Cast</span></span>
        </a>
        <div class="nav-links">
            ${NAV.map(p => p.items ? `
                <details class="nav-menu ${p.items.some(i => i.file === here) ? 'active' : ''}">
                    <summary class="nav-link">${p[LANG]} ▾</summary>
                    <div class="nav-dropdown">${p.items.map(i => `<a href="${link(i.file)}" class="nav-link ${i.file === here ? 'active' : ''}"><span aria-hidden="true">${i.icon}</span> ${i[LANG]}</a>`).join('')}</div>
                </details>` : item(p)).join('')}
        </div>
        <div class="nav-lang">
            <button type="button" class="city-trigger" aria-haspopup="dialog" title="${tr('Change city', 'Changer de ville')}">
                <span class="pin" aria-hidden="true">📍</span><span class="label">${esc(current.name)}</span><kbd aria-hidden="true">/</kbd>
            </button>
            <a href="${link(here, { lang: LANG === 'en' ? 'fr' : 'en' })}" class="lang-btn" lang="${LANG === 'en' ? 'fr' : 'en'}"
               aria-label="${tr('Passer en français', 'Switch to English')}">${LANG === 'en' ? 'FR' : 'EN'}</a>
            <button type="button" class="icon-btn" id="theme-btn"></button>
        </div>`;
    skip.after(nav);
    updateThemeButton();
    nav.querySelector('#theme-btn').addEventListener('click', cycleTheme);
    nav.querySelector('.city-trigger').addEventListener('click', () => openCityPicker(index));

    nav.querySelectorAll('details.nav-menu').forEach(d => d.addEventListener('toggle', () => {
        if (d.open) nav.querySelectorAll('details.nav-menu').forEach(o => { if (o !== d) o.open = false; });
    }));
    document.addEventListener('click', e => {
        if (!e.target.closest('.nav-menu')) nav.querySelectorAll('details.nav-menu').forEach(d => { d.open = false; });
        if (!e.target.closest('.more-sheet, [data-more]')) document.querySelector('.more-sheet')?.remove();
    });
    document.addEventListener('keydown', e => {
        const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
        if ((e.key === '/' && !typing) || (e.key.toLowerCase() === 'k' && (e.metaKey || e.ctrlKey))) {
            e.preventDefault();
            openCityPicker(index);
        }
        if (e.key === 'Escape') document.querySelector('.more-sheet')?.remove();
    });

    // Phone tab bar
    const tools = NAV.filter(n => n.items).flatMap(n => n.items);
    const tab = document.createElement('nav');
    tab.className = 'tabbar';
    tab.setAttribute('aria-label', tr('Sections', 'Sections'));
    const inMore = tools.some(t => t.file === here);
    tab.innerHTML = NAV.filter(n => !n.items).map(p =>
        `<a href="${link(p.file)}" class="${p.file === here ? 'active' : ''}"><span class="ico" aria-hidden="true">${p.icon}</span>${p[LANG]}</a>`).join('') +
        `<button type="button" data-cities><span class="ico" aria-hidden="true">📍</span>${tr('City', 'Ville')}</button>
         <button type="button" data-more class="${inMore ? 'active' : ''}" aria-haspopup="true"><span class="ico" aria-hidden="true">☰</span>${tr('More', 'Plus')}</button>`;
    document.body.appendChild(tab);
    tab.querySelector('[data-cities]').addEventListener('click', () => openCityPicker(index));
    tab.querySelector('[data-more]').addEventListener('click', () => {
        if (document.querySelector('.more-sheet')) return document.querySelector('.more-sheet').remove();
        const sheet = document.createElement('div');
        sheet.className = 'more-sheet';
        sheet.innerHTML = tools.map(t => `<a href="${link(t.file)}" class="${t.file === here ? 'active' : ''}"><span aria-hidden="true">${t.icon}</span> ${t[LANG]}</a>`).join('');
        document.body.appendChild(sheet);
    });

    if (index.warnings && index.warnings.length) {
        const w = Object.assign(document.createElement('div'), { className: 'site-warning', textContent: index.warnings.join(' ') });
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
    document.querySelectorAll('[data-en-placeholder]').forEach(el => { el.placeholder = el.dataset[LANG + 'Placeholder']; });
    const title = document.querySelector('title');
    if (title && title.dataset[LANG]) title.textContent = title.dataset[LANG];
    revealOnScroll();
}

// ── Motion helpers ───────────────────────────────────────────────────────────
const REDUCED_MOTION = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false;

/** Fade cards in as they scroll into view. */
function revealOnScroll(root) {
    const els = (root || document).querySelectorAll('.card, .hero, .status-item');
    if (REDUCED_MOTION || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver(entries => entries.forEach(en => {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
    }), { rootMargin: '0px 0px -8% 0px' });
    els.forEach((el, i) => {
        if (el.getBoundingClientRect().top < innerHeight) return;   // already visible: no flash
        el.classList.add('reveal');
        el.style.animationDelay = `${Math.min(i, 6) * 40}ms`;
        io.observe(el);
    });
}

/** Animate a number into an element: countUp(el, 192.3, v => v.toFixed(1)). */
function countUp(el, to, format, ms) {
    const from = parseFloat(el.dataset.value || '0') || to * 0.9;
    el.dataset.value = to;
    if (REDUCED_MOTION) { el.textContent = format(to); return; }
    const t0 = performance.now(), dur = ms || 900;
    const step = now => {
        const p = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
        el.textContent = format(from + (to - from) * e);
        if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

// ── Data ─────────────────────────────────────────────────────────────────────
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
        store('gasprix_product', current);
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

// ── Charts ───────────────────────────────────────────────────────────────────
const CHART_FONT = { family: "'Plus Jakarta Sans', system-ui, sans-serif", size: 11 };
const GRID = { color: 'rgba(0,0,0,0.05)' };   // replaced by theme colours below
const ROLE_TOKENS = { actual: 'chart-actual', forecast: 'chart-forecast', alt: 'chart-alt' };

/** Apply theme colours to Chart.js defaults and every live chart (series tagged with a role). */
function themeCharts() {
    if (!window.Chart) return;
    const ink = themeColor('muted'), grid = themeColor('chart-grid');
    Chart.defaults.color = ink;
    Chart.defaults.font.family = CHART_FONT.family;
    Chart.defaults.borderColor = grid;
    Chart.defaults.plugins.tooltip.backgroundColor = themeColor('ink');
    Chart.defaults.plugins.tooltip.titleColor = themeColor('bg');
    Chart.defaults.plugins.tooltip.bodyColor = themeColor('bg');
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Object.values(Chart.instances || {}).forEach(ch => {
        ch.data.datasets.forEach(ds => {
            if (ds._role) ds.borderColor = themeColor(ROLE_TOKENS[ds._role]);
            if (ds._band) ds.backgroundColor = themeColor('chart-band');
        });
        Object.values(ch.options.scales || {}).forEach(sc => {
            if (sc.grid) sc.grid.color = grid;
            if (sc.ticks) sc.ticks.color = ink;
        });
        ch.update('none');
    });
}
window.addEventListener('themechange', themeCharts);

/** Vertical marker line at a label index: chart.$marker = i; chart.update(). */
const markerPlugin = {
    id: 'marker',
    afterDatasetsDraw(chart) {
        if (chart.$marker == null) return;
        const { ctx, chartArea, scales: { x } } = chart;
        const px = x.getPixelForValue(chart.$marker);
        ctx.save();
        ctx.strokeStyle = themeColor('ink');
        ctx.globalAlpha = 0.35;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(px, chartArea.top); ctx.lineTo(px, chartArea.bottom); ctx.stroke();
        ctx.restore();
    },
};

/**
 * Line chart with an optional shaded band.
 * series = [{label, data, role: 'actual'|'forecast'|'alt' (or color), dash, band: [low, high]}]
 */
function bandChart(canvas, labels, series, extra) {
    themeCharts();
    const datasets = [];
    for (const s of series) {
        const color = s.role ? themeColor(ROLE_TOKENS[s.role]) : s.color;
        if (s.band) {
            datasets.push({ label: '_hi', data: s.band[1], borderColor: 'transparent', backgroundColor: s.role ? themeColor('chart-band') : s.bandColor,
                            _band: !!s.role, pointRadius: 0, fill: '+1', tension: 0.3, spanGaps: false });
            datasets.push({ label: '_lo', data: s.band[0], borderColor: 'transparent', pointRadius: 0, fill: false, tension: 0.3 });
        }
        datasets.push({ label: s.label, data: s.data, borderColor: color, _role: s.role, borderWidth: s.width || 2.25,
                        borderDash: s.dash || [], pointRadius: 0, pointHoverRadius: 5, pointHoverBorderWidth: 2,
                        pointHoverBackgroundColor: themeColor('surface'), tension: 0.3, fill: false, spanGaps: !!s.spanGaps });
    }
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets },
        plugins: [markerPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: extra && extra.animation === false ? false : { duration: REDUCED_MOTION ? 0 : 700, easing: 'easeOutCubic' },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { filter: i => !i.text.startsWith('_'), font: { ...CHART_FONT, size: 12 }, boxWidth: 8, boxHeight: 8 } },
                tooltip: {
                    filter: i => !i.dataset.label.startsWith('_') && i.parsed.y != null,
                    callbacks: {
                        title: items => items.length ? fmt.date(items[0].label, { weekday: 'short', month: 'short', day: 'numeric' }) : '',
                        label: c => ` ${c.dataset.label}: ${c.parsed.y.toFixed(1)} ¢/L`,
                    },
                },
            },
            scales: {
                x: { ticks: { maxTicksLimit: 8, font: CHART_FONT, maxRotation: 0, callback(v) { return fmt.date(this.getLabelForValue(v), extra && extra.dateFmt); } },
                     grid: { display: false } },
                y: { ticks: { callback: v => v + '¢', font: CHART_FONT }, grid: { color: themeColor('chart-grid') }, border: { display: false } },
            },
        },
    });
}

// Installable app: cache the shell and last data for offline use
if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
