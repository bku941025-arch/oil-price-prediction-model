/* sw.js — Offline support: pages and assets cache-first, data network-first. */
const VERSION = 'pumpcast-v3';
const SHELL = ['index.html', 'map.html', 'fillup.html', 'budget.html', 'trip.html', 'compare.html', 'ev.html',
               'breakdown.html', 'history.html', 'accuracy.html', 'api.html', 'model.html', 'roadtrip.html', 'mylog.html',
               'assets/style.css', 'assets/app.js', 'assets/icon.svg', 'manifest.webmanifest'];

self.addEventListener('install', e => {
    e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
    e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
        .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
    const url = new URL(e.request.url);
    if (e.request.method !== 'GET' || url.origin !== location.origin) return;
    // Everything same-origin is network-first so daily data and new deploys show up immediately;
    // the cache is only a fallback when offline.
    e.respondWith(fetch(e.request).then(res => {
        if (res.ok) {
            const copy = res.clone();
            caches.open(VERSION).then(c => c.put(e.request, copy));
        }
        return res;
    }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
