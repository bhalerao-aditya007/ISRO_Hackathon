# PRISM — Lunar Resource Intelligence (Frontend)

Static, no-build-step frontend for PRISM. Plain HTML + Tailwind (via CDN) +
vanilla JS. Ready to deploy on Vercel as-is.

## Structure

```
index.html                  Landing page (served at "/")
mission_control.html        Mission Control dashboard
dataset.html                 Dataset / data-source ecosystem view
intelligence_dashboard.html  Executive mission intelligence / results view
animation.html               Standalone Three.js hero animation (not yet
                              embedded anywhere — available if you want to
                              drop it into a page via <iframe>)
assets/
  js/
    config.js                Single place to set the backend API base URL
    api.js                   Tiny fetch() wrapper (PrismAPI.get/post/...)
  concept-reference.png       Reference concept image (not used in any page)
vercel.json                  Deployment config (clean URLs, asset caching)
```

## Navigation

All four pages share the same top nav / sidebar pattern and link to each
other with real relative paths (`index.html`, `mission_control.html`,
`dataset.html`, `intelligence_dashboard.html`). A few secondary links
(footer "Privacy Protocol", "Support", etc.) intentionally still point to
`#` because there's no corresponding page yet — wire these up once those
pages exist.

## Connecting the backend

1. Open `assets/js/config.js` and set `API_BASE_URL` to your backend's
   origin (e.g. `https://api.yourapp.com`), or to a relative path like
   `/api` if you're deploying serverless functions in this same Vercel
   project.
2. `assets/js/api.js` exposes a small helper already wired into every
   page's `<head>`:
   ```js
   const data = await PrismAPI.get("/missions");
   const result = await PrismAPI.post("/analysis/run", { siteId: "..." });
   ```
3. Search the codebase for `TODO(backend)` — each one marks a spot that
   currently renders mock/static numbers (mission metrics, dataset
   counters, the "Resync Agents" analysis simulation in
   `mission_control.html`) and shows the exact call to swap in.

Nothing currently makes a network request — the whole frontend runs on
the static numbers and animations already in the markup, so it's safe to
deploy immediately and wire up the backend incrementally afterward.

## Deploying to Vercel

No build command or output directory needed — this is a static site.

```
vercel
```

or connect the repo in the Vercel dashboard with:
- Framework Preset: **Other**
- Build Command: *(leave empty)*
- Output Directory: *(leave empty / root)*
