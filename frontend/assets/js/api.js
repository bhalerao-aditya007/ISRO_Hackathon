/**
 * PRISM Frontend — API Helper
 * -----------------------------------------------------------------------
 * Thin fetch wrapper so every page calls the backend the same way.
 * Right now every page renders mock/static numbers, so nothing calls
 * this yet. Search each page for "TODO(backend)" comments to find the
 * exact spots designed to be swapped over to PrismAPI.* calls.
 *
 * Usage once a backend exists:
 *   const data = await PrismAPI.get("/missions");
 *   const result = await PrismAPI.post("/analysis/run", { siteId: "shackleton-1" });
 */
window.PrismAPI = (function () {
  function baseUrl() {
    return (window.PRISM_CONFIG && window.PRISM_CONFIG.API_BASE_URL) || "";
  }

  function isConfigured() {
    return Boolean(baseUrl());
  }

  async function request(path, options = {}) {
    if (!isConfigured()) {
      console.warn(
        `[PrismAPI] No API_BASE_URL configured — skipping real request to "${path}". ` +
        `Set window.PRISM_CONFIG.API_BASE_URL in assets/js/config.js once the backend is live.`
      );
      throw new Error("PRISM_API_NOT_CONFIGURED");
    }

    const res = await fetch(`${baseUrl()}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });

    if (!res.ok) {
      throw new Error(`PrismAPI request failed: ${res.status} ${res.statusText}`);
    }

    const contentType = res.headers.get("content-type") || "";
    return contentType.includes("application/json") ? res.json() : res.text();
  }

  return {
    isConfigured,
    get: (path) => request(path, { method: "GET" }),
    post: (path, body) => request(path, { method: "POST", body: JSON.stringify(body) }),
    put: (path, body) => request(path, { method: "PUT", body: JSON.stringify(body) }),
    del: (path) => request(path, { method: "DELETE" }),
  };
})();
