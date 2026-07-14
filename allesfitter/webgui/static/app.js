// Shared shell, request helpers, and small formatting utilities.
(function () {
  const KEY = "allesfitter-theme";
  const root = document.documentElement;
  const saved = localStorage.getItem(KEY);
  if (saved) root.setAttribute("data-theme", saved);

  const btn = document.getElementById("theme-toggle");
  function currentTheme() {
    const explicit = root.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function updateThemeButton() {
    if (!btn) return;
    const dark = currentTheme() === "dark";
    btn.title = dark ? "Use light theme" : "Use dark theme";
    btn.setAttribute("aria-label", btn.title);
    const icon = btn.querySelector("span");
    if (icon) icon.textContent = dark ? "☀" : "◐";
  }
  if (btn) {
    btn.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem(KEY, next);
      updateThemeButton();
    });
  }
  updateThemeButton();
})();

// Small shared helpers on window.AF.
window.AF = {
  url(path) {
    const suffix = path.startsWith("/") ? path : "/" + path;
    return (window.AF_ROOT_PATH || "") + suffix;
  },
  async postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    return { ok: res.ok, status: res.status, data };
  },
  el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const k in (attrs || {})) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    for (const kid of kids) n.append(kid);
    return n;
  },
  methodName(method) {
    return ({ mcmc: "MCMC", ns: "Nested sampling", optimize: "Optimization" })[method] || method || "—";
  },
  stateGroup(state) {
    if (["created", "pending", "preparing", "running", "stopping"].includes(state)) return "active";
    if (["failed", "stopped"].includes(state)) return "failed";
    return state;
  },
  relativeTime(timestamp) {
    if (!timestamp) return "—";
    const seconds = Math.max(0, Date.now() / 1000 - Number(timestamp));
    if (seconds < 60) return "just now";
    if (seconds < 3600) return Math.floor(seconds / 60) + "m ago";
    if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
    if (seconds < 604800) return Math.floor(seconds / 86400) + "d ago";
    return new Date(Number(timestamp) * 1000).toLocaleDateString();
  },
  toast(message, kind) {
    const region = document.getElementById("toast-region");
    if (!region) return;
    const node = document.createElement("div");
    node.className = "toast" + (kind ? " " + kind : "");
    node.textContent = message;
    region.append(node);
    setTimeout(() => node.remove(), 4200);
  },
};
