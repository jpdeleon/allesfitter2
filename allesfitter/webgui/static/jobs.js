// Jobs dashboard: poll /jobs/status, stream a run's log, and act on runs.
(function () {
  const body = document.getElementById("jobs-body");
  const logPanel = document.getElementById("log-panel");
  const logTitle = document.getElementById("log-title");
  const logBody = document.getElementById("log-body");
  const el = window.AF.el;
  let following = null; // { id, url }

  const PREP_STATES = new Set(["preparing", "prepared"]);
  const ACTIVE_STATES = new Set(["preparing", "running", "pending", "created"]);

  function stateBadge(state) {
    return el("span", { class: "badge state-" + state }, state);
  }

  // prepare runs log to prepare.log; fits log to results/run.log.
  function logUrl(r) {
    return (PREP_STATES.has(r.state) ? "/prepare/log/" : "/jobs/log/") + r.run_id;
  }

  async function fitRun(r) {
    const sel = document.getElementById("sampler-" + r.run_id);
    const sampler = sel ? sel.value : r.sampler || "mcmc";
    await window.AF.postJSON("/jobs/fit/" + r.run_id + "?sampler=" + sampler, {});
    refresh();
  }

  async function stopRun(r) {
    await window.AF.postJSON("/jobs/stop/" + r.run_id, {});
    refresh();
  }

  function actions(r) {
    if (r.state === "prepared") {
      const sel = el("select", { class: "mini", id: "sampler-" + r.run_id },
        el("option", { value: "mcmc" }, "mcmc"),
        el("option", { value: "ns" }, "ns"));
      if (r.sampler === "ns") sel.value = "ns";
      const fitBtn = el("button", { class: "btn primary mini" }, "Fit →");
      fitBtn.addEventListener("click", () => fitRun(r));
      return el("span", { class: "action-cell" }, sel, fitBtn);
    }
    if (ACTIVE_STATES.has(r.state)) {
      const stopBtn = el("button", { class: "ghost" }, "stop");
      stopBtn.addEventListener("click", () => stopRun(r));
      return stopBtn;
    }
    return el("span", { class: "muted" }, "—");
  }

  function row(r) {
    const runLink = el("a", { href: "/results/" + r.run_id }, r.run_id);
    const logBtn = el("button", { class: "ghost" }, "log");
    logBtn.addEventListener("click", () => follow(r));
    return el("tr", {},
      el("td", {}, runLink),
      el("td", {}, r.target),
      el("td", { class: "mono small" }, r.insts || "—"),
      el("td", {}, r.sampler),
      el("td", {}, stateBadge(r.state)),
      el("td", { class: "mono small" }, r.logz || "—"),
      el("td", { class: "center" }, logBtn),
      el("td", { class: "center" }, actions(r)),
    );
  }

  async function refresh() {
    try {
      const res = await fetch("/jobs/status");
      const { runs } = await res.json();
      body.replaceChildren();
      if (!runs.length) {
        body.append(el("tr", {}, el("td", { colspan: "8", class: "muted" }, "No runs yet.")));
      } else {
        for (const r of runs) body.append(row(r));
      }
    } catch (e) { /* transient; try again next tick */ }
  }

  async function refreshLog() {
    if (!following) return;
    try {
      const res = await fetch(following.url);
      logBody.textContent = (await res.text()) || "(no output yet)";
      logBody.scrollTop = logBody.scrollHeight;
    } catch (e) { /* ignore */ }
  }

  function follow(r) {
    following = { id: r.run_id, url: logUrl(r) };
    logTitle.textContent = "Log · " + r.run_id;
    logPanel.classList.remove("hidden");
    refreshLog();
  }

  document.getElementById("log-close").addEventListener("click", () => {
    following = null;
    logPanel.classList.add("hidden");
  });

  refresh();
  refreshLog();
  setInterval(refresh, 3000);
  setInterval(refreshLog, 2000);
})();
