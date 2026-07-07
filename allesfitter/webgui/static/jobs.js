// Jobs dashboard: poll /jobs/status and stream a selected run's log.
(function () {
  const body = document.getElementById("jobs-body");
  const logPanel = document.getElementById("log-panel");
  const logTitle = document.getElementById("log-title");
  const logBody = document.getElementById("log-body");
  const el = window.AF.el;
  let following = null;

  function stateBadge(state) {
    return el("span", { class: "badge state-" + state }, state);
  }

  function row(r) {
    const runLink = el("a", { href: "/results/" + r.run_id }, r.run_id);
    const logBtn = el("button", { class: "ghost" }, "log");
    logBtn.addEventListener("click", () => follow(r.run_id));
    return el("tr", {},
      el("td", {}, runLink),
      el("td", {}, r.target),
      el("td", { class: "mono small" }, r.insts || "—"),
      el("td", {}, r.sampler),
      el("td", {}, stateBadge(r.state)),
      el("td", { class: "mono small" }, r.logz || "—"),
      el("td", { class: "center" }, logBtn),
    );
  }

  async function refresh() {
    try {
      const res = await fetch("/jobs/status");
      const { runs } = await res.json();
      body.replaceChildren();
      if (!runs.length) {
        body.append(el("tr", {}, el("td", { colspan: "7", class: "muted" }, "No runs yet.")));
      } else {
        for (const r of runs) body.append(row(r));
      }
    } catch (e) { /* transient; try again next tick */ }
  }

  async function refreshLog() {
    if (!following) return;
    try {
      const res = await fetch("/jobs/log/" + following);
      logBody.textContent = (await res.text()) || "(no output yet)";
      logBody.scrollTop = logBody.scrollHeight;
    } catch (e) { /* ignore */ }
  }

  function follow(runId) {
    following = runId;
    logTitle.textContent = "Log · " + runId;
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
