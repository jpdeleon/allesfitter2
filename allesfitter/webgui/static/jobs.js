// Live run monitor: filterable lifecycle table, actions, and log drawer.
(function () {
  const body = document.getElementById("jobs-body");
  if (!body) return;
  const logPanel = document.getElementById("log-panel");
  const logTitle = document.getElementById("log-title");
  const logBody = document.getElementById("log-body");
  const search = document.getElementById("run-search");
  const el = window.AF.el;
  const activeStates = new Set(["preparing", "running", "pending", "created", "stopping"]);
  let runs = [];
  let filter = "all";
  let following = null;

  const requestedTarget = new URLSearchParams(window.location.search).get("target");
  if (requestedTarget) search.value = requestedTarget;

  function stateBadge(state) {
    return el("span", { class: "badge state-" + state }, el("i"), state);
  }

  async function stopRun(run) {
    const response = await window.AF.postJSON(window.AF.url("/jobs/stop/" + encodeURIComponent(run.run_id)), {});
    window.AF.toast(response.ok ? "Stop signal sent to " + run.target + "." : "Could not stop run.", response.ok ? "" : "error");
    refresh();
  }

  async function deleteRun(run) {
    if (!confirm("Delete “" + run.target + "” run? Its generated files and results will be permanently removed.")) return;
    const response = await window.AF.postJSON(window.AF.url("/jobs/delete/" + encodeURIComponent(run.run_id)), {});
    window.AF.toast(response.ok ? "Run deleted." : ((response.data && response.data.detail) || "Could not delete run."), response.ok ? "" : "error");
    if (following && following.id === run.run_id) closeLog();
    refresh();
  }

  function actionButton(label, className, handler) {
    const button = el("button", { class: className || "btn mini", type: "button" }, label);
    button.addEventListener("click", handler);
    return button;
  }

  function actions(run) {
    const cell = el("div", { class: "action-cell" });
    if (run.state === "prepared") {
      cell.append(el("a", { class: "btn primary mini", href: window.AF.url("/results/" + encodeURIComponent(run.run_id)) }, "Review"));
    } else {
      cell.append(actionButton("Log", "btn mini", () => follow(run)));
    }
    if (activeStates.has(run.state)) {
      cell.append(actionButton("Stop", "ghost mini danger", () => stopRun(run)));
    } else {
      cell.append(actionButton("Delete", "ghost mini danger", () => deleteRun(run)));
    }
    return cell;
  }

  function row(run) {
    const targetCell = el("td", {},
      el("a", { class: "run-primary", href: window.AF.url("/results/" + encodeURIComponent(run.run_id)) }, run.target || "Untitled"),
      el("span", { class: "run-secondary mono", title: run.run_id }, run.run_id),
    );
    return el("tr", { "data-state": run.state },
      targetCell,
      el("td", { class: "mono small" }, run.insts || "not resolved"),
      el("td", {}, window.AF.methodName(run.sampler)),
      el("td", {}, stateBadge(run.state)),
      el("td", { class: "small muted", title: new Date(Number(run.created_at) * 1000).toLocaleString() }, window.AF.relativeTime(run.started_at || run.created_at)),
      el("td", { class: "mono small" }, run.logz || "—"),
      el("td", {}, actions(run)),
    );
  }

  function matches(run) {
    const query = search.value.trim().toLowerCase();
    if (query && !(run.target + " " + run.run_id + " " + (run.insts || "")).toLowerCase().includes(query)) return false;
    if (filter === "all") return true;
    return window.AF.stateGroup(run.state) === filter;
  }

  function render() {
    const visible = runs.filter(matches);
    body.replaceChildren();
    if (!visible.length) {
      const message = runs.length ? "No runs match these filters." : "No runs yet. Prepare a target to start the pipeline.";
      body.append(el("tr", {}, el("td", { colspan: "7", class: "table-empty" }, message)));
    } else {
      visible.forEach((run) => body.append(row(run)));
    }
    const groups = { active: 0, prepared: 0, done: 0, failed: 0 };
    runs.forEach((run) => { const group = window.AF.stateGroup(run.state); if (group in groups) groups[group] += 1; });
    document.getElementById("metric-active").textContent = groups.active;
    document.getElementById("metric-review").textContent = groups.prepared;
    document.getElementById("metric-complete").textContent = groups.done;
    document.getElementById("metric-failed").textContent = groups.failed;
    document.getElementById("count-all").textContent = runs.length;
  }

  async function refresh() {
    const note = document.getElementById("poll-note");
    try {
      const response = await fetch(window.AF.url("/jobs/status"));
      if (!response.ok) throw new Error("status " + response.status);
      runs = (await response.json()).runs || [];
      render();
      note.innerHTML = "<i></i> Live";
      note.title = "Updated " + new Date().toLocaleTimeString();
    } catch (error) {
      note.textContent = "Update interrupted";
      note.title = String(error);
    }
  }

  async function refreshLog() {
    if (!following) return;
    try {
      const response = await fetch(window.AF.url("/jobs/log/" + encodeURIComponent(following.id)));
      const nearBottom = logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight < 70;
      logBody.textContent = (await response.text()) || "(waiting for process output)";
      if (nearBottom) logBody.scrollTop = logBody.scrollHeight;
    } catch (_) { /* retry on the next polling interval */ }
  }

  function follow(run) {
    following = { id: run.run_id };
    logTitle.textContent = run.target + " · " + window.AF.methodName(run.sampler);
    logPanel.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    refreshLog();
    document.getElementById("log-close").focus();
  }

  function closeLog() {
    following = null;
    logPanel.classList.add("hidden");
    document.body.style.overflow = "";
  }

  document.getElementById("state-filters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-state]");
    if (!button) return;
    filter = button.dataset.state;
    document.querySelectorAll("#state-filters button").forEach((item) => item.classList.toggle("active", item === button));
    render();
  });
  search.addEventListener("input", render);
  document.getElementById("log-close").addEventListener("click", closeLog);
  document.getElementById("log-scrim").addEventListener("click", closeLog);
  document.getElementById("copy-log").addEventListener("click", async () => {
    await navigator.clipboard.writeText(logBody.textContent);
    window.AF.toast("Log copied to clipboard.");
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && following) closeLog(); });

  refresh();
  setInterval(refresh, 3000);
  setInterval(refreshLog, 1800);
})();
