// Run workspace: scientific review, fitting actions, live state, and figure inspection.
(function () {
  const workspace = document.getElementById("run-workspace");
  if (!workspace) return;
  const runId = workspace.dataset.runId;
  const initialState = workspace.dataset.state;
  const activeStates = new Set(["created", "pending", "preparing", "running", "stopping"]);
  let currentState = initialState;

  function validationMessage(message, kind, detail) {
    const result = document.getElementById("config-result");
    if (!result) return;
    result.className = "validation-status " + (kind || "");
    const icon = document.createElement("span");
    icon.className = "status-icon";
    icon.textContent = kind === "working" ? "↻" : kind === "error" ? "!" : "✓";
    const copy = document.createElement("p");
    const strong = document.createElement("strong");
    const small = document.createElement("small");
    strong.textContent = message;
    small.textContent = detail || "";
    copy.append(strong, small);
    result.replaceChildren(icon, copy);
  }

  // Configuration review and validation.
  const editor = document.getElementById("config-editor");
  let dirty = false;
  if (editor) {
    const save = document.getElementById("save-config");
    const samplerButtons = Array.from(editor.parentElement.querySelectorAll("[data-sampler]"));
    const textareas = [document.getElementById("params-csv"), document.getElementById("settings-csv")];
    const originals = new Map(textareas.map((area) => [area.id, area.value]));

    function updateEditor(area) {
      const changed = area.value !== originals.get(area.id);
      const tab = document.querySelector('[aria-controls="view-' + (area.id === "params-csv" ? "params" : "settings") + '"]');
      if (tab) tab.classList.toggle("dirty", changed);
      const lines = area.value ? area.value.split("\n").length : 0;
      const stats = document.querySelector('[data-stats="' + area.id + '"]');
      if (stats) stats.textContent = lines + " lines · " + area.value.length.toLocaleString() + " characters";
      dirty = textareas.some((item) => item.value !== originals.get(item.id));
    }

    textareas.forEach((area) => {
      area.addEventListener("input", () => updateEditor(area));
      area.addEventListener("keydown", (event) => {
        if (event.key === "Tab") {
          event.preventDefault();
          const start = area.selectionStart;
          area.setRangeText("  ", start, area.selectionEnd, "end");
          updateEditor(area);
        }
      });
      updateEditor(area);
    });

    document.querySelectorAll("[data-review-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-review-tab]").forEach((item) => {
          const active = item === button;
          item.classList.toggle("active", active);
          item.setAttribute("aria-selected", String(active));
        });
        document.querySelectorAll(".review-view").forEach((view) => {
          const active = view.id === "view-" + button.dataset.reviewTab;
          view.classList.toggle("active", active);
          view.hidden = !active;
        });
      });
    });

    document.querySelectorAll(".editor-reset").forEach((button) => {
      button.addEventListener("click", () => {
        const area = document.getElementById(button.dataset.editor);
        area.value = originals.get(area.id);
        updateEditor(area);
        window.AF.toast("Unsaved edits discarded.");
      });
    });

    async function saveConfig() {
      save.disabled = true;
      samplerButtons.forEach((button) => { button.disabled = true; });
      validationMessage("Validating configuration…", "working", "Parsing the model and rendering a fresh initial guess.");
      try {
        const response = await window.AF.postJSON(
          window.AF.url("/results/" + encodeURIComponent(runId) + "/config"),
          { params_csv: textareas[0].value, settings_csv: textareas[1].value },
        );
        if (!response.ok || !response.data || !response.data.ok) {
          validationMessage("Validation failed", "error", (response.data && (response.data.error || response.data.detail)) || "Check the CSV configuration and try again.");
          return false;
        }
        textareas.forEach((area) => { originals.set(area.id, area.value); updateEditor(area); });
        const image = document.getElementById("initial-guess");
        if (image) image.src = window.AF.url("/results/" + encodeURIComponent(runId) + "/initial-guess?t=" + Date.now());
        const count = response.data.n_free_params;
        validationMessage("Configuration saved and valid", "", count == null ? "Initial model preview refreshed." : count + " free parameters · initial model preview refreshed.");
        window.AF.toast("Model validated successfully.");
        return true;
      } catch (error) {
        validationMessage("Could not validate", "error", "Connection error: " + error);
        return false;
      } finally {
        save.disabled = false;
        samplerButtons.forEach((button) => { button.disabled = false; });
      }
    }

    save.addEventListener("click", saveConfig);
    samplerButtons.forEach((button) => {
      button.addEventListener("click", async () => {
        if (!(await saveConfig())) return;
        const sampler = button.dataset.sampler;
        samplerButtons.forEach((item) => { item.disabled = true; });
        validationMessage("Starting " + window.AF.methodName(sampler) + "…", "working", "Submitting the validated model to the local fit queue.");
        let url = "/jobs/fit/" + encodeURIComponent(runId) + "?sampler=" + encodeURIComponent(sampler);
        if (sampler === "optimize") {
          url += "&method=" + encodeURIComponent(document.getElementById("optimize-method").value) +
            "&refine=" + encodeURIComponent(document.getElementById("optimize-refine").checked) +
            "&restarts=" + encodeURIComponent(document.getElementById("optimize-restarts").value);
        }
        try {
          const response = await window.AF.postJSON(window.AF.url(url), {});
          if (response.ok) {
            dirty = false;
            window.location.reload();
          } else {
            samplerButtons.forEach((item) => { item.disabled = false; });
            validationMessage("Fit could not start", "error", (response.data && response.data.detail) || "The run was not accepted by the queue.");
          }
        } catch (error) {
          samplerButtons.forEach((item) => { item.disabled = false; });
          validationMessage("Fit could not start", "error", "Connection error: " + error);
        }
      });
    });

    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveConfig();
      }
    });
    window.addEventListener("beforeunload", (event) => { if (dirty) event.preventDefault(); });
  }

  // Figure filtering and lightbox.
  document.querySelectorAll("[data-figure-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-figure-filter]").forEach((item) => item.classList.toggle("active", item === button));
      document.querySelectorAll("#figure-gallery figure").forEach((figure) => {
        figure.classList.toggle("hidden", button.dataset.figureFilter !== "all" && figure.dataset.category !== button.dataset.figureFilter);
      });
    });
  });
  const lightbox = document.getElementById("figure-lightbox");
  function closeLightbox() { lightbox.classList.add("hidden"); document.body.style.overflow = ""; }
  document.querySelectorAll(".figure-open").forEach((button) => {
    button.addEventListener("click", () => {
      lightbox.querySelector("img").src = button.dataset.fullSrc;
      lightbox.querySelector("img").alt = button.dataset.caption;
      lightbox.querySelector("p").textContent = button.dataset.caption;
      lightbox.classList.remove("hidden");
      document.body.style.overflow = "hidden";
      lightbox.querySelector("button").focus();
    });
  });
  lightbox.querySelector("button").addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", (event) => { if (event.target === lightbox) closeLightbox(); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !lightbox.classList.contains("hidden")) closeLightbox(); });

  // Live process status and log. Reload only when a lifecycle transition changes
  // which scientific controls or outputs should be rendered.
  async function refreshStatus() {
    if (!activeStates.has(currentState)) return;
    try {
      const response = await fetch(window.AF.url("/jobs/status"));
      const run = (await response.json()).runs.find((item) => item.run_id === runId);
      if (!run) return;
      const oldGroup = window.AF.stateGroup(currentState);
      const newGroup = window.AF.stateGroup(run.state);
      currentState = run.state;
      document.querySelectorAll("#run-state-badge, #sidebar-state").forEach((badge) => {
        badge.className = "badge state-" + run.state;
        badge.textContent = run.state;
      });
      document.getElementById("sidebar-sampler").textContent = window.AF.methodName(run.sampler);
      document.getElementById("sidebar-logz").textContent = run.logz || "—";
      const stop = document.getElementById("stop-run");
      if (stop) stop.classList.toggle("hidden", !activeStates.has(run.state));
      if (oldGroup !== newGroup || ["prepared", "done", "failed"].includes(newGroup)) window.location.reload();
    } catch (_) { /* retry on next poll */ }
  }

  async function refreshLog() {
    const log = document.getElementById("live-run-log");
    if (!log || !activeStates.has(currentState)) return;
    try {
      const response = await fetch(window.AF.url("/jobs/log/" + encodeURIComponent(runId)));
      const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 70;
      log.textContent = (await response.text()) || "(waiting for process output)";
      if (nearBottom) log.scrollTop = log.scrollHeight;
    } catch (_) { /* retry on next poll */ }
  }

  const stop = document.getElementById("stop-run");
  if (stop) {
    stop.classList.toggle("hidden", !activeStates.has(currentState));
    stop.addEventListener("click", async () => {
      stop.disabled = true;
      const response = await window.AF.postJSON(window.AF.url("/jobs/stop/" + encodeURIComponent(runId)), {});
      window.AF.toast(response.ok ? "Stop signal sent." : "Could not stop run.", response.ok ? "" : "error");
      stop.disabled = false;
    });
  }
  const copyLog = document.getElementById("copy-run-log");
  if (copyLog) copyLog.addEventListener("click", async () => {
    await navigator.clipboard.writeText(document.getElementById("live-run-log").textContent);
    window.AF.toast("Log copied to clipboard.");
  });
  document.querySelectorAll("pre.log").forEach((log) => { log.scrollTop = log.scrollHeight; });
  refreshStatus();
  refreshLog();
  setInterval(refreshStatus, 3000);
  setInterval(refreshLog, 1800);
})();
