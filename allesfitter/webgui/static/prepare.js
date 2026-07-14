// Prepare page: POST /prepare/run then hand off to the Jobs dashboard.
(function () {
  const $ = (id) => document.getElementById(id);
  const el = window.AF.el;
  const result = $("result");
  const btn = $("prepare");
  const missionDefaults = {
    tess: { segment: "TESS sector(s)", instrument: "tess" },
    k2: { segment: "K2 campaign(s)", instrument: "k2" },
    kepler: { segment: "Kepler quarter(s)", instrument: "kepler" },
  };

  // --- sector/campaign/quarter hint --------------------------------------
  // Live-looks-up what -s/-c/-q would actually accept for this target before
  // the user submits and waits on a download. Click a chip to toggle it into
  // the (space-separated, possibly multi-value) sectors field.
  const sectorsInput = $("sectors");
  const sectorHint = $("sector-hint");
  const exptimeSelect = $("exptime");
  let sectorSegments = null; // null = unknown, [] = none found, [str...] = found
  let lastSectorQuery = null; // dedupe identical target+id_type+mission+pipeline lookups

  // Canonical TESS cadences (seconds). For TESS the dropdown always offers
  // exactly these; other missions fall back to what the archive lookup reports.
  const TESS_EXPTIMES = [20, 120, 600, 1800];

  // Rebuild the exposure-time <select>, always keeping an "Auto" choice first
  // and preserving the current selection if still valid.
  function renderExptimeOptions(archiveExptimes) {
    const mission = ($("mission").value || "").toLowerCase();
    const list =
      mission === "tess"
        ? TESS_EXPTIMES
        : Array.isArray(archiveExptimes)
          ? archiveExptimes
          : [];
    const prev = exptimeSelect.value;
    const options = [el("option", { value: "" }, "Auto (any available)")];
    for (const e of list) options.push(el("option", { value: String(e) }, e + " s"));
    exptimeSelect.replaceChildren(...options);
    exptimeSelect.value = list.map(String).includes(prev) ? prev : "";
    updateSummary();
  }

  function currentSectorTokens() {
    return new Set(sectorsInput.value.trim().split(/\s+/).filter(Boolean));
  }

  function renderSectorChips() {
    if (!sectorHint || sectorSegments === null) return;
    if (sectorSegments.length === 0) {
      sectorHint.replaceChildren(el("span", { class: "sector-warn" }, "No sectors found."));
      return;
    }
    const current = currentSectorTokens();
    const nodes = [el("span", {}, "Available:")];
    for (const seg of sectorSegments) {
      const cls = current.has(seg) ? "sector-chip active" : "sector-chip";
      const chip = el("button", { type: "button", class: cls, title: "Toggle " + seg }, seg);
      chip.addEventListener("click", () => {
        const tokens = currentSectorTokens();
        if (tokens.has(seg)) tokens.delete(seg);
        else tokens.add(seg);
        sectorsInput.value = Array.from(tokens).join(" ");
        renderSectorChips();
        updateSummary();
      });
      nodes.push(chip);
    }
    sectorHint.replaceChildren(...nodes);
  }

  async function refreshSectorHint() {
    if (!sectorHint) return;
    const target = $("target").value.trim();
    if (!target) {
      sectorHint.replaceChildren();
      sectorSegments = null;
      lastSectorQuery = null;
      renderExptimeOptions([]);
      return;
    }
    const idType = $("id-type").value;
    const mission = $("mission").value;
    const pipeline = $("pipeline").value;
    const key = [target, idType, mission, pipeline].join("|");
    if (key === lastSectorQuery) {
      renderSectorChips();
      return;
    }
    lastSectorQuery = key;
    sectorHint.replaceChildren(el("span", {}, "Looking up sectors…"));
    try {
      const params = new URLSearchParams({ target, id_type: idType, mission, pipeline });
      const res = await fetch(window.AF.url("/prepare/sectors?" + params.toString()));
      const data = await res.json();
      if (!data.ok) {
        sectorSegments = [];
        sectorHint.replaceChildren(
          el("span", { class: "sector-warn" }, data.error || "No sectors found."),
        );
        renderExptimeOptions([]);
        return;
      }
      sectorSegments = data.segments || [];
      renderSectorChips();
      renderExptimeOptions(data.exptimes);
    } catch (e) {
      sectorHint.replaceChildren(el("span", { class: "sector-warn" }, "Could not fetch: " + e));
      sectorSegments = null;
      lastSectorQuery = null;
    }
  }

  $("target").addEventListener("blur", refreshSectorHint);
  $("mission").addEventListener("change", () => {
    lastSectorQuery = null;
    const defaults = missionDefaults[$("mission").value] || missionDefaults.tess;
    $("segment-label").textContent = defaults.segment;
    if (!$("filename").dataset.touched) $("filename").value = defaults.instrument;
    autofillBandpass();
    renderExptimeOptions([]); // instant: fixed TESS set, or clear pending archive list
    updateSummary();
    refreshSectorHint();
  });
  $("pipeline").addEventListener("change", () => {
    lastSectorQuery = null;
    refreshSectorHint();
  });
  sectorsInput.addEventListener("input", renderSectorChips);

  function value(id) { return $(id).value.trim(); }

  function updateSummary() {
    const target = value("target");
    const mission = value("mission").toUpperCase();
    const pipeline = value("pipeline").toUpperCase();
    const sectors = value("sectors");
    const bands = value("bandpass");
    const exptime = value("exptime");
    $("summary-target").textContent = target || "Untitled target";
    $("summary-source").textContent =
      mission + " · " + pipeline + (exptime ? " · " + exptime + " s" : "");
    $("summary-sectors").textContent = sectors || "Not selected";
    $("summary-model").textContent = bands ? "Chromatic · " + bands : "Achromatic";
    $("check-target").classList.toggle("ready", Boolean(target));
    $("check-sectors").classList.toggle("ready", Boolean(sectors));
  }

  // Default the bandpass to "tess" when the sole instrument label is tess, so a
  // single-instrument TESS run is band-labelled without extra typing. Never
  // overrides a bandpass the user typed; clears a prior autofill otherwise.
  function autofillBandpass() {
    const bandpass = $("bandpass");
    if (bandpass.dataset.touched) return;
    const insts = value("filename").split(/\s+/).filter(Boolean);
    const onlyTess = insts.length === 1 && insts[0].toLowerCase() === "tess";
    bandpass.value = onlyTess ? "tess" : "";
    updateSummary();
  }

  ["target", "sectors", "pipeline", "bandpass", "filename"].forEach((id) => {
    $(id).addEventListener("input", updateSummary);
    $(id).addEventListener("change", updateSummary);
  });
  $("filename").addEventListener("input", () => {
    $("filename").dataset.touched = "true";
    autofillBandpass();
  });
  $("filename").addEventListener("change", autofillBandpass);
  $("exptime").addEventListener("change", updateSummary);
  // Remember an explicit bandpass so autofill never clobbers a user value.
  $("bandpass").addEventListener("input", () => { $("bandpass").dataset.touched = "true"; });

  function validate() {
    const missing = [];
    for (const [id, label] of [["target", "target identifier"], ["sectors", "observation window"]]) {
      const input = $(id);
      const invalid = !input.value.trim();
      input.setAttribute("aria-invalid", String(invalid));
      if (invalid) missing.push({ input, label });
    }
    const files = value("filename").split(/\s+/).filter(Boolean);
    const bands = value("bandpass").split(/\s+/).filter(Boolean);
    if (bands.length && bands.length !== files.length) {
      $("bandpass").setAttribute("aria-invalid", "true");
      show("Provide one bandpass label per instrument, or leave bandpasses blank.", "err");
      $("bandpass").focus();
      return false;
    }
    $("bandpass").setAttribute("aria-invalid", "false");
    if (missing.length) {
      show("Add a " + missing.map((item) => item.label).join(" and ") + " to continue.", "err");
      missing[0].input.focus();
      return false;
    }
    return true;
  }

  function payload() {
    return {
      target: $("target").value.trim(),
      id_type: $("id-type").value,
      mission: $("mission").value,
      sectors: $("sectors").value.trim(),
      pipeline: $("pipeline").value,
      exptime: $("exptime").value,
      lc_type: $("lc-type").value,
      quality: $("quality").value,
      sigma: $("sigma").value.trim(),
      filename: $("filename").value.trim(),
      bandpass: $("bandpass").value.trim(),
      sampler: $("sampler").value,
      ttv: $("ttv").checked,
      overwrite: $("overwrite").checked,
    };
  }

  function show(msg, cls) {
    result.className = "result " + (cls || "");
    result.textContent = msg;
  }

  btn.addEventListener("click", async () => {
    if (!validate()) return;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-inline" aria-hidden="true"></span> Starting preparation…';
    show("Creating a reproducible workspace and contacting the archive…", "");
    try {
      const { ok, data } = await window.AF.postJSON(window.AF.url("/prepare/run"), payload());
      if (ok && data && data.run_id) {
        show("Preparing " + data.run_id + " — opening review page…", "ok");
        window.location.href = window.AF.url("/results/" + encodeURIComponent(data.run_id));
        return;
      }
      show((data && data.detail) || "Prepare failed.", "err");
    } catch (e) {
      show("Network error: " + e, "err");
    } finally {
      btn.disabled = false;
      btn.innerHTML = 'Prepare observations <span aria-hidden="true">→</span>';
    }
  });

  $("target").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      refreshSectorHint();
    }
  });
  autofillBandpass();
  renderExptimeOptions([]); // seed the dropdown (TESS is the default mission)
  updateSummary();
})();
