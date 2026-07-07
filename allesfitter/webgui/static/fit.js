// New-fit form: dynamic instrument rows -> JSON payload -> validate / run.
(function () {
  const $ = (id) => document.getElementById(id);
  const body = $("inst-body");
  const tpl = $("inst-row");

  function addRow(preset) {
    const row = tpl.content.firstElementChild.cloneNode(true);
    if (preset) {
      row.querySelector(".i-label").value = preset.label || "";
      row.querySelector(".i-band").value = preset.band || "";
      row.querySelector(".i-baseline").value = preset.baseline || "sample_linear";
    }
    row.querySelector(".del").addEventListener("click", () => row.remove());
    // suggest band from label as the user types (only if band is empty)
    const label = row.querySelector(".i-label");
    const band = row.querySelector(".i-band");
    label.addEventListener("blur", async () => {
      if (band.value.trim() || !label.value.trim()) return;
      // lightweight heuristic mirrored server-side; leave blank if unsure
      const m = label.value.toLowerCase();
      if (/qlp|spoc|tess/.test(m)) band.value = "tess";
      else { const t = m.replace(/[0-9]+$/, ""); const c = t.match(/([ugriz y])(?!.*[ugriz])/); if (c) band.value = c[1]; }
    });
    body.append(row);
    return row;
  }

  function collect() {
    const insts = [];
    for (const tr of body.querySelectorAll("tr")) {
      const label = tr.querySelector(".i-label").value.trim();
      if (!label) continue;
      const cols = tr.querySelector(".i-cols").value.trim();
      const texp = tr.querySelector(".i-texp").value;
      insts.push({
        label,
        band: tr.querySelector(".i-band").value.trim(),
        data_file: tr.querySelector(".i-file").value.trim(),
        baseline: tr.querySelector(".i-baseline").value,
        baseline_cols: cols ? cols.split(/\s+/) : [],
        ld_law: tr.querySelector(".i-ldlaw").value,
        t_exp: texp === "" ? null : Number(texp),
        dilution: tr.querySelector(".i-dil").checked,
      });
    }
    const shareRaw = $("share-groups").value.trim();
    const share_groups = shareRaw
      ? shareRaw.split(/\s+/).map((g) => g.split(":").filter(Boolean)).filter((g) => g.length > 1)
      : [];
    return {
      target: $("target").value.trim(),
      companions: [{
        name: "b",
        period: Number($("c-period").value),
        period_err: $("c-period-err").value === "" ? null : Number($("c-period-err").value),
        epoch: Number($("c-epoch").value),
        epoch_err: $("c-epoch-err").value === "" ? null : Number($("c-epoch-err").value),
        rr: Number($("c-rr").value),
      }],
      instruments: insts,
      share_groups,
      chromatic: $("chromatic").value,
      sampler: $("sampler").value,
      multiprocess_cores: Number($("cores").value) || 4,
      use_host_density_prior: $("rho").checked,
      fast_fit: $("fastfit").checked,
      fit_ttvs: $("ttvs").checked,
    };
  }

  function show(msg, ok) {
    const r = $("result");
    r.textContent = msg;
    r.className = "result " + (ok ? "ok" : "err");
  }

  $("add-inst").addEventListener("click", () => addRow());

  $("autofill").addEventListener("click", async () => {
    const target = $("target").value.trim();
    if (!target) { show("Enter a target first.", false); return; }
    const res = await fetch("/catalog?target=" + encodeURIComponent(target));
    const eph = await res.json();
    if (eph.period) $("c-period").value = eph.period;
    if (eph.epoch) $("c-epoch").value = eph.epoch;
    $("autofill-src").textContent = eph.source === "none" ? "not found" : eph.source;
  });

  $("validate").addEventListener("click", async () => {
    show("Validating…", true);
    const { ok, data } = await window.AF.postJSON("/fit/validate", collect());
    if (ok && data && data.ok) show(`Valid ✓  (${data.n_free_params} free parameters)`, true);
    else show("Invalid: " + ((data && data.error) || "unknown error"), false);
  });

  $("run").addEventListener("click", async () => {
    show("Launching…", true);
    const { ok, data } = await window.AF.postJSON("/fit/run", collect());
    if (ok && data && data.run_id) location.href = "/results/" + data.run_id;
    else show("Could not start: " + ((data && data.detail) || (data && data.error) || "error"), false);
  });

  // start with one blank instrument row
  addRow();
})();
