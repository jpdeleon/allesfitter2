// Prepare page: POST /prepare/run then hand off to the Jobs dashboard.
(function () {
  const $ = (id) => document.getElementById(id);
  const result = $("result");
  const btn = $("prepare");

  function payload() {
    return {
      target: $("target").value.trim(),
      id_type: $("id-type").value,
      mission: $("mission").value,
      sectors: $("sectors").value.trim(),
      pipeline: $("pipeline").value,
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
    btn.disabled = true;
    show("Launching prepare…", "");
    try {
      const { ok, data } = await window.AF.postJSON("/prepare/run", payload());
      if (ok && data && data.run_id) {
        show("Prepared run " + data.run_id + " — redirecting to Jobs…", "ok");
        window.location.href = "/jobs";
        return;
      }
      show((data && data.detail) || "Prepare failed.", "err");
    } catch (e) {
      show("Network error: " + e, "err");
    } finally {
      btn.disabled = false;
    }
  });
})();
