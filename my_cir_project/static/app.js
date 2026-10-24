/* PixelPhrase — vanilla JS. Plug real data in where marked "PLUG". */
(() => {
  // PLUG: set false once your Flask endpoints exist.
  const MOCK = false;
  const API_QUERY = "/api/query";     // POST multipart: image, instruction -> {base_caption, target_caption, results:[{id,src,caption,score}]}
  const API_GALLERY = "/api/gallery"; // GET -> [{id,src,caption}]
  const TOP_K = 6;

  const MOCK_GALLERY = [
    ["g01","a red barn in a green field under a clear sky"],
    ["g02","a snowy mountain cabin with smoke from the chimney"],
    ["g03","a black and white photo of a city street"],
    ["g04","a golden retriever running on a beach"],
    ["g05","a city skyline at night with lit windows"],
    ["g06","a bowl of ramen on a wooden table"],
    ["g07","a red barn covered in snow in winter"],
    ["g08","a cyclist on a country road in autumn"],
    ["g09","a lighthouse on a rocky coast at sunset"],
    ["g10","a monochrome portrait of an old fisherman"],
    ["g11","a pine forest dusted with fresh snow"],
    ["g12","a yellow taxi on a rainy street at night"],
  ].map(([id, caption]) => ({ id, caption, src: `gallery/${id}.jpg` }));

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("query-form"), drop: $("dropzone"), file: $("file-input"), img: $("dz-img"),
    name: $("dz-name"), clear: $("dz-clear"), instr: $("instruction"), chips: $("chips"),
    btn: $("submit-btn"), btnText: document.querySelector("#submit-btn .btn-text"),
    elapsed: $("elapsed"), note: $("form-note"),
    capBase: $("cap-base"), capTarget: $("cap-target"),
    empty: $("state-empty"), error: $("state-error"), errorMsg: $("error-msg"), retry: $("retry-btn"),
    results: $("results-grid"), resultsMeta: $("results-meta"),
    gallery: $("gallery-grid"), galleryMeta: $("gallery-meta"),
  };

  let file = null, timer = null, busy = false;

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));

  const thumb = (item, extra = "") => `
    <div class="thumb">
      <span class="thumb-label">${esc(item.id)}</span>
      ${item.src ? `<img src="${esc(item.src)}" alt="${esc(item.caption)}" loading="lazy" onerror="this.remove()">` : ""}
      ${extra}
    </div>`;

  /* ---------- Public render API ---------- */
  function renderGallery(items) {
    el.gallery.innerHTML = items.map((it) => `
      <article class="card">
        ${thumb(it)}
        <div class="card-body">
          <p class="card-caption">${esc(it.caption)}</p>
          <span class="card-id">${esc(it.id)}</span>
        </div>
      </article>`).join("");
    el.galleryMeta.textContent = `${items.length} images`;
  }

  function renderResults(items) {
    if (!items || !items.length) {
      setState("error", "No matches came back for that instruction. Try rephrasing it.");
      return;
    }
    el.results.innerHTML = items.map((it, i) => {
      const pct = Math.max(0, Math.min(1, it.score)) * 100;
      return `
      <article class="card${i === 0 ? " top" : ""}">
        ${thumb(it, `<span class="rank">#${i + 1}</span>`)}
        <div class="card-body">
          <p class="card-caption">${esc(it.caption)}</p>
          <div class="score" title="Cosine similarity">
            <span class="score-bar"><i style="width:${pct.toFixed(1)}%"></i></span>
            <span class="score-val">${Number(it.score).toFixed(3)}</span>
          </div>
        </div>
      </article>`;
    }).join("");
    el.resultsMeta.textContent = `top ${items.length} · cosine similarity`;
    setState("results");
  }

  function setCaptions(base, target) {
    [[el.capBase, base], [el.capTarget, target]].forEach(([node, text]) => {
      node.textContent = text || "—";
      node.classList.toggle("is-empty", !text);
    });
  }

  function renderSkeletons(n = TOP_K) {
    el.results.innerHTML = Array.from({ length: n }, () => `
      <article class="card skeleton" aria-hidden="true">
        <div class="thumb"></div>
        <div class="card-body">
          <div class="skel skel-line" style="width:90%"></div>
          <div class="skel skel-line" style="width:60%"></div>
          <div class="skel skel-line" style="width:100%;height:4px;margin-top:6px"></div>
        </div>
      </article>`).join("");
  }

  // state: "empty" | "loading" | "results" | "error"
  function setState(state, msg) {
    el.empty.hidden = state !== "empty";
    el.error.hidden = state !== "error";
    el.results.hidden = !(state === "results" || state === "loading");
    if (state === "error" && msg) el.errorMsg.textContent = msg;
    if (state === "loading") {
      renderSkeletons();
      el.resultsMeta.textContent = "searching…";
      [el.capBase, el.capTarget].forEach((n) => { n.innerHTML = '<span class="skel skel-line" style="display:block;width:80%"></span>'; n.classList.remove("is-empty"); });
    }
    if (state === "empty" || state === "error") el.resultsMeta.textContent = "";
    setBusy(state === "loading");
  }

  /* ---------- Busy button ---------- */
  function setBusy(on) {
    busy = on;
    el.btn.classList.toggle("is-busy", on);
    el.btn.setAttribute("aria-busy", on);
    el.btnText.textContent = on ? "Searching…" : "Find matches";
    el.instr.disabled = on;
    el.file.disabled = on;
    clearInterval(timer);
    if (on) {
      const t0 = performance.now();
      el.elapsed.textContent = "0.0s";
      timer = setInterval(() => { el.elapsed.textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"; }, 100);
      el.note.textContent = "Captioning, rewriting and ranking — this can take a few seconds.";
    }
    validate();
  }

  function validate() {
    const ok = !!file && el.instr.value.trim().length > 0;
    el.btn.disabled = busy || !ok;
    if (!busy) el.note.textContent = ok ? "Press Enter or click to search." : !file ? "Add a photo to get started." : "Now describe the edit.";
  }

  /* ---------- Dropzone ---------- */
  function setFile(f) {
    if (f && !f.type.startsWith("image/")) { el.note.textContent = "That file isn’t an image."; return; }
    file = f || null;
    if (el.img.src.startsWith("blob:")) URL.revokeObjectURL(el.img.src);
    if (file) { el.img.src = URL.createObjectURL(file); el.name.textContent = file.name; }
    else { el.img.removeAttribute("src"); el.file.value = ""; }
    el.drop.classList.toggle("has-file", !!file);
    validate();
  }

  el.file.addEventListener("change", () => setFile(el.file.files[0]));
  el.clear.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); setFile(null); });
  ["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); if (!busy) el.drop.classList.add("is-drag"); }));
  ["dragleave", "dragend", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { if (t !== "dragleave" || !el.drop.contains(e.relatedTarget)) el.drop.classList.remove("is-drag"); }));
  el.drop.addEventListener("drop", (e) => { e.preventDefault(); if (!busy && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

  el.instr.addEventListener("input", validate);
  el.chips.addEventListener("click", (e) => {
    if (e.target.matches(".chip") && !busy) { el.instr.value = e.target.textContent; validate(); el.instr.focus(); }
  });

  /* ---------- Query ---------- */
  async function runQuery() {
    if (busy || !file || !el.instr.value.trim()) return;
    setState("loading");
    try {
      const data = MOCK ? await mockQuery(el.instr.value.trim()) : await realQuery();
      setCaptions(data.base_caption, data.target_caption);
      renderResults(data.results);
    } catch (err) {
      setCaptions("", "");
      setState("error", err.message || "Request failed. Try again.");
    }
  }

  async function realQuery() {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("instruction", el.instr.value.trim());
    fd.append("k", TOP_K);
    const res = await fetch(API_QUERY, { method: "POST", body: fd });
    if (!res.ok) {
      let message = `Server error (${res.status}). Try again.`;
      try { const body = await res.json(); if (body.error) message = body.error; } catch {}
      throw new Error(message);
    }
    return res.json();
  }

  function mockQuery(instr) {
    return new Promise((resolve) => setTimeout(() => {
      const words = instr.toLowerCase().split(/\W+/).filter((w) => w.length > 2);
      const results = MOCK_GALLERY
        .map((g) => ({ ...g, score: 0.42 + words.filter((w) => g.caption.includes(w)).length * 0.18 + Math.random() * 0.12 }))
        .sort((a, b) => b.score - a.score).slice(0, TOP_K)
        .map((r) => ({ ...r, score: Math.min(r.score, 0.97) }));
      resolve({ base_caption: "a red barn in a green field under a clear sky", target_caption: `a red barn in a field under a clear sky, ${instr}`, results });
    }, 2600));
  }

  el.form.addEventListener("submit", (e) => { e.preventDefault(); runQuery(); });
  el.retry.addEventListener("click", runQuery);

  /* ---------- Init ---------- */
  async function loadGallery() {
    if (MOCK) return renderGallery(MOCK_GALLERY);
    try { renderGallery(await (await fetch(API_GALLERY)).json()); }
    catch { el.gallery.innerHTML = '<div class="state error"><h3>Couldn’t load the gallery</h3></div>'; }
  }

  loadGallery();
  setState("empty");
  validate();

  // Preview helper: index.html?state=loading|results|error|empty
  const demo = new URLSearchParams(location.search).get("state");
  if (demo === "loading") { file = new File([""], "reference.jpg", { type: "image/jpeg" }); el.drop.classList.add("has-file"); el.name.textContent = "reference.jpg"; el.instr.value = "add snow"; setState("loading"); }
  if (demo === "results") mockQuery("add snow").then((d) => { el.instr.value = "add snow"; setCaptions(d.base_caption, d.target_caption); renderResults(d.results); });
  if (demo === "error") setState("error");

  // Expose for manual wiring / debugging
  window.PixelPhrase = { renderGallery, renderResults, setCaptions, setState };
})();
