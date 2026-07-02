// atomic-notes GUI — Frontend-Logik (vanilla JS, SSE via EventSource).
// Bewusst ohne Framework/CDN: laeuft komplett offline.

const STAGES = [
  [1, "PDF & Chunking"], [2, "Vault-Kontext"], [3, "Quellen-Qualität"],
  [4, "Planner"], [5, "Extractor"], [6, "Verifier & Critic"],
  [7, "Vault-Writer"], [8, "Qualitäts-Eval"],
];

const $ = (id) => document.getElementById(id);
let currentPdfStem = "";
let running = false;
let userCancelled = false;
let activeStage = 0;
let stageStartedAt = 0;
let elapsedTimer = null;

function renderStepper() {
  const ol = $("stepper");
  ol.innerHTML = "";
  for (const [num, label] of STAGES) {
    const li = document.createElement("li");
    li.id = `step-${num}`;
    li.innerHTML = `<span class="dot" aria-hidden="true"></span><span class="step-label">${num}. ${label}</span><span class="elapsed" aria-hidden="true"></span>`;
    ol.appendChild(li);
  }
}

function setStage(num) {
  if (num !== activeStage) { activeStage = num; stageStartedAt = Date.now(); }
  for (const [n] of STAGES) {
    const li = $(`step-${n}`);
    if (!li) continue;
    li.classList.toggle("done", n < num);
    li.classList.toggle("active", n === num);
    li.classList.remove("error");
    if (n !== num) { const e = li.querySelector(".elapsed"); if (e) e.textContent = ""; }
  }
}

function markStageError(num) {
  const li = $(`step-${num}`);
  if (li) { li.classList.remove("active"); li.classList.add("error"); }
}

function tickElapsed() {
  // NN/g: bei der laufenden Stufe ab ~1s die verstrichene Zeit zeigen (Systemstatus).
  const li = $(`step-${activeStage}`);
  if (!li || !running) return;
  const s = Math.floor((Date.now() - stageStartedAt) / 1000);
  const e = li.querySelector(".elapsed");
  if (e) e.textContent = s >= 1 ? `${s}s` : "";
}

function startElapsed() {
  $("stepper").setAttribute("aria-busy", "true");
  stopElapsed();
  elapsedTimer = setInterval(tickElapsed, 1000);
}

function stopElapsed() {
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  $("stepper").removeAttribute("aria-busy");
}

function resetRun() {
  renderStepper();
  activeStage = 0;
  $("preview-list").innerHTML = "";
  $("preview-empty").style.display = "block";
  $("note-progress").textContent = "";
  $("run-header").textContent = "";
  $("log").textContent = "";
  const b = $("error-banner"); b.hidden = true; b.textContent = "";
  $("results-section").hidden = true;
  $("results-list").innerHTML = "";
  $("results-empty").style.display = "block";
  $("download-all").hidden = true;
}

function buildRunOptions() {
  // Nur tatsächlich vom Server-Default abweichende Werte mitschicken —
  // leere Felder bedeuten "Server-Default" (Rückwärtskompatibilität, siehe P1).
  const options = {};
  const backend = $("opt-backend").value;
  const profile = $("opt-profile").value;
  if (backend) options.backend = backend;
  if (profile) options.profile = profile;
  if ($("opt-no-llm").checked) options.no_llm = true;
  return options;
}

function renderRunHeader(options) {
  const backend = options.backend || "Server-Default";
  const profile = options.profile || "Server-Default";
  const noLlm = options.no_llm ? "ja" : "nein";
  $("run-header").textContent = `Backend: ${backend} · Profil: ${profile} · ohne LLM: ${noLlm}`;
}

function showBanner(text) {
  const b = $("error-banner");
  b.hidden = false;
  // mehrere Hinweise sammeln, Duplikate vermeiden
  if (![...b.children].some((c) => c.textContent === text)) {
    const p = document.createElement("div");
    p.textContent = "⚠ " + text;
    b.appendChild(p);
  }
}

function addPreview(ev) {
  $("preview-empty").style.display = "none";
  const li = document.createElement("li");
  li.className = "preview-card";
  const routingLabel = { vault: "Vault-Empfehlung", inbox: "Inbox-Review", merge: "Merge-Stub" }[ev.routing] || ev.routing;
  const confClass = { high: "ok", low: "warn" }[ev.confidence] || "";
  const flags = (ev.flags || "").trim();
  li.innerHTML = `
    <div class="title">${escapeHtml(ev.name)}</div>
    <div class="meta">
      <span class="badge ${ev.routing}">${routingLabel}</span>
      <span class="badge">Score ${ev.score ?? "?"}/5</span>
      <span class="badge">Hard-Gates ${ev.hard_gates ? "pass" : "fail"}</span>
      <span class="badge ${confClass}">Confidence ${escapeHtml(ev.confidence || "?")}</span>
      ${ev.reason ? `<span class="badge">${escapeHtml(ev.reason)}</span>` : ""}
      ${ev.merge_target ? `<span class="badge">→ ${escapeHtml(ev.merge_target)}</span>` : ""}
    </div>
    ${flags ? `<div class="flags">⚠ ${escapeHtml(flags)}</div>` : ""}
    <details class="note-body"><summary>Note-Text anzeigen</summary><pre class="body-content muted">…</pre></details>`;
  // Body lazy laden beim ersten Aufklappen (nur im Dry-Run vorhanden).
  const det = li.querySelector("details");
  const body = li.querySelector(".body-content");
  det.addEventListener("toggle", async () => {
    if (!det.open || det.dataset.loaded) return;
    det.dataset.loaded = "1";
    try {
      const r = await fetch(`/api/preview?pdf_stem=${encodeURIComponent(currentPdfStem)}&name=${encodeURIComponent(ev.name)}`);
      if (r.ok) { const d = await r.json(); body.textContent = d.body; body.classList.remove("muted"); }
      else { body.textContent = "(Note-Text nur nach einem Vorschau-Lauf verfügbar.)"; }
    } catch { body.textContent = "(Konnte Note-Text nicht laden.)"; }
  });
  $("preview-list").appendChild(li);
}

// --- Ergebnis-Sektion (P3: /api/outputs, nach `exited(rc=0)`) -------------

function routingLabel(routing) {
  return { vault: "Vault-Empfehlung", inbox: "Inbox-Review", merge: "Merge-Stub" }[routing] || routing;
}

function addResultCard(item) {
  const li = document.createElement("li");
  li.className = "preview-card";
  const confClass = { high: "ok", low: "warn" }[item.confidence] || "";
  const badges = [`<span class="badge ${item.routing}">${routingLabel(item.routing)}</span>`];
  // Score/Confidence gibt es nur im Dry-Run (vault_writer druckt sie im
  // Schreib-Lauf nicht) — nur anzeigen, wenn tatsächlich vorhanden (L5).
  if (item.score !== undefined) badges.push(`<span class="badge">Score ${item.score}/5</span>`);
  if (item.confidence !== undefined) badges.push(`<span class="badge ${confClass}">Confidence ${escapeHtml(item.confidence)}</span>`);
  if (item.merge_target) badges.push(`<span class="badge">→ ${escapeHtml(item.merge_target)}</span>`);
  const flags = (item.flags || "").trim();
  const downloadHtml = item.path
    ? `<a class="button" href="/api/outputs/file?path=${encodeURIComponent(item.path)}" download>Herunterladen</a>`
    : `<span class="hint">Kein Download verfügbar.</span>`;
  li.innerHTML = `
    <div class="title">${escapeHtml(item.title)}</div>
    <div class="meta">${badges.join("")}</div>
    ${flags ? `<div class="flags">⚠ ${escapeHtml(flags)}</div>` : ""}
    <details class="note-body"><summary>Note-Text anzeigen</summary><pre class="body-content muted">…</pre></details>
    <div class="actions">${downloadHtml}</div>`;
  const det = li.querySelector("details");
  if (item.path) {
    const body = li.querySelector(".body-content");
    det.addEventListener("toggle", async () => {
      if (!det.open || det.dataset.loaded) return;
      det.dataset.loaded = "1";
      try {
        const r = await fetch(`/api/outputs/file?path=${encodeURIComponent(item.path)}`);
        if (r.ok) { body.textContent = await r.text(); body.classList.remove("muted"); }
        else { body.textContent = "(Note-Text konnte nicht geladen werden.)"; }
      } catch { body.textContent = "(Note-Text konnte nicht geladen werden.)"; }
    });
  } else {
    det.remove();
  }
  $("results-list").appendChild(li);
}

async function loadOutputs() {
  try {
    const r = await fetch("/api/outputs");
    if (!r.ok) return;
    const { items, dry_run } = await r.json();
    $("results-h").textContent = dry_run ? "Vorschau-Ergebnisse (nichts im Vault geschrieben)" : "Ergebnisse";
    $("results-section").hidden = false;
    $("results-list").innerHTML = "";
    const hasItems = items.length > 0;
    $("results-empty").style.display = hasItems ? "none" : "block";
    $("download-all").hidden = !hasItems;
    for (const item of items) addResultCard(item);
  } catch { }
}

// --- Verlauf (P4: GET /api/runs, read-only Ergebnis-Sektion) --------------

function formatHistoryDate(ts) {
  if (!ts) return "?";
  return new Date(ts * 1000).toLocaleString("de-DE");
}

function baseName(p) {
  return (p || "").split(/[\\/]/).pop() || "?";
}

function historyOptionsShort(options) {
  if (!options || Object.keys(options).length === 0) return "Server-Default";
  const parts = [];
  if (options.backend) parts.push(`Backend: ${options.backend}`);
  if (options.profile) parts.push(`Profil: ${options.profile}`);
  if (options.no_llm) parts.push("ohne LLM");
  return parts.length ? parts.join(" · ") : "Server-Default";
}

function renderHistoryResults(record) {
  // Read-only: laedt die bestehende Ergebnis-Sektion aus einem historischen
  // Record statt aus der aktuellen RunSession — reine Anzeige, kein Re-Run.
  $("results-h").textContent = `Verlauf: ${escapeHtml(baseName(record.source_pdf))} · ${formatHistoryDate(record.finished_at)}`;
  $("results-section").hidden = false;
  $("results-list").innerHTML = "";
  // ZIP-Endpunkt bezieht sich auf die aktuelle RunSession, nicht auf diesen
  // historischen Lauf — hier verstecken statt falscher Inhalte anzubieten.
  $("download-all").hidden = true;
  const notes = record.notes || [];
  const hasItems = notes.length > 0;
  $("results-empty").style.display = hasItems ? "none" : "block";
  $("results-empty").textContent = hasItems ? "" : "Keine Notes in diesem Lauf.";
  for (const item of notes) addResultCard(item);
  $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

function addHistoryEntry(record) {
  const li = document.createElement("li");
  li.className = "preview-card history-card";
  li.tabIndex = 0;
  li.setAttribute("role", "button");
  const rcKnown = record.rc !== null && record.rc !== undefined;
  const rcOk = record.rc === 0;
  const rcClass = rcKnown ? (rcOk ? "ok" : "warn") : "warn";
  const rcLabel = !rcKnown ? "abgebrochen/Fehler" : rcOk ? "erfolgreich" : `Fehlercode ${record.rc}`;
  const notesCount = (record.notes || []).length;
  li.innerHTML = `
    <div class="title">${escapeHtml(baseName(record.source_pdf))}</div>
    <div class="meta">
      <span class="hint">${formatHistoryDate(record.finished_at)}</span>
      <span class="badge">${record.dry_run ? "Vorschau" : "Geschrieben"}</span>
      <span class="badge ${rcClass}">${rcLabel}</span>
      <span class="badge">${notesCount} Notes</span>
    </div>
    <div class="hint">${escapeHtml(historyOptionsShort(record.options))}</div>`;
  const open = () => renderHistoryResults(record);
  li.addEventListener("click", open);
  li.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
  });
  $("history-list").appendChild(li);
}

async function loadHistory() {
  try {
    const r = await fetch("/api/runs");
    if (!r.ok) return;
    const { runs } = await r.json();
    $("history-list").innerHTML = "";
    $("history-empty").style.display = runs.length ? "none" : "block";
    for (const record of runs) addHistoryEntry(record);
  } catch { }
}

function logLine(text) {
  const el = $("log");
  el.textContent += text + "\n";
  el.scrollTop = el.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function startStream() {
  const es = new EventSource("/api/stream");
  startElapsed();
  // [0/7] = optionales Enrichment (Vor-Stufe) → keinen Step markieren.
  const onStage = (e) => { const d = JSON.parse(e.data); if (d.num >= 1) setStage(d.num); };
  es.addEventListener("started", (e) => logLine("» Lauf gestartet"));
  es.addEventListener("stage", onStage);
  es.addEventListener("note_progress", (e) => {
    const d = JSON.parse(e.data);
    $("note-progress").textContent = `Note ${d.index}/${d.total}: ${d.title}`;
  });
  es.addEventListener("preview", (e) => addPreview(JSON.parse(e.data)));
  es.addEventListener("log", (e) => { try { logLine(JSON.parse(e.data).text); } catch { } });
  es.addEventListener("error_hint", (e) => { try { showBanner(JSON.parse(e.data).text); } catch { } });
  // `done` = Pipeline hat geschrieben; der Lauf macht ggf. noch Stage-8-Eval.
  // NICHT schließen — erst `exited` beendet den Stream.
  es.addEventListener("done", (e) => {
    try { const d = JSON.parse(e.data); logLine(`✓ Pipeline fertig: ${d.written} Notes ${d.dry_run ? "(Dry-Run)" : "geschrieben"}`); } catch { }
  });
  const close = () => {
    es.close(); stopElapsed(); running = false; userCancelled = false;
    $("start-btn").disabled = false; $("stop-btn").hidden = true; applyStartGate();
  };
  es.addEventListener("exited", (e) => {
    let rc = 0;
    try { rc = JSON.parse(e.data).returncode; } catch { }
    if (userCancelled) { markStageError(activeStage); logLine("■ Lauf abgebrochen."); }
    else if (rc === 0) { setStage(99); logLine("● Lauf beendet."); loadOutputs(); }
    else { markStageError(activeStage); logLine(`✗ Lauf mit Fehlercode ${rc} beendet.`); }
    loadHistory();
    close();
  });
  es.addEventListener("error", (e) => {
    // Eigenes error-Event (RunSession-Exception) ODER EventSource-Verbindungsende.
    if (e.data) { markStageError(activeStage); logLine("✗ Fehler im Lauf."); }
    close();
  });
}

async function loadPdfs() {
  const r = await fetch("/api/pdfs");
  const { pdfs } = await r.json();
  const sel = $("pdf");
  sel.innerHTML = "";
  if (!pdfs.length) {
    sel.innerHTML = `<option value="">— keine PDFs gefunden —</option>`;
    return;
  }
  for (const p of pdfs) {
    const o = document.createElement("option");
    o.value = p.path; o.textContent = p.name;
    sel.appendChild(o);
  }
}

function selectUploadedPdf(path, name) {
  const sel = $("pdf");
  // bestehende Upload-Option (falls vorhanden) entfernen, neue oben einfügen
  [...sel.options].filter((o) => o.dataset.uploaded).forEach((o) => o.remove());
  const o = document.createElement("option");
  o.value = path; o.textContent = `${name} (hochgeladen)`; o.dataset.uploaded = "1";
  sel.insertBefore(o, sel.firstChild);
  sel.value = path;
}

async function uploadFile(file) {
  const status = $("upload-status");
  if (!file) return;
  if (!/\.pdf$/i.test(file.name)) { status.textContent = "Nur PDF-Dateien."; return; }
  status.textContent = `Lade „${file.name}“ hoch…`;
  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) { status.textContent = "✗ " + (d.error || "Upload fehlgeschlagen"); return; }
    selectUploadedPdf(d.path, d.name);
    status.textContent = `✓ „${d.name}“ bereit — Lauf starten.`;
  } catch {
    status.textContent = "✗ Upload fehlgeschlagen.";
  }
}

function wireUpload() {
  const dz = $("dropzone");
  const input = $("file-input");
  $("upload-btn").addEventListener("click", () => input.click());
  dz.addEventListener("click", (e) => { if (e.target === dz) input.click(); });
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => uploadFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) uploadFile(f);
  });
}

let doctorOk = true;

function applyBackendGate(d) {
  // Doctor-Gating (P1): litellm-Auswahl serverseitig entschieden (litellm_available
  // aus /api/doctor) — JS sperrt nur die Option und zeigt die Begründung an.
  const sel = $("opt-backend");
  if (!sel) return;
  const defaultOpt = sel.querySelector('option[value=""]');
  if (defaultOpt) defaultOpt.textContent = `Server-Default (${d.backend})`;
  const litellmOpt = sel.querySelector('option[value="litellm"]');
  const hintEl = $("opt-backend-hint");
  const available = !!d.litellm_available;
  if (litellmOpt) {
    litellmOpt.disabled = !available;
    if (sel.value === "litellm" && !available) sel.value = "";
  }
  if (hintEl) {
    hintEl.textContent = available ? "" : (d.litellm_hint || "litellm nicht verfügbar — kein Provider-Key gefunden.");
  }
}

async function loadDoctor() {
  const el = $("doctor");
  try {
    const d = await (await fetch("/api/doctor")).json();
    doctorOk = d.ok;
    const fails = (d.checks || []).filter((c) => !c.ok);
    const summary = `Backend: ${d.backend} · Vault: ${d.vault}`;
    if (d.ok) {
      el.textContent = `${summary} ✓`;
      el.classList.remove("bad");
    } else {
      const probleme = fails.map((c) => `${c.name}${c.required ? "" : " (optional)"}`).join(", ");
      el.innerHTML = `${escapeHtml(summary)} <strong>✗ Preflight: ${escapeHtml(probleme)}</strong>` +
        fails.filter((c) => c.hint).map((c) => `<br><span class="muted">→ ${escapeHtml(c.hint)}</span>`).join("");
      el.classList.toggle("bad", fails.some((c) => c.required));
    }
    applyBackendGate(d);
  } catch {
    el.textContent = "Preflight konnte nicht geladen werden.";
  }
  applyStartGate();
}

function applyStartGate() {
  // Start sperren, wenn ein required-Preflight-Check rot ist (Fehler vermeiden
  // statt mitten im Lauf scheitern).
  const btn = $("start-btn");
  if (!doctorOk) {
    btn.disabled = true;
    btn.title = "Preflight fehlgeschlagen — siehe Statuszeile oben.";
  } else if (btn.title) {
    btn.disabled = false;
    btn.title = "";
  }
}

function updateModeHint() {
  const dry = $("dry-run").checked;
  const btn = $("start-btn");
  // Den einzigen irreversiblen Schritt visuell klar absetzen (Label + Warnfarbe),
  // statt nur als unscheinbare Checkbox (kein Modal — Vorschau ist der Gate).
  btn.textContent = dry ? "Vorschau starten" : "In Vault schreiben";
  btn.classList.toggle("danger", !dry);
  $("mode-hint").textContent = dry
    ? "Vorschau: erzeugt Notes, schreibt nichts in den Vault (nur lokale .cache-Kopien). Durchläuft die volle Pipeline inkl. Qualitäts-Eval — verursacht also LLM-Aufrufe/Kosten. Ergebnis erscheint unter „Erzeugte Notes“."
    : "Schreibt Notes nach 00-inbox im Vault. Frischer Lauf — Scores können von einer vorherigen Vorschau leicht abweichen.";
}

async function attachIfRunActive() {
  // Lädt die Seite, während (woanders) bereits ein Lauf aktiv ist: anhängen
  // statt in die 409-Sackgasse zu laufen — Stop-Button + Stream-Reattach.
  try {
    const s = await (await fetch("/api/status")).json();
    if (s.active) {
      resetRun();
      running = true; userCancelled = false;
      $("stop-btn").hidden = false; $("start-btn").disabled = true;
      currentPdfStem = (s.pdf || "").split(/[\\/]/).pop().replace(/\.pdf$/i, "");
      logLine("» Laufender Pipeline-Lauf erkannt — angehängt.");
      startStream();
    }
  } catch { }
}

document.addEventListener("DOMContentLoaded", () => {
  renderStepper();
  loadPdfs();
  loadDoctor();
  updateModeHint();
  wireUpload();
  attachIfRunActive();
  loadHistory();
  $("dry-run").addEventListener("change", updateModeHint);

  $("run-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!doctorOk) { logLine("✗ Preflight fehlgeschlagen — bitte oben beheben."); return; }
    const pdf = $("pdf").value;
    if (!pdf) return;
    resetRun();
    $("start-btn").disabled = true;
    const options = buildRunOptions();
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf, dry_run: $("dry-run").checked, options }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      logLine("✗ " + (err.error || `Start fehlgeschlagen (${r.status})`));
      $("start-btn").disabled = false;
      applyStartGate();
      return;
    }
    const started = await r.json().catch(() => ({}));
    currentPdfStem = pdf.split(/[\\/]/).pop().replace(/\.pdf$/i, "");
    renderRunHeader(started.options || options);
    running = true; userCancelled = false; $("stop-btn").hidden = false;
    startStream();
  });

  $("stop-btn").addEventListener("click", async () => {
    userCancelled = true;
    $("stop-btn").disabled = true;
    logLine("■ Abbruch angefordert…");
    try { await fetch("/api/cancel", { method: "POST" }); } catch { }
    $("stop-btn").disabled = false;
  });

  // Tab/Fenster wird während eines Laufs geschlossen → Subprocess nicht verwaisen lassen.
  window.addEventListener("pagehide", () => {
    if (running && navigator.sendBeacon) navigator.sendBeacon("/api/cancel");
  });
});
