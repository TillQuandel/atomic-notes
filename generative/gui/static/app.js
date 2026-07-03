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
let stageDurations = {}; // Stage-Nummer -> abgeschlossene Dauer in s (P5)
let runSummary = null; // vom `run_summary`-Event (P5: Zeit/Tokens Final-Report)

// --- reine Formatierungs-Helfer (P5, ohne DOM-Zugriff — per `node --check`
// syntaktisch pruefbar; Logik manuell gegen die Plan-Beispiele verifiziert). ---

function formatSeconds(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, "0")} min`;
}

function formatTokenCount(n) {
  return Number(n).toLocaleString("de-DE");
}

function buildRunSummaryText(stageCount, summary) {
  const parts = [];
  if (stageCount) parts.push(`${stageCount} Stages`);
  if (summary && summary.duration_s !== undefined) parts.push(formatSeconds(summary.duration_s));
  if (summary && summary.tokens && summary.tokens.total !== undefined) {
    parts.push(`${formatTokenCount(summary.tokens.total)} Tokens`);
  }
  return parts.join(" · ");
}

function renderStepper() {
  const ol = $("stepper");
  ol.innerHTML = "";
  for (const [num, label] of STAGES) {
    const li = document.createElement("li");
    li.id = `step-${num}`;
    li.innerHTML = `<span class="dot" aria-hidden="true"></span><span class="step-label">${num}. ${label}</span><span class="state"></span><span class="elapsed" aria-hidden="true"></span>`;
    ol.appendChild(li);
  }
}

function setStage(num) {
  if (num !== activeStage) {
    // Abgeschlossene Stufe: Dauer festhalten, damit sie als Zeit stehen bleibt.
    if (activeStage > 0 && stageStartedAt) {
      stageDurations[activeStage] = (Date.now() - stageStartedAt) / 1000;
    }
    activeStage = num;
    stageStartedAt = Date.now();
  }
  for (const [n] of STAGES) {
    const li = $(`step-${n}`);
    if (!li) continue;
    li.classList.toggle("done", n < num);
    li.classList.toggle("active", n === num);
    li.classList.remove("error");
    const s = li.querySelector(".state");
    if (s) s.textContent = n < num ? "fertig" : n === num ? "läuft…" : "";
    const e = li.querySelector(".elapsed");
    if (!e) continue;
    if (n === num) { e.textContent = ""; continue; } // laufende Stufe: tickElapsed() uebernimmt
    e.textContent = stageDurations[n] !== undefined ? formatSeconds(stageDurations[n]) : "";
  }
}

function markStageError(num) {
  const li = $(`step-${num}`);
  if (li) {
    li.classList.remove("active"); li.classList.add("error");
    const s = li.querySelector(".state");
    if (s) s.textContent = "Fehler";
  }
}

function renderRunSummaryLine() {
  const el = $("run-summary");
  if (!el) return;
  el.textContent = buildRunSummaryText(Object.keys(stageDurations).length, runSummary);
}

function tickElapsed() {
  // NN/g: Indeterminate-Wartezeiten ab >10s brauchen einen sichtbaren Zaehler
  // (darunter reicht der Spinner/Status — kein Text-Rauschen fuer kurze Stufen).
  const li = $(`step-${activeStage}`);
  if (!li || !running) return;
  const s = Math.floor((Date.now() - stageStartedAt) / 1000);
  const e = li.querySelector(".elapsed");
  if (e) e.textContent = s > 10 ? formatSeconds(s) : "";
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
  stageDurations = {};
  runSummary = null;
  $("run-summary").textContent = "";
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
  // B3: Export-Ordner statt Vault-Inbox — nur relevant im Schreib-Modus, der
  // Server ignoriert inbox_dir im Dry-Run ohnehin (kein Fehler, s. build_run_spec).
  if ($("target-export-dir").checked) {
    const path = $("target-export-path").value.trim();
    if (path) options.inbox_dir = path;
  }
  return options;
}

// --- Ziel-Ordner (B3): Vault-Inbox (Default) vs. freier Export-Ordner -----

function applyTargetGate() {
  const usingExportDir = $("target-export-dir").checked;
  $("target-export-path-wrap").hidden = !usingExportDir;
  const path = $("target-export-path").value.trim();
  const invalid = usingExportDir && !path;
  $("target-export-path").classList.toggle("bad", invalid);
  return invalid;
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
    p.textContent = text;
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
    ${flags ? `<div class="flags">Hinweis: ${escapeHtml(flags)}</div>` : ""}
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
    ${flags ? `<div class="flags">Hinweis: ${escapeHtml(flags)}</div>` : ""}
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
  // textContent escaped selbst — kein escapeHtml noetig (sonst erscheinen Entities literal).
  $("results-h").textContent = `Verlauf: ${baseName(record.source_pdf)} · ${formatHistoryDate(record.finished_at)}`;
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
  const rcKnown = record.rc !== null && record.rc !== undefined;
  const rcOk = record.rc === 0;
  const rcClass = rcKnown ? (rcOk ? "ok" : "warn") : "warn";
  const rcLabel = !rcKnown ? "abgebrochen/Fehler" : rcOk ? "erfolgreich" : `Fehlercode ${record.rc}`;
  const notesCount = (record.notes || []).length;
  // P5: Zeit/Tokens nur zeigen, wenn der Lauf ein run_summary-Event hatte
  // (kein Erfinden bei aelteren/gecrashten Records ohne diese Felder, L5).
  const summaryBadges = [];
  if (record.duration_s !== undefined) summaryBadges.push(`<span class="badge">${formatSeconds(record.duration_s)}</span>`);
  if (record.tokens && record.tokens.total !== undefined) {
    summaryBadges.push(`<span class="badge">${formatTokenCount(record.tokens.total)} Tokens</span>`);
  }
  // P8: Karteninhalt in einem <button> statt li[role=button] — native
  // Fokus-/Aktivierungs-Semantik, kein eigener keydown-Handler noetig.
  li.innerHTML = `
    <button type="button" class="history-open">
      <div class="title">${escapeHtml(baseName(record.source_pdf))}</div>
      <div class="meta">
        <span class="hint">${formatHistoryDate(record.finished_at)}</span>
        <span class="badge">${record.dry_run ? "Vorschau" : "Geschrieben"}</span>
        <span class="badge ${rcClass}">${rcLabel}</span>
        <span class="badge">${notesCount} Notes</span>
        ${summaryBadges.join("")}
      </div>
      <div class="hint">${escapeHtml(historyOptionsShort(record.options))}</div>
    </button>`;
  li.querySelector(".history-open").addEventListener("click", () => renderHistoryResults(record));
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
  // P8: Fokus auf die Pipeline-Sektion, sobald der Lauf sichtbar startet
  // (SSE-Reattach spielt "started" erneut ab -- Fokus-Sprung dann harmlos).
  es.addEventListener("started", (e) => { logLine("» Lauf gestartet"); $("steps-h").focus(); });
  es.addEventListener("stage", onStage);
  es.addEventListener("note_progress", (e) => {
    const d = JSON.parse(e.data);
    $("note-progress").textContent = `Note ${d.index}/${d.total}: ${d.title}`;
  });
  es.addEventListener("preview", (e) => addPreview(JSON.parse(e.data)));
  es.addEventListener("log", (e) => { try { logLine(JSON.parse(e.data).text); } catch { } });
  es.addEventListener("error_hint", (e) => { try { showBanner(JSON.parse(e.data).text); } catch { } });
  // P5: Final-Report Zeit/Tokens (run_parser.py) — Summenzeile unter dem Stepper.
  es.addEventListener("run_summary", (e) => {
    try { runSummary = JSON.parse(e.data); renderRunSummaryLine(); } catch { }
  });
  // `done` = Pipeline hat geschrieben; der Lauf macht ggf. noch Stage-8-Eval.
  // NICHT schließen — erst `exited` beendet den Stream.
  es.addEventListener("done", (e) => {
    try { const d = JSON.parse(e.data); logLine(`✓ Pipeline fertig: ${d.written} Notes ${d.dry_run ? "(Dry-Run)" : "geschrieben"}`); } catch { }
  });
  const close = () => {
    es.close(); stopElapsed(); running = false; userCancelled = false;
    $("start-btn").disabled = false; $("stop-btn").hidden = true; applyStartGate();
    applyVaultGate();
  };
  es.addEventListener("exited", (e) => {
    let rc = 0;
    try { rc = JSON.parse(e.data).returncode; } catch { }
    if (userCancelled) { markStageError(activeStage); logLine("■ Lauf abgebrochen."); }
    else if (rc === 0) {
      setStage(99); renderRunSummaryLine(); logLine("● Lauf beendet.");
      // P8: Fokus-Sprung erst NACH dem Laden/Sichtbarwerden der Ergebnis-
      // Sektion, und nur bei erfolgreichem Lauf (Fehler kommunizieren Banner/Stepper).
      loadOutputs().then(() => $("results-h").focus());
    }
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
    if (!r.ok) { status.textContent = d.error || "Upload fehlgeschlagen"; return; }
    selectUploadedPdf(d.path, d.name);
    status.textContent = `„${d.name}“ bereit — Lauf starten.`;
  } catch {
    status.textContent = "Upload fehlgeschlagen.";
  }
}

function wireUpload() {
  const dz = $("dropzone");
  const input = $("file-input");
  $("upload-btn").addEventListener("click", () => input.click());
  // Tastatur-Zugang laeuft ueber den nativ fokussierbaren "Datei auswaehlen…"-
  // Button (P8: Dropzone selbst nicht mehr interaktiv, kein eigener keydown).
  dz.addEventListener("click", (e) => { if (e.target === dz) input.click(); });
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
let litellmAvailable = true; // P1-Doctor-Gating, von loadDoctor() gesetzt

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

// B1a: "Zugang"-Panel -- zeigt nur Namen/Booleans aus `access`, nie Key-Werte.
function renderAccess(access) {
  if (!access) return;
  $("access-backend").textContent = `Aktives Backend: ${access.backend}`;

  const sub = access.subscription || {};
  const subOk = !!sub.cli_found && !!sub.credentials_present;
  const subDot = document.querySelector("#access-subscription .access-dot");
  const subStatus = document.querySelector("#access-subscription .access-status");
  subDot.classList.toggle("ok", subOk);
  subDot.classList.toggle("bad", !subOk);
  subStatus.textContent = subOk
    ? "verfügbar"
    : `nicht verfügbar (CLI: ${sub.cli_found ? "gefunden" : "fehlt"}, Login: ${sub.credentials_present ? "vorhanden" : "fehlt"})`;

  const lit = access.litellm || {};
  const litOk = !!lit.available;
  const litDot = document.querySelector("#access-litellm .access-dot");
  const litStatus = document.querySelector("#access-litellm .access-status");
  litDot.classList.toggle("ok", litOk);
  litDot.classList.toggle("bad", !litOk);
  litStatus.textContent = litOk
    ? `verfügbar (${(lit.key_vars_set || []).join(", ")})`
    : "nicht verfügbar (kein Provider-Key gesetzt)";

  // L6: ehrlicher Hinweis statt GUI-Login -- der existiert headless nicht.
  $("access-subscription-hint").hidden = subOk;
}

async function loadDoctor() {
  const el = $("doctor");
  try {
    const d = await (await fetch("/api/doctor")).json();
    doctorOk = d.ok;
    litellmAvailable = !!d.litellm_available;
    const fails = (d.checks || []).filter((c) => !c.ok);
    const summary = `Backend: ${d.backend} · Vault: ${d.vault}`;
    if (d.ok) {
      el.textContent = `${summary} — ok`;
      el.classList.remove("bad");
    } else {
      const probleme = fails.map((c) => `${c.name}${c.required ? "" : " (optional)"}`).join(", ");
      el.innerHTML = `${escapeHtml(summary)} <strong>Preflight: ${escapeHtml(probleme)}</strong>` +
        fails.filter((c) => c.hint).map((c) => `<br><span class="muted">→ ${escapeHtml(c.hint)}</span>`).join("");
      el.classList.toggle("bad", fails.some((c) => c.required));
    }
    applyBackendGate(d);
    renderAccess(d.access);
  } catch {
    el.textContent = "Preflight konnte nicht geladen werden.";
  }
  applyStartGate();
}

// --- Persistierte Einstellungen (P2: GET/PUT /api/settings) ---------------
// Stiller Auto-Save bei Aenderung eines der vier Felder, kein "Speichern"-
// Button (passend zum Bestand). Laden erst NACH loadDoctor(), damit das
// Doctor-Gating (litellm-Verfuegbarkeit) beim Vorbelegen bereits feststeht.

function currentSettingsPayload() {
  const payload = {
    backend: $("opt-backend").value,
    profile: $("opt-profile").value,
    no_llm: $("opt-no-llm").checked,
    dry_run: $("dry-run").checked,
  };
  // B2: PUT /api/settings ersetzt die Datei vollstaendig (kein Merge) -- ohne
  // den zuletzt geladenen/uebernommenen Vault hier mitzuschicken wuerde ein
  // spaeterer Autosave (Backend/Profil/…) den per PUT /api/vault gesetzten
  // vault_path wieder loeschen.
  if (currentVaultPath) payload.vault_path = currentVaultPath;
  return payload;
}

async function loadSettings() {
  try {
    const r = await fetch("/api/settings");
    if (!r.ok) return {};
    const s = await r.json();
    if (s.warning) showBanner(s.warning);
    return s;
  } catch {
    return {};
  }
}

function applyStoredSettings(settings) {
  if (!settings) return;
  if (settings.backend) {
    if (settings.backend === "litellm" && !litellmAvailable) {
      showBanner('Gespeichertes Backend „litellm" ist nicht verfügbar (kein Provider-Key) — Server-Default wird verwendet.');
    } else {
      $("opt-backend").value = settings.backend;
    }
  }
  if (settings.profile) $("opt-profile").value = settings.profile;
  if (settings.no_llm !== undefined) $("opt-no-llm").checked = !!settings.no_llm;
  if (settings.dry_run !== undefined) $("dry-run").checked = !!settings.dry_run;
  updateModeHint();
}

async function saveSettings() {
  try {
    await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentSettingsPayload()),
    });
  } catch { }
}

function applyStartGate() {
  // Start sperren, wenn ein required-Preflight-Check rot ist (Fehler vermeiden
  // statt mitten im Lauf scheitern) ODER (B3) „Export-Ordner" gewaehlt, aber
  // kein Pfad eingetragen ist (vorab gesperrt statt nachtraeglicher 400, L5).
  const btn = $("start-btn");
  const targetInvalid = applyTargetGate();
  if (!doctorOk) {
    btn.disabled = true;
    btn.title = "Preflight fehlgeschlagen — siehe Statuszeile oben.";
  } else if (targetInvalid) {
    btn.disabled = true;
    btn.title = "Export-Ordner-Pfad fehlt.";
  } else {
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

// --- Ziel-Vault (B2: PUT/GET /api/vault) ----------------------------------

let currentVaultPath = null; // zuletzt geladener/uebernommener Vault, s. currentSettingsPayload()

async function loadVault() {
  try {
    const r = await fetch("/api/vault");
    if (!r.ok) return;
    const { vault } = await r.json();
    currentVaultPath = vault;
    $("vault-path").value = vault;
    $("vault-current").textContent = `Aktueller Vault: ${vault}`;
  } catch { }
}

function applyVaultGate() {
  // Wechsel waehrend eines aktiven Laufs ist serverseitig gesperrt (409, R1) --
  // das Feld hier zusaetzlich deaktivieren, statt den Nutzer erst den
  // Fehlschlag erleben zu lassen.
  const btn = $("vault-apply-btn");
  const input = $("vault-path");
  if (btn) btn.disabled = running;
  if (input) input.disabled = running;
}

function setVaultStatus(text, isError) {
  const el = $("vault-status");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("bad", !!isError);
  el.classList.toggle("ok", text !== "" && !isError);
  $("vault-path").classList.toggle("bad", !!isError);
}

async function applyVault(path) {
  setVaultStatus("");
  try {
    const r = await fetch("/api/vault", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = d.error || `Vault-Wechsel fehlgeschlagen (${r.status})`;
      showBanner(msg);
      setVaultStatus(msg, true);
      return;
    }
    currentVaultPath = d.vault;
    $("vault-path").value = d.vault;
    $("vault-current").textContent = `Aktueller Vault: ${d.vault}`;
    setVaultStatus("Übernommen.", false);
    await loadDoctor(); // Preflight (vault_exists) haengt vom neuen Vault ab.
  } catch {
    setVaultStatus("Vault-Wechsel fehlgeschlagen (Netzwerkfehler).", true);
  }
}

function wireVaultForm() {
  $("vault-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const path = $("vault-path").value.trim();
    if (!path) return;
    applyVault(path);
  });
}

// --- litellm-API-Key setzen (B1b: write-only, nie zurueckgegeben/angezeigt) -

function setLitellmKeyStatus(text, isError) {
  const el = $("litellm-key-status");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("bad", !!isError);
  el.classList.toggle("ok", text !== "" && !isError);
}

async function saveLitellmKey(provider, key) {
  setLitellmKeyStatus("");
  try {
    const r = await fetch("/api/access/litellm-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, key }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = d.error || `Speichern fehlgeschlagen (${r.status})`;
      showBanner(msg);
      setLitellmKeyStatus(msg, true);
      return;
    }
    // Key nie zurueckanzeigen -- nur Erfolgsmeldung, Feld leeren.
    $("litellm-key-value").value = "";
    // L6-Ehrlichkeit: der laufende GUI-Prozess liest `.env`/os.environ NICHT
    // neu (config.py laedt sie nur beim Start, override=False) -- ein frisch
    // gesetzter Key wird erst nach GUI-Neustart aktiv. KEIN loadDoctor()-Aufruf
    // hier: er wuerde weiter "litellm nicht verfuegbar" zeigen und so faelsch-
    // lich Unwirksamkeit suggerieren. Die Meldung benennt die Neustart-Grenze.
    setLitellmKeyStatus("Key gespeichert. GUI neu starten, damit er aktiv wird.", false);
  } catch {
    setLitellmKeyStatus("Speichern fehlgeschlagen (Netzwerkfehler).", true);
  }
}

function wireLitellmKeyForm() {
  $("litellm-key-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const provider = $("litellm-key-provider").value;
    const key = $("litellm-key-value").value;
    if (!key.trim()) { setLitellmKeyStatus("Key darf nicht leer sein.", true); return; }
    saveLitellmKey(provider, key);
  });
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
      renderRunHeader(s.options || {});
      logLine("» Laufender Pipeline-Lauf erkannt — angehängt.");
      applyVaultGate();
      startStream();
    }
  } catch { }
}

document.addEventListener("DOMContentLoaded", async () => {
  renderStepper();
  loadPdfs();
  updateModeHint();
  wireUpload();
  wireVaultForm();
  wireLitellmKeyForm();
  $("access-copy-btn").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText("claude");
      const btn = $("access-copy-btn");
      const original = btn.textContent;
      btn.textContent = "Kopiert!";
      setTimeout(() => { btn.textContent = original; }, 1500);
    } catch { /* Clipboard-API evtl. nicht verfuegbar -- kein Fallback noetig, Text ist sichtbar. */ }
  });
  loadVault();
  attachIfRunActive();
  loadHistory();
  // Reihenfolge bewusst: erst Doctor (litellm-Verfuegbarkeit steht fest), dann
  // gespeicherte Einstellungen anwenden — sonst koennte ein nicht verfuegbares
  // "litellm" aus einer frueheren Sitzung stillschweigend uebernommen werden.
  await loadDoctor();
  applyStoredSettings(await loadSettings());
  $("dry-run").addEventListener("change", () => { updateModeHint(); saveSettings(); });
  $("opt-backend").addEventListener("change", saveSettings);
  $("opt-profile").addEventListener("change", saveSettings);
  $("opt-no-llm").addEventListener("change", saveSettings);
  // B3: Ziel-Ordner-Wahl nicht in P2-Settings persistiert (nur `inbox_dir` als
  // Lauf-Option, s. buildRunOptions) — nur der Sperr-Zustand aktualisiert sich.
  $("target-vault-inbox").addEventListener("change", applyStartGate);
  $("target-export-dir").addEventListener("change", applyStartGate);
  $("target-export-path").addEventListener("input", applyStartGate);

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
      const msg = err.error || `Start fehlgeschlagen (${r.status})`;
      logLine("✗ " + msg);
      // B3: serverseitig abgelehnter Export-Ordner (400) landet — wie jeder
      // andere Lauf-Start-Fehler — im bestehenden Fehler-Banner (L5).
      showBanner(msg);
      $("start-btn").disabled = false;
      applyStartGate();
      return;
    }
    const started = await r.json().catch(() => ({}));
    currentPdfStem = pdf.split(/[\\/]/).pop().replace(/\.pdf$/i, "");
    renderRunHeader(started.options || options);
    running = true; userCancelled = false; $("stop-btn").hidden = false;
    applyVaultGate();
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
