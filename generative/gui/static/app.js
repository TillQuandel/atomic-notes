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

function renderStepper() {
  const ol = $("stepper");
  ol.innerHTML = "";
  for (const [num, label] of STAGES) {
    const li = document.createElement("li");
    li.id = `step-${num}`;
    li.innerHTML = `<span class="dot" aria-hidden="true"></span><span>${num}. ${label}</span>`;
    ol.appendChild(li);
  }
}

function setStage(num) {
  for (const [n] of STAGES) {
    const li = $(`step-${n}`);
    if (!li) continue;
    li.classList.toggle("done", n < num);
    li.classList.toggle("active", n === num);
  }
}

function resetRun() {
  renderStepper();
  $("preview-list").innerHTML = "";
  $("preview-empty").style.display = "block";
  $("note-progress").textContent = "";
  $("log").textContent = "";
  const b = $("error-banner"); b.hidden = true; b.textContent = "";
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
  const onStage = (e) => { const d = JSON.parse(e.data); setStage(d.num); };
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
    es.close(); running = false; userCancelled = false;
    $("start-btn").disabled = false; $("stop-btn").hidden = true; applyStartGate();
  };
  es.addEventListener("exited", (e) => {
    let rc = 0;
    try { rc = JSON.parse(e.data).returncode; } catch { }
    if (userCancelled) { logLine("■ Lauf abgebrochen."); }
    else if (rc === 0) { setStage(99); logLine("● Lauf beendet."); }
    else { logLine(`✗ Lauf mit Fehlercode ${rc} beendet.`); }
    close();
  });
  es.addEventListener("error", (e) => {
    // Eigenes error-Event (RunSession-Exception) ODER EventSource-Verbindungsende.
    if (e.data) { logLine("✗ Fehler im Lauf."); }
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
  $("mode-hint").textContent = dry
    ? "Vorschau: erzeugt Notes, schreibt nichts. Ergebnis erscheint unter „Erzeugte Notes“."
    : "Schreibt in den Vault (00-inbox). Frischer Lauf — Scores können von einer vorherigen Vorschau leicht abweichen.";
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
  $("dry-run").addEventListener("change", updateModeHint);

  $("run-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!doctorOk) { logLine("✗ Preflight fehlgeschlagen — bitte oben beheben."); return; }
    const pdf = $("pdf").value;
    if (!pdf) return;
    resetRun();
    $("start-btn").disabled = true;
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf, dry_run: $("dry-run").checked }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      logLine("✗ " + (err.error || `Start fehlgeschlagen (${r.status})`));
      $("start-btn").disabled = false;
      applyStartGate();
      return;
    }
    currentPdfStem = pdf.split(/[\\/]/).pop().replace(/\.pdf$/i, "");
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
