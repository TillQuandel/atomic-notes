// atomic-notes GUI — Frontend-Logik (vanilla JS, SSE via EventSource).
// Bewusst ohne Framework/CDN: laeuft komplett offline.

const STAGES = [
  [1, "PDF & Chunking"], [2, "Vault-Kontext"], [3, "Quellen-Qualität"],
  [4, "Planner"], [5, "Extractor"], [6, "Verifier & Critic"],
  [7, "Vault-Writer"], [8, "Qualitäts-Eval"],
];

const $ = (id) => document.getElementById(id);
let currentPdfStem = "";

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
}

function addPreview(ev) {
  $("preview-empty").style.display = "none";
  const li = document.createElement("li");
  li.className = "preview-card";
  const routingLabel = { vault: "Vault-Empfehlung", inbox: "Inbox-Review", merge: "Merge-Stub" }[ev.routing] || ev.routing;
  const flags = (ev.flags || []).map((f) => `<span class="badge flag">${escapeHtml(f)}</span>`).join("");
  li.innerHTML = `
    <div class="title">${escapeHtml(ev.name)}</div>
    <div class="meta">
      <span class="badge ${ev.routing}">${routingLabel}</span>
      <span class="badge">Score ${ev.score ?? "?"}/5</span>
      <span class="badge">Hard-Gates ${ev.hard_gates ? "pass" : "fail"}</span>
      <span class="badge">Confidence ${escapeHtml(ev.confidence || "?")}</span>
      ${ev.reason ? `<span class="badge">${escapeHtml(ev.reason)}</span>` : ""}
      ${ev.merge_target ? `<span class="badge">→ ${escapeHtml(ev.merge_target)}</span>` : ""}
      ${flags}
    </div>`;
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
  const finish = (e) => {
    try { const d = JSON.parse(e.data); logLine(`✓ Fertig: ${d.written} Notes ${d.dry_run ? "(Dry-Run)" : "geschrieben"}`); } catch { }
    setStage(99); es.close(); $("start-btn").disabled = false;
  };
  es.addEventListener("done", finish);
  es.addEventListener("error", (e) => {
    logLine("✗ Fehler im Lauf."); es.close(); $("start-btn").disabled = false;
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

async function loadDoctor() {
  const r = await fetch("/api/doctor");
  const d = await r.json();
  const el = $("doctor");
  el.textContent = `Backend: ${d.backend} · Vault: ${d.vault} ${d.vault_exists ? "✓" : "✗ (nicht gefunden)"}`;
  el.classList.toggle("bad", !d.vault_exists);
}

function updateModeHint() {
  const dry = $("dry-run").checked;
  $("mode-hint").textContent = dry
    ? "Vorschau: erzeugt Notes, schreibt nichts. Ergebnis erscheint unter „Erzeugte Notes“."
    : "Schreibt in den Vault (00-inbox). Frischer Lauf — Scores können von einer vorherigen Vorschau leicht abweichen.";
}

document.addEventListener("DOMContentLoaded", () => {
  renderStepper();
  loadPdfs();
  loadDoctor();
  updateModeHint();
  $("dry-run").addEventListener("change", updateModeHint);

  $("run-form").addEventListener("submit", async (e) => {
    e.preventDefault();
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
      return;
    }
    currentPdfStem = pdf.split(/[\\/]/).pop().replace(/\.pdf$/i, "");
    startStream();
  });
});
