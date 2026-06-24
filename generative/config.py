import os
from pathlib import Path
from dotenv import load_dotenv

# Explizit generative/.env (neben dieser Datei) laden — CWD-unabhängig. Ein
# nacktes load_dotenv() sucht CWD-relativ und verfehlt generative/.env beim
# dokumentierten Start aus dem Repo-Root.
load_dotenv(Path(__file__).resolve().parent / ".env")

BACKEND = os.getenv("ATOMIC_AGENT_BACKEND", "subscription")

VAULT = Path(os.environ.get(
    "ATOMIC_AGENT_VAULT_PATH",
    str(Path.home() / "Obsidian_Vault")
))
INBOX = VAULT / "00-inbox"
WISSEN = VAULT / "04-wissen"
BA_DIR = VAULT / "01-studium" / "bachelorarbeit"
SCRIPTS_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPTS_DIR / ".cache"
# Eval-Historie (Stage-8). Hier statt in eval_quality_v4 definiert, damit reine
# Label-Tools (kappa.py, build_labels.py, collect.py) den Pfad importieren können,
# ohne eval_quality_v4s Import-Zeit-sys.exit() bei fehlendem fitz/rapidfuzz zu ziehen.
QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"
# Externe Quell-PDFs. Renderer baut hieraus den `file://`-Link
# in den Quellen-Callout, damit User die PDF aus der Note öffnen kann.
LITERATURE_DIR = Path(os.environ.get(
    "ATOMIC_AGENT_PDF_BASE",
    str(Path.home() / "Documents" / "Literatur")
))

_contact = os.environ.get("ATOMIC_AGENT_CONTACT", "atomic-agent-user")
USER_AGENT = f"AtomicAgent/1.0 (mailto:{_contact})"

# Phoenix-Observability: OTLP-Ziel + Server-venv für den Auto-Start.
# Nur relevant bei ATOMIC_AGENT_TRACING=phoenix. Beide per ENV überschreibbar,
# damit der Server-venv-Pfad nicht hartkodiert ist (fremde Installs ohne
# .venv-phoenix bleiben unberührt, weil Tracing per Default aus ist).
PHOENIX_PORT = int(os.environ.get("ATOMIC_AGENT_PHOENIX_PORT", "6006"))
PHOENIX_VENV = Path(os.environ.get(
    "ATOMIC_AGENT_PHOENIX_VENV",
    str(SCRIPTS_DIR.parent / ".venv-phoenix")
))

# Claude-Aufruf via CLI-Subprocess (Pro/Max-Subscription, OAuth)
CLAUDE_BIN = "claude"

# Modell-Routing: Sonnet als Default (A/B-Test 2026-05-20: niedrigere Halluzinationsrate als Opus,
# 3x günstiger per API, gleiche Note-Anzahl). Opus per ENV überschreibbar für Vergleichstests.
MODEL_OPUS  = os.getenv("ATOMIC_AGENT_MODEL_OPUS",  "anthropic/claude-sonnet-4-6")
MODEL_HAIKU = os.getenv("ATOMIC_AGENT_MODEL_HAIKU", "anthropic/claude-haiku-4-5-20251001")

# Pro-Agent Mapping (siehe Plan Milestone 2.1)
MODEL_PLANNER = MODEL_OPUS
MODEL_EXTRACTOR = MODEL_OPUS
MODEL_EXTENDER = MODEL_OPUS         # Backlog v1.1
MODEL_VERIFIER = MODEL_HAIKU
MODEL_CROSS_REF = MODEL_HAIKU
MODEL_CONFIDENCE = MODEL_HAIKU
MODEL_CRITIC = MODEL_HAIKU
MODEL_SUMMARY = MODEL_HAIKU         # Map-Reduce-Summary für lange PDFs (Backlog)

# Critic-Schwelle: Auto-Write nur bei Score >= Schwelle UND alle Hard-Gates pass
CRITIC_AUTO_THRESHOLD = 4   # von 5 Tests, siehe Schema-Konzept Milestone 1.1

# Cross-lingualer Rettungsanker für filter_hallucinated (planner.py): der lexikalische
# Token-Coverage-Filter ist sprachblind — ein deutscher (paraphrasierter) Konzept-Titel
# hat null wörtlichen Overlap mit einer englischen Quelle und würde fälschlich als
# „halluziniert" verworfen (Ebner-Run 2026-06-23: der Paper-Kernbefund „Lern-Zufriedenheits-
# Dissoziation" starb genau so). Geprüft wird dann die semantische Präsenz: MAX-Cosine des
# Titel-Embeddings gegen die Satz-Embeddings des Volltexts (multilinguales MiniLM, bereits
# geladen). Schwelle gemessen auf der Ebner-EN-Quelle (n=1, NICHT voll kalibriert): echte
# Konzepte 0.575–0.825, echte Halluzinationen 0.357/0.358 → 0.50 trennt mit großem Abstand.
# Reiner OR-RETTUNGSANKER: greift nur, wenn der lexikalische Filter ablehnt — kann ein
# Konzept also nur RETTEN, nie zusätzlich verwerfen. ENV-überschreibbar für Kalibrierung.
TITLE_PRESENCE_COSINE_THRESHOLD = float(os.getenv("ATOMIC_AGENT_TITLE_PRESENCE_COSINE", "0.50"))

# Typ-bewusstes Dedup-Blocking: Note-Typen, die per Vault-Design mit Konzept-Notes
# KOEXISTIEREN und daher nie Duplikat-Kandidaten sind. Eine `type: literature`-Note ist die
# Note ÜBER ein Paper, eine `type: atomic`-Note die Note ÜBER ein Konzept DARIN — beide
# existieren gleichzeitig (Schema-Lit vs. Schema-Konzept); `moc`/`merge-stub` sind Pointer-
# bzw. Zwischen-Notes. Ohne diesen Filter flaggt cross_reference eine Konzept-Note fälschlich
# als Duplikat ihrer eigenen Lit-Note (Ebner-Run 2026-06-23: „Webinar" → Dup von
# ba-lit-ebner-gegenfurtner-2019). related-LINKS über Typgrenzen bleiben erlaubt (eine
# Konzept-Note SOLL auf ihre Quelle verlinken) — nur der Dup/extend- und Merge-Stub-Pfad
# wird typ-bewusst.
DEDUP_EXCLUDE_TYPES = frozenset({"literature", "moc", "merge-stub"})

# #8 Body-Redundanz-Detektion: Schwelle, ab der zwei DISTINKTE create-Notes EINES Laufs
# als inhaltlich stark überlappend geflaggt werden (seiteneffekt-freier Review-Hinweis, kein
# Merge, kein Strip). Zwei empirische Gates (Ebner-Audit 2026-06-23) zeigten: solche
# Geschwister sind weder mergebar (distinkte Konzepte) noch satz-strippbar (Redundanz
# paraphrasiert: exakt 0/10, fuzzy≥0.93 nur 1/10 Sätze) — der einzige verlustfreie Eingriff
# ist ein Flag für den menschlichen Reviewer. Default 0.90 liegt deutlich über typischer
# Distinkt-Note-Cosine, unter dem gemessenen #8-Paar (0.967). Tiefer als der ER-Hard-Merge-
# Gate (0.985), weil ein Flag risikolos ist; ENV-überschreibbar für Kalibrierung.
REDUNDANT_SIBLING_COSINE_THRESHOLD = float(
    os.getenv("ATOMIC_AGENT_REDUNDANT_SIBLING_COSINE", "0.90"))

# Chunk-Größe Fallback (Wörter)
CHUNK_WORDS = 3000

# Textqualitäts-Gate (G6/#27): mittlere Wörter pro nichtleerer Seite, unter denen
# der extrahierte Text als zu dünn gilt (gescannt/kaputt/copy-protected) → OCR-Warnung.
# Bewusst KONSERVATIV und UNKALIBRIERT: eine normale Buch-Textseite hat 250–500 Wörter,
# ein un-OCR'tes Scan-PDF ~0. 50 trennt das mit großem Sicherheitsabstand und minimalem
# False-Positive-Risiko. Das Gate warnt nur (fail-open), blockiert nie. Echte
# Schwellen-Kalibrierung gegen Gold-Labels ist Issue #29/G3, nicht hier.
MIN_WORDS_PER_PAGE = 50

# Maximale Chunk-Anzahl für kurze Dokumente (< 50 Seiten).
# Verhindert Chapter-Split-Explosion bei Review-Artikeln die viele Section-Header haben
# aber kein echtes Buch sind. 28 Chunks bei 12-Seiten-Paper war der v35-Bug.
# Bei Überschreitung: Fallback auf Word-Count-Splitting.
MAX_CHUNKS_SHORT_DOC = 10
MAX_PAGES_SHORT_DOC  = 50

# Token-Budget pro Pipeline-Lauf (Hard-Cap, Fail-Fast bei Überschreitung)
MAX_TOKENS_PER_RUN = 500_000

# Concurrency-Cap für parallele Claude-Calls
MAX_CONCURRENT_CALLS = 4

# Pro-Call Hard-Timeout (Sekunden). Wenn ein Call länger braucht (Anthropic-Stuck,
# Netzwerk-Hänger), wirft die Pipeline TimeoutExpired und macht mit der nächsten Note
# weiter. Beobachtet: normale Calls 5–60s; ein hängender Critic-Call hat 110min blockiert.
# Porst-Run 2026-05-28: lange Sonnet-Extractor-Calls brauchen bis ~160s reine
# Output-Zeit; 180s schnitt die mittlere Verteilung zu oft ab.
CALL_TIMEOUT_SEC = int(os.getenv("ATOMIC_AGENT_CALL_TIMEOUT", "300"))

# Pipeline-Version für agent-version Frontmatter
AGENT_VERSION = "v0.3.140"  # RuntimeConfig bis in LLM-Backends verdrahtet

# Background-Extractor (Stage-0.5): Trainingswissen pro Konzept vor Extractor abfragen.
# Deaktivierbar via ENV ENABLE_BACKGROUND_EXTRACTOR=0 — z.B. für Baseline-Eval-Tests
# bei denen Prompt-Erweiterung die Vergleichbarkeit mit v35-Baseline stört.
ENABLE_BACKGROUND_EXTRACTOR = os.getenv("ENABLE_BACKGROUND_EXTRACTOR", "0") not in ("0", "false", "False")

# Extraction-Modus: A/B/C-Test für Halluzinations-Reduktion (2026-05-20)
# "prose"  = aktueller Modus (Fließtext mit Inline-Ankern)
# "table"  = Claim-Tabelle → Prose-Agent konvertiert (hohe Precision, niedrigerer Recall)
# "hybrid" = Claim-Tabelle (Kernfakten) + freier Synthese-Absatz (Mittelweg)
EXTRACTION_MODE = os.getenv("ATOMIC_AGENT_EXTRACTION_MODE", "prose")

# no-LLM-Modus: Stage-6-Agents (Verifier, CrossRef, Critic) überspringen LLM-Calls
# und nutzen deterministische FOSS-Alternativen. Extractor + Planner sind nicht ersetzbar
# → Pipeline bricht bei ENABLE_LLM=0 nach Stage 4 ab wenn kein --source-drafts-cache.
# Primär für kostenfreie E2E-Eval-Regressionstests (Stage-6-only via Drafts-Cache).
# Aktivieren: ENABLE_LLM=0 oder --no-llm (orchestrator-Flag setzt env var).
ENABLE_LLM = os.getenv("ENABLE_LLM", "1") not in ("0", "false", "False")

# Entity-Resolution-Pipeline (4-Stage: Blocking → Embedding → Clustering → Canonicalization)
# Cosine-Threshold für Body-Cluster. Empirisch zu kalibrieren — Start bei 0.85 (GraphRAG-
# Pattern, LlamaIndex Default-Bereich). Höher = strenger (weniger Cluster), niedriger =
# großzügiger (mehr Merges).
# Body-Cosine-Threshold für Auto-Cluster (>= Schwelle → auto-merge ohne LLM).
# Angehoben von 0.85 auf 0.985 nach v35-False-Positive (cos=0.974 triggerte Merge
# von "Information Behavior" + "Geschichte der IB-Forschung" — inhaltlich getrennt,
# sprachlich eng verwandt). Cross-Model-Review Mai 2026: beide Modelle empfahlen 0.985.
# Ambiguous Zone 0.85-0.985 → ENABLE_LLM_DEDUP entscheidet per Haiku-Call.
ER_BODY_COSINE_THRESHOLD = 0.985

# Title-Cosine-Threshold für semantic deduplication (v35). Adressiert die Lücke
# bei Titel-Varianten mit null Token-Overlap (z.B. Übersetzung EN-DE).
# Schwelle 0.93: konservativer gewählt nach Afzal-Run-Befund (Issue #2) — Präfix-
# dominante Titel ("Kompetenzbereich X") erzeugten bei 0.88 false-positive
# Stage-2.5-Paare. 0.93 hält DE-EN-Übersetzungs-Match (>0.95) weiterhin sicher.
ER_TITLE_COSINE_THRESHOLD = 0.93
# Blocking: Title-Token-Subset als HARD-Constraint (siehe entity_resolution).
# Jaccard-Konstante ist deprecated (vorher als Vorfilter genutzt) — Subset-Test
# ist strukturell präziser und verhindert ISP-Phase-Kollaps.
ER_BLOCKING_JACCARD = 0.3  # noch importiert für Backward-Compat, nicht mehr verwendet
# Feature-Flag — bei False: ER-Stage übersprungen, Drafts unverändert weitergereicht.
# Default True (Pipeline-Default), per ENV `ENABLE_ENTITY_RESOLUTION=0` deaktivierbar.
ENABLE_ENTITY_RESOLUTION = os.getenv("ENABLE_ENTITY_RESOLUTION", "1") not in ("0", "false", "False")
# Asymmetrischer Subset: erlaube Cluster nur wenn |b\a| ≤ ER_MAX_TOKEN_DIFF.
# Hub-Sub-Pairs (z.B. „Information Need" ⊂ „Taylor Vier-Stufen-Typologie Information Need")
# haben |b\a| ≥ 2 → werden ausgeschlossen. Author-Suffix-Pattern („X" vs „X (Bates)") erlaubt.
ER_MAX_TOKEN_DIFF = 1
# Hub-Generic-Token-Blocklist: kürzerer Titel der NUR aus diesen Tokens besteht ist
# wahrscheinlich ein Hub-Konzept, kein Duplikat-Kandidat. Codex-Cross-Review-Empfehlung
# (2026-05-09): Token-Differenz allein greift „Information Need" vs „Wilson Information Need"
# nicht ab.
ER_HUB_GENERIC_TOKENS = frozenset({
    "information", "need", "needs", "behavior", "behaviour",
    "search", "process", "system", "user", "model", "concept",
    "theory", "framework", "principle",
})
# Canonicalization-Modell: Body-Merge ist kreativ-synthetische Aufgabe → Opus
MODEL_CANONICALIZER = MODEL_OPUS

# LLM-Dedup (Stage 2.5): Haiku-Batch-Call für ambiguous Zone (0.3 ≤ cosine < ER_BODY_COSINE_THRESHOLD).
# Paare die Stage-1-Blocking passiert haben aber zu geringe Body-Cosine für Auto-Cluster
# werden per LLM binär entschieden (SAME/DIFFERENT). Ein Call pro Batch, Haiku = kosteneffizient.
# Untere Schranke: unterhalb ER_AMBIGUOUS_LOWER definitiv verschieden → kein LLM-Call.
ENABLE_LLM_DEDUP = os.getenv("ENABLE_LLM_DEDUP", "1") not in ("0", "false", "False")
# Untergrenze 0.6: unter 0.6 Body-Cosine teilen Texte kaum semantischen Inhalt → definitiv DIFFERENT.
# Gemini-Review 2026-05-14: 0.3 war zu niedrig (riesige Batches, sinnlose LLM-Calls).
ER_AMBIGUOUS_LOWER = float(os.getenv("ER_AMBIGUOUS_LOWER", "0.6"))
MODEL_LLM_DEDUP = MODEL_HAIKU  # binäre Entscheidung, kein Opus nötig

# Snapshot aller Agent-Modell-Zuweisungen — für Run-Trace und Eval-Vergleiche.
MODEL_CONFIG = {
    "planner":       MODEL_PLANNER,
    "extractor":     MODEL_EXTRACTOR,
    "verifier":      MODEL_VERIFIER,
    "cross_ref":     MODEL_CROSS_REF,
    "critic":        MODEL_CRITIC,
    "canonicalizer": MODEL_CANONICALIZER,
}

# Verifier Tier-3: semantischer Pre-Pass via sentence-transformers.
# Paraphrasen-Matching zwischen Ankerzitat und Chunk-Sätzen. Kein neues Modell —
# nutzt paraphrase-multilingual-MiniLM-L12-v2 (bereits geladen für ER-Embeddings).
# Schwelle 0.75 nach FOSS-NLP-Recherche 2026-05-12 für Paraphrase-Detection.
SEMANTIC_PREPASS_THRESHOLD = float(os.getenv("SEMANTIC_PREPASS_THRESHOLD", "0.75"))

# Eval-Quality v4: Adaptive TOP_K Thresholds für Retrieval-Kontext.
# Gemini-Review 2026-05-18: Thresholds modell-spezifisch (paraphrase-multilingual-MiniLM-L12-v2).
# Bei Modellwechsel neu kalibrieren — stummes Kaputtgehen ohne diese Konstanten nicht möglich.
EVAL_ADAPTIVE_K_HIGH = float(os.getenv("EVAL_ADAPTIVE_K_HIGH", "0.85"))  # → TOP_K=2
EVAL_ADAPTIVE_K_MID  = float(os.getenv("EVAL_ADAPTIVE_K_MID",  "0.65"))  # → TOP_K=3

# NLI-Validation: AND-Kombination mit Haiku für CrossRef-Widerspruchserkennung.
# Haiku identifiziert Widersprüche, DeBERTa bestätigt — nur bei Übereinstimmung
# beider wird ein quality_flag gesetzt. Reduziert False-Positives (v21-Problem:
# Atomic-Notes-Hierarchie fälschlich als Widerspruch).
# Modell: cross-encoder/nli-deberta-v3-small (~70MB, auto-download bei erstem Use).
# Default: disabled — per ENV ENABLE_NLI_VALIDATION=1 aktivieren oder nach
# manuellem pip install sentence-transformers (bereits vorhanden für ER).
ENABLE_NLI_VALIDATION = os.getenv("ENABLE_NLI_VALIDATION", "0") in ("1", "true", "True")
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
# Schwellenwert für DeBERTa contradiction-Score (Softmax-Probability 0–1).
# 0.7 = konservativ (hohe Precision, akzeptiert Recall-Verlust).
# Kalibrierung empfohlen: Gold-Dataset 50–100 Paare, ROC-Kurve, Youden-Index.
NLI_CONTRADICTION_THRESHOLD = 0.7

# mDeBERTa NLI-Scorer für eval_quality.py (EVAL_VERSION 1.3).
# Ersetzt MiniLM-Cosine bei Cross-Language durch echte Entailment-Erkennung.
# Modell: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (~280MB, 100+ Sprachen)
# 512-Token-Limit → Token-Budget-Window-Extraction für lange PDF-Seiten.
# Default: disabled — per ENV ENABLE_MDEBERTA_NLI=1 aktivieren.
ENABLE_MDEBERTA_NLI = os.getenv("ENABLE_MDEBERTA_NLI", "0") in ("1", "true", "True")
MDEBERTA_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# Thresholds für NLI-Labels (unkalibriert — Youden-J-Kalibrierung vorgesehen).
# entailment >= CONFIRMED → confirmed; contradiction >= CONTRA → hallucinated; else uncertain.
# CONFIRMED: nur bei direkten Zitaten / sehr engem Entailment (0.7+)
# CONTRA: ab 0.3 verdächtig — Youden-J-Kalibrierung ausstehend
MDEBERTA_THRESHOLD_CONFIRMED = float(os.getenv("MDEBERTA_THRESHOLD_CONFIRMED", "0.7"))
MDEBERTA_THRESHOLD_CONTRA = float(os.getenv("MDEBERTA_THRESHOLD_CONTRA", "0.3"))

# Preise in USD pro Million Tokens (Stand 2026-05)
# cache_read: Anthropic berechnet Cache-Reads als eigene günstige Zeile
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude (Anthropic API) — https://anthropic.com/pricing
    "claude-opus-4-7":   {"input": 15.0,  "output": 75.0,  "cache_read": 1.50},
    "claude-sonnet-4-6": {"input": 3.0,   "output": 15.0,  "cache_read": 0.30},
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.0,   "cache_read": 0.03},
    # Gemini (Google API) — https://ai.google.dev/pricing
    "gemini-3.1-pro":    {"input": 2.50,  "output": 10.0,  "cache_read": 0.0},
    "gemini-2.5-flash":  {"input": 0.075, "output": 0.30,  "cache_read": 0.0},
    # OpenAI
    "gpt-4o":            {"input": 5.0,   "output": 15.0,  "cache_read": 0.0},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60,  "cache_read": 0.0},
    # Subscription-Aliase → kein Preis (kein API-Key nötig)
    "opus":  {},
    "haiku": {},
}


def compute_cost_per_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Kosten eines einzelnen LLM-Calls in USD.
    Gibt 0.0 zurück wenn BACKEND!='api' oder Modell nicht in MODEL_PRICING.
    Strips provider-Prefix (z.B. 'anthropic/claude-opus-4-7' → 'claude-opus-4-7').
    """
    if BACKEND == "subscription":
        return 0.0
    # Provider-Prefix normalisieren: "anthropic/model" → "model"
    model_key = model.split("/", 1)[-1] if "/" in model else model
    pricing = MODEL_PRICING.get(model_key, {})
    if not pricing:
        return 0.0
    per_m = 1_000_000
    return round(
        pricing.get("input",      0.0) * input_tokens      / per_m +
        pricing.get("output",     0.0) * output_tokens     / per_m +
        pricing.get("cache_read", 0.0) * cache_read_tokens / per_m,
        6,
    )
