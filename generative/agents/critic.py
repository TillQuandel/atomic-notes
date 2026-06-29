"""Critic-Agent: prüft Atomic-Note gegen 5 Tests, davon 3 Hard-Gates.

Hard-Gates (Glance, Future-Self, Quellen): einer fail → Note geht zwingend nach 00-inbox/,
unabhängig vom Score. Future-Self-Test läuft als deterministische Regex-Vorprüfung
vor dem LLM-Call (fängt 30–50% der Fails ohne Token-Verbrauch).
"""

from __future__ import annotations
import re

from generative.agents.base import call_claude, trace_event
from generative.agents.structured_output import parse_critic_output
from generative import config as _config
from generative.config import MODEL_CRITIC
from generative.schemas.atomic_note import AtomicNoteDraft

# --- Future-Self-Regex-Pre-Check (deterministisch, kein LLM) ---

# Verweise auf Originaldokument — Note wird beim Lesen in 5 Jahren nicht selbsttragend
_REF_TO_SOURCE_DOC = re.compile(
    r"\b(siehe|vgl\.?|in)\s+(Abschnitt|Kapitel|Section|Chapter|S\.?)\s*\d+",
    re.IGNORECASE,
)

# Zeitliche Deixis — „aktuell" bedeutet in 5 Jahren etwas anderes als heute.
# Wort-Stem-Match (Suffixe wie -e, -er, -en, -es, -em zugelassen) + Umlaut/ASCII-Varianten.
_TEMPORAL_DEIXIS = re.compile(
    r"\b("
    r"aktuell\w*|"
    r"j(?:ü|ue)ngst\w*|"
    r"j(?:ü|ue)nger\w*|"
    r"k(?:ü|ue)rzlich|"
    r"momentan\w*|"
    r"derzeit\w*|"
    r"heutzutage|"
    r"neuer\w*|"  # 'neu' allein ist zu generisch — nur 'neuere/neueste/neuer'
    r"recently|currently|nowadays"
    r")\b",
    re.IGNORECASE,
)

# Pronomen-Heuristik: Absatz beginnt mit „Dieser/Jener Ansatz/Punkt/Aspekt" ohne nominales
# Subjekt davor → wahrscheinlich Antezedens fehlt. Nur Absatzbeginn (^), nicht inline.
_DANGLING_PRONOUN = re.compile(
    r"^\s*(Dieser|Jener|Diese|Jene|Dieses|Jenes)\s+(Ansatz|Punkt|Aspekt|Begriff|Autor|Autorin|Modell|Theorie|Konzept)\b",
    re.MULTILINE,
)


def future_self_regex_check(body: str) -> tuple[bool, list[str]]:
    """Returns (pass, list of violation reasons).

    Pass = keine Verstöße gefunden. Violations sind aussagekräftige 1-Satz-Begründungen
    fürs quality_flags, damit der Future-Reader/Critic weiß warum die Note in Inbox ging.
    """
    violations: list[str] = []
    if m := _REF_TO_SOURCE_DOC.search(body):
        violations.append(f"Verweis auf Originaldokument: '{m.group(0)}' — in 5 Jahren ohne PDF nicht auflösbar")
    if m := _TEMPORAL_DEIXIS.search(body):
        violations.append(f"Zeitliche Deixis: '{m.group(0)}' — wird beim späteren Lesen falsch verstanden")
    if m := _DANGLING_PRONOUN.search(body):
        violations.append(f"Hängendes Pronomen am Absatzbeginn: '{m.group(0).strip()}' — Antezedens unklar")
    return (len(violations) == 0, violations)


# --- Hub-Detector (deterministisch, kein LLM) ---

# Wikilinks im Body — Form `[[Title]]`, `[[Title|Display]]`, `[[Title#Anchor]]`.
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[#|][^\]]*)?\]\]")

# Hub = Note die ≥3 existierende Atomic-Konzepte als Sub-Struktur referenziert.
HUB_MIN_SUBCONCEPTS = 3
# Mehrwort-Filter für Plain-Text-Match: Single-Token-Titel (z.B. „Information", „Theory")
# matchen zu breit. Multi-Token-Match gegen existing_concepts-Keys ist hinreichend
# spezifisch und vermeidet False-Positives.
_HUB_MIN_TOKENS_FOR_PLAIN_MATCH = 2
# Bridging-Coefficient-Schwelle: ab welcher Sub-Cluster-Dichte (Anteil verlinkter
# Sub-Paare an allen Sub-Paaren) gilt das Sub-Set als Geschwister-Cluster und
# disqualifiziert die Note als Hub. 0.5 = mindestens die Hälfte der ungerichteten
# Paare ist verlinkt. HITS-/Bridging-Coefficient-inspiriert: True Hubs verbinden
# unverlinkte Cluster, atomic Notes nennen interlinkende Geschwister.
# Recherche-Befund (Burt 2000, Cat2Type, Bhosale 2013): pure Topologie greift bei
# kohärenten Hub-Notes (alle Subs aus einer Domain) nicht — Marker-Whitelist im
# Titel (Wikipedia-Disambig-Pattern, Cat2Type-Template-Signal) trumpft die Density.
HUB_SIBLING_DENSITY_THRESHOLD = 0.5
# Title-Marker, die eine Note als Hub/Übersichts-Note kennzeichnen — überschreibt
# den Bridging-Cluster-Test. Klein gehalten und kuratiert wachsen lassen, statt
# spekulative Liste. Wikipedia/PKM-Systeme nutzen genau dieses Annotation-Pattern.
HUB_OVERVIEW_MARKERS = {
    "modell",
    "begriffsgeschichte",
    "framework",
    "moc",
    "übersicht",
    "atlas",
    "taxonomie",
}

# Post-Critic Quellen-Test-Override: ab welcher Body-Anker-Coverage (Anteil der
# Saetze mit `(S. N)`-Anker) wird ein Critic-quellen_test=false als Sampling-
# Drift abgewiesen. 0.85 = mindestens 85% der substantiellen Saetze haben einen
# Anker. Konservativ: deckt Critic-FPs (Halluzinations-Drift wie ISP-Collection)
# ohne legitime Quellen-Luecken durchzulassen.
ANCHOR_COVERAGE_OVERRIDE = 0.85
from generative.pipeline.anchor_patterns import (
    SENTENCE_SPLIT_RE as _SENTENCE_SPLIT_RE,
    PAGE_ANCHOR_RE as _PAGE_ANCHOR_RE,
)

_MIN_SUBSTANTIVE_LEN = 40  # Saetze < 40 Zeichen werden nicht als „substantiell" gezaehlt


def _sentence_anchor_coverage(body: str) -> float:
    """Anteil substantieller Saetze (>=_MIN_SUBSTANTIVE_LEN Zeichen) mit `(S. N)`-Anker."""
    sentences: list[str] = []
    for para in body.split("\n\n"):
        stripped = para.strip()
        if not stripped or stripped.startswith(("- ", "* ", "#", "> ", "|")):
            continue
        sentences.extend(_SENTENCE_SPLIT_RE.split(stripped))
    substantive = [s for s in sentences if len(s.strip()) >= _MIN_SUBSTANTIVE_LEN]
    if not substantive:
        return 0.0
    anchored = sum(1 for s in substantive if _PAGE_ANCHOR_RE.search(s))
    return anchored / len(substantive)


def _has_overview_marker(title: str) -> bool:
    title_low = title.lower()
    return any(re.search(rf"\b{re.escape(m)}\b", title_low) for m in HUB_OVERVIEW_MARKERS)


def _strip_disambig(s: str) -> str:
    """Entfernt Disambig-Klammern für Plain-Text-Match: 'Foo (Bar)' → 'Foo'."""
    return re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()


def _sibling_cluster_density(
    sub_keys: list[str], existing_concepts: dict[str, str], concept_links: dict[str, set[str]]
) -> float:
    """Anteil ungerichtet verlinkter Sub-Paare. 0.0 = vollständig unverlinkt
    (Hub-Pattern), 1.0 = vollständiger Cluster (Atomic-mit-Geschwistern).

    Resolves Sub-Keys zu File-Paths via existing_concepts (Alias-robust). Ein Paar
    (s_i, s_j) gilt als verlinkt, wenn s_i's Note s_j referenziert ODER umgekehrt.
    """
    paths = [existing_concepts.get(k.lower()) for k in sub_keys]
    paths = [p for p in paths if p]
    n = len(paths)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    linked_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_i, p_j = paths[i], paths[j]
            i_to_j = p_j in concept_links.get(p_i, set())
            j_to_i = p_i in concept_links.get(p_j, set())
            if i_to_j or j_to_i:
                linked_pairs += 1
    return linked_pairs / total_pairs


def _canonical_title_from_path(path: str) -> str:
    """Leitet canonical Title aus File-Path ab. `<sibling:Title>`-Pseudo-Pfade
    liefern den Suffix; echte .md-Pfade den File-Stem. Damit landen Plain-Match-
    Treffer in MoC-Frontmatter mit korrektem Title-Casing statt lowercase-Key."""
    if path.startswith("<sibling:") and path.endswith(">"):
        return path[len("<sibling:") : -1]
    from pathlib import Path as _P

    return _P(path).stem


def hub_test(body: str, existing_concepts: dict[str, str], self_keys: set[str] | None = None) -> list[str]:
    """Listet existierende Atomic-Konzepte, die als Sub-Struktur im Body referenziert sind.

    Zwei Match-Kanäle: Wikilinks (`[[Title]]`) und Plain-Text-Mentions mit Wortgrenzen
    für Multi-Token-Titel. Disambig-Klammern werden für Plain-Match abgestreift
    (`Red Thread of Information (Bates)` → matcht auch bei `red thread of information`).
    `self_keys` filtert Selbstreferenzen (Title + Aliases der Draft).

    Rückgabe enthält canonical Titles (Title-Case): Wikilink-Match übernimmt die
    User-Schreibweise aus `[[..]]`, Plain-Match leitet den Title aus dem File-Stem
    von `existing_concepts[key]` ab. Damit landet `[[Information Search Process]]`
    statt `[[information search process (kuhlthau)]]` in der MoC-Frontmatter.
    """
    if not existing_concepts:
        return []
    self_keys = self_keys or set()
    seen: set[str] = set()
    matches: list[str] = []

    for m in _WIKILINK_RE.finditer(body):
        raw = m.group(1).strip()
        key = raw.lower()
        if key in self_keys or key in seen:
            continue
        if key in existing_concepts:
            seen.add(key)
            matches.append(raw)

    # Plain-Text: für jeden Concept-Key prüfen ob Volltext oder Disambig-Stem im Body steht.
    # Mapping stem→canonical_key, damit ein Body-Hit auf 'red thread of information' den
    # Original-Key `red thread of information (bates)` als Match registriert.
    # Cheap substring-Pre-Filter (body_lower) reduziert die regex-Calls auf
    # Tatsächliche Kandidaten — O(n·m) → O(n + m·n_hit).
    body_lower = body.lower()
    for key in existing_concepts:
        if key in seen or key in self_keys:
            continue
        candidates = {key}
        stem = _strip_disambig(key)
        if stem and stem != key:
            candidates.add(stem)
        for cand in candidates:
            if len(cand.split()) < _HUB_MIN_TOKENS_FOR_PLAIN_MATCH:
                continue
            if cand not in body_lower:
                continue
            if re.search(rf"\b{re.escape(cand)}\b", body, re.IGNORECASE):
                seen.add(key)
                matches.append(key)
                break

    # Substring-Bleed-Dedup: wenn Tokens(A) ⊆ Tokens(B), zähle A nicht zusätzlich
    # (sonst zählen 'red thread', 'red thread of information', 'red thread of information (bates)'
    # als 3 separate Sub-Konzepte für ein und denselben Body-Hit).
    # Disambig-Klammern vor Tokenisierung strippen, sonst kommen `(bates)` etc. als Pseudo-
    # Token in das Set und zerstören die Subset-Beziehung. Bei identischen Token-Sets nach
    # Strippen (z.B. 'red thread of information' vs 'red thread of information (bates)')
    # nur den canonical-längsten (raw incl. Disambig) als Repräsentanten behalten.
    token_lists = [(set(_strip_disambig(m).lower().split()), m) for m in matches]
    deduped: list[str] = []
    for i, (tokens_a, m_a) in enumerate(token_lists):
        absorbed = False
        for j, (tokens_b, m_b) in enumerate(token_lists):
            if i == j:
                continue
            if tokens_a < tokens_b:
                absorbed = True
                break
            if tokens_a == tokens_b and (len(m_b) > len(m_a) or (len(m_b) == len(m_a) and j < i)):
                absorbed = True
                break
        if not absorbed:
            deduped.append(m_a)

    # Casing-Normalisierung: Plain-Match-Pfad hat lowercase-Keys angefügt, Wikilink-
    # Match die User-Schreibweise. Für die MoC-Frontmatter brauchen wir canonical
    # Title-Case. Wikilinks bleiben unverändert (User-Casing ist authoritativ);
    # für jeden Match der lowercase im existing_concepts-Mapping liegt, ersetze
    # ihn durch den canonical Title aus dem File-Stem.
    canonical: list[str] = []
    for m in deduped:
        if m.lower() == m and m in existing_concepts:
            canonical.append(_canonical_title_from_path(existing_concepts[m]))
        else:
            canonical.append(m)
    return canonical


# --- LLM-Critic ---

_PROMPT = """Du prüfst eine Atomic Note gegen FÜNF Tests. Drei davon sind Hard-Gates — bei Fail muss die Note zur manuellen Review.

## Wichtig: Sekundärzitate sind OK, wenn als solche markiert
Diese Note wurde aus EINEM Quellen-PDF extrahiert. Sekundärzitate über andere Autoren sind erwartet und legitim, wenn klar markiert (z.B. „Wilson (1981) argumentiert … (zit. n. Schlebbe & Greifeneder, S. 5)" oder im laufenden Text Schlebbe & Greifeneder als Bezugsquelle erkennbar). **Verlange NICHT, dass die Note Originalquellen direkt zitiert** — das ist Aufgabe einer späteren Erweiterung. Was du verlangst: jede Aussage hat einen nachvollziehbaren Anker mit Seitenzahl in der gelesenen Quelle.

Monoquellige Notes sind erwartet. Die `synthesis-confidence: low/medium`-Markierung trägt dieses Risiko bereits. Du musst es nicht zusätzlich über Hard-Gates bestrafen.

## Die fünf Tests

1. **Title-Test** (soft): Ist der Titel knapp und enthält er nur EINE Idee? (kein "X und Y", kein "Einführung in X")
2. **Glance-Test** (HARD): Steht die Kernaussage klar im ersten Absatz, ohne dass man die ganze Note lesen muss?
3. **Future-Self-Test** (HARD): Versteht ein Leser in 5 Jahren ohne Original-PDF/Vault/Erinnerung sofort, worum es geht? Akronyme aufgelöst beim ersten Vorkommen? Sekundärzitate als solche markiert (per `zit. n.` oder klarem Bezug auf die gelesene Quelle)? PASS auch bei monoquelligen Notes wenn die EINE Quelle nachvollziehbar ist.
4. **Quellen-Test** (HARD): Hat jede inhaltliche Aussage einen Text-Anker (Zitat oder Paraphrase) mit **Seitenzahl in der gelesenen Quelle**? Sekundärzitate brauchen Anker auf die GELESENE Quelle, nicht auf die Originalquelle. PASS wenn die Anker in der gelesenen Quelle vollständig sind.
5. **Deletion-Test** (soft): Wenn diese Note gelöscht würde — verschwindet substanzielle Komplexität (Pass-Through, Quellen-Inflation, Hub-Doppelung) oder taucht sie verteilt in N anderen Notes wieder auf? Pass = Note verdient Existenz.

## Regex-Vorprüfung (bereits gelaufen)

{regex_violations}

## Output — NUR dieses Format, kein erklärender Text, KEIN JSON:

title_test: true
glance_test: true
future_self_test: true
quellen_test: true
deletion_test: true
score: 4
<!--REVISION_HINT-->
1 Satz konkrete Verbesserung — kann beliebige Quotes oder Doppelpunkte enthalten.
<!--END-->

**Format-Regeln**: Header-Lines `key: true|false` für die Tests, `score: 0-5` als Integer. `<!--REVISION_HINT-->` als Heredoc-Block — wenn keine Revision nötig ist, lass den ganzen Block weg (kein `null`-Marker schreiben).

## Note zu prüfen

### Titel: {title}

### Body
{body}

### Vorhandene Anker
{anchors}
"""


def _log_score_result(draft) -> None:
    trace_event(
        "critic",
        "score_result",
        {
            "title": draft.title,
            "score": draft.critic_score,
            "hard_gates_pass": draft.hard_gates_pass,
        },
    )


def run(
    draft: AtomicNoteDraft,
    existing_concepts: dict[str, str] | None = None,
    concept_links: dict[str, set[str]] | None = None,
) -> AtomicNoteDraft:
    try:
        # Schritt 0: Hub-Detector (Hebel #4) — vor LLM-Call.
        # Notes mit ≥HUB_MIN_SUBCONCEPTS Wikilinks auf existierende Atomic-Notes im Body
        # sind strukturell Hubs (referenzieren Sub-Konzepte) und werden als MoC gerendert.
        # Hub-Routing trumpft create/extend: strukturell ist eine Note mit vielen Sub-
        # Konzept-Referenzen ein Hub, auch wenn cross_reference Duplikat-Risiko sah.
        if existing_concepts and draft.action != "hub":
            self_keys = {draft.title.lower()} | {a.lower() for a in draft.aliases}
            sub = hub_test(draft.body, existing_concepts, self_keys=self_keys)
            if len(sub) >= HUB_MIN_SUBCONCEPTS:
                # Bridging-Test: dichter Sub-Cluster = Atomic-mit-Geschwistern.
                # Marker-Whitelist im Titel trumpft die Density — kohärente Hub-Notes
                # (z.B. „ISP-Modell (Kuhlthau)") werden über das explizite Naming-Signal
                # gerettet. Recherche-Anker: Wikipedia-Disambig-Templates, Cat2Type
                # Category-Embeddings, Burt 2000 Structural-Holes-Theorie.
                density = _sibling_cluster_density(sub, existing_concepts, concept_links) if concept_links else 0.0
                has_marker = _has_overview_marker(draft.title)
                atomic_pattern = density >= HUB_SIBLING_DENSITY_THRESHOLD and not has_marker

                if atomic_pattern:
                    # Soft-Flag (Tavily-/Obsidian-PKM-Pattern): kein Hub-Routing, aber
                    # Hinweis im Quality-Flag — User kann beim Inbox-Review eingreifen.
                    draft.quality_flags.append(
                        f"⚠️ Hub-Kandidat verworfen: Sub-Cluster-Dichte {density:.2f} ≥ "
                        f"{HUB_SIBLING_DENSITY_THRESHOLD} ohne Übersichts-Marker im Titel "
                        f"(atomic-Pattern, Geschwister-Cluster)"
                    )
                else:
                    draft.action = "hub"
                    draft.hub_subconcepts = sub
                    density_note = f", Sub-Dichte {density:.2f}" if concept_links else ""
                    marker_note = (
                        " [Marker-Override]" if has_marker and density >= HUB_SIBLING_DENSITY_THRESHOLD else ""
                    )
                    draft.quality_flags.append(
                        f"Hub-Kandidat: referenziert {len(sub)} Sub-Konzepte "
                        f"({', '.join(sub[:4])}{density_note}){marker_note} → MoC-Routing"
                    )

        # Schritt 1a: Struktureller Pre-Filter — Note zu schwach für sinnvollen LLM-Critic.
        # Keine Anker + kurzer Body → quellen_test + glance_test sowieso fail → LLM spart nichts.
        body_words = len(draft.body.split())
        if not draft.source_anchors and body_words < 80:
            draft.hard_gates_pass = False
            draft.critic_score = 1
            draft.quality_flags.append(f"⚠️ Critic-Pre-Filter: keine Anker + Body {body_words} Wörter < 80 — Inbox")
            return draft

        # Schritt 1: Regex-Pre-Check (deterministisch, billig)
        regex_pass, violations = future_self_regex_check(draft.body)
        if violations:
            for v in violations:
                draft.quality_flags.append(f"⚠️ Future-Self: {v}")

        regex_block = (
            "Keine Verstöße per Regex gefunden — du kannst future_self_test nach LLM-Urteil setzen."
            if regex_pass
            else "Bereits gefundene Verstöße (LLM darf future_self_test = false setzen, oder zusätzliche Verstöße benennen):\n"
            + "\n".join(f"- {v}" for v in violations)
        )

        anchors_str = (
            "\n".join(f'- "{a.quote}" ({a.page or "keine Seite"})' for a in draft.source_anchors) or "(keine Anker)"
        )

        # Body-Truncation: Atomic-Notes bleiben i.d.R. < 6000 Zeichen. Wenn die Note
        # länger ist, urteilt der Critic nur über den Head — Anker/Aussagen am Tail
        # bleiben ungeprüft. Konservatives Routing in 00-inbox/ via hard_gates_pass=False,
        # damit ein Vault-Auto-Write nicht über unsichtbare Tail-Teile wegabstrahiert.
        _BODY_LIMIT = 6000
        body_for_prompt = draft.body[:_BODY_LIMIT]
        body_truncated = len(draft.body) > _BODY_LIMIT
        if body_truncated:
            draft.quality_flags.append(
                f"⚠️ Critic sah nur erste {_BODY_LIMIT} von {len(draft.body)} Zeichen — "
                f"Tail unbeurteilt, Routing nach 00-inbox/ erzwungen"
            )

        # no-LLM-Modus: deterministisches Scoring ohne API-Call.
        # Primär für kostenfreie E2E-Eval-Regressionstests.
        if not _config.ENABLE_LLM:
            future_self_pass = regex_pass
            quellen_pass = _sentence_anchor_coverage(draft.body) >= ANCHOR_COVERAGE_OVERRIDE
            glance_pass = True  # nicht deterministisch prüfbar ohne LLM
            passes = sum([future_self_pass, quellen_pass])
            draft.hard_gates_pass = False
            if body_truncated:
                draft.hard_gates_pass = False
            draft.critic_score = min(5, passes + 2)  # 2-4/5 ohne LLM
            draft.quality_flags.append("ℹ️ Critic: no-LLM-Modus (deterministisch)")
            return draft

        prompt = _PROMPT.format(
            regex_violations=regex_block,
            title=draft.title,
            body=body_for_prompt,
            anchors=anchors_str,
        )

        try:
            raw = call_claude(prompt, model=MODEL_CRITIC, agent="critic")
            data, parse_warnings = parse_critic_output(raw)
            for w in parse_warnings:
                draft.quality_flags.append(f"⚠️ Critic-Parse-Warning: {w}")
        except RuntimeError as e:
            draft.critic_score = 0
            draft.hard_gates_pass = False
            draft.quality_flags.append(f"⚠️ Critic-Check fehlgeschlagen: {str(e)[:100]}")
            return draft

        # LLM-Urteil zu future_self_test mit Regex-Result kombinieren — Regex hat Vetorecht
        future_self_pass = regex_pass and data["future_self_test"]
        glance_pass = data["glance_test"]
        quellen_pass = data["quellen_test"]

        # Post-Critic Quellen-Override: wenn Critic quellen_test=false setzt, aber jeder
        # substantielle Satz im Body einen `(S. N)`-Anker trägt (Coverage ≥ COVERAGE_THRESHOLD),
        # ist das ein Critic-Sampling-Drift (Eval 2026-05-07: ISP-Collection — Critic verlangt
        # andere Seitenzahl als am Satzende steht). Deterministischer Override schützt vor
        # solchen False-Positives.
        if not quellen_pass:
            coverage = _sentence_anchor_coverage(draft.body)
            if coverage >= ANCHOR_COVERAGE_OVERRIDE:
                quellen_pass = True
                draft.quality_flags.append(
                    f"Quellen-Test-Override: Body-Anker-Coverage {coverage:.0%} ≥ "
                    f"{ANCHOR_COVERAGE_OVERRIDE:.0%} (Critic-FP gefiltert)"
                )

        draft.hard_gates_pass = glance_pass and future_self_pass and quellen_pass
        # Truncation forciert Inbox-Routing (Tail unbeurteilt → kein Vault-Auto-Write).
        if body_truncated:
            draft.hard_gates_pass = False

        # Score wird vom Parser bereits auf 0–5 geclamped + robust gegen
        # bool/float/string-Drift. Defensive Fallback hier nicht mehr nötig.
        draft.critic_score = data["score"]

        hint = data["revision_hint"]
        if hint:
            draft.revision_hint = hint
            draft.quality_flags.append(f"Critic: {hint}")

        # CERQual-Konsistenz: Quellen-Test fail → confidence runter (keine high-Behauptung ohne Anker)
        if not quellen_pass and draft.synthesis_confidence == "high":
            draft.synthesis_confidence = "medium"

        return draft
    finally:
        _log_score_result(draft)
