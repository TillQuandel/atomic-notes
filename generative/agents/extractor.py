"""Extractor-Agent: Chunk + ConceptPlan → Draft-AtomicNotes.

Asynchron — mehrere Chunks parallel verarbeitbar.
Output erfüllt [[Schema-Konzept]]: Body 30–60 Zeilen destilliert, Anker inline mit
Seitenzahl, deutsche Sprache, Akronyme aufgelöst, aliases-Liste, kein Pass-Through.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from generative.agents.base import call_claude_async
from generative.agents.structured_output import parse_extractor_output
from generative.config import MODEL_EXTRACTOR
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor, ConceptPlan

_PROMPT = """Du extrahierst eine Atomic Note aus dem unten stehenden Textabschnitt — schemakonform für einen Obsidian-Vault.

## Quellen-Metadaten (NUR diese verwenden — niemals andere Autorennamen erfinden)
{source_meta}

## Wichtigste Regel — Quellenbindung
JEDE inhaltliche Aussage muss aus dem Text unten stammen. Erfinde NICHTS aus deinem Gedächtnis.
- Der Text ist mit `[S. N]`-Markern versehen — beim Generieren eines Ankers nutze die Seitenzahl der Stelle die du zitierst/paraphrasierst.
- Paraphrasen + Sekundärzitate: am Satzende mit `(S. N)`-Marker (Renderer konvertiert die später deterministisch in Footnotes — du schreibst weiterhin `(S. N)` inline).
- Direktzitate werden als **Block-Quote-Callout** geführt (siehe unten), NICHT als Inline-`„..."` im Fließtext.
- Wenn die Quelle einen anderen Autor nennt (Sekundärzitat), Format: `... führt Wilson aus (zit. n. {author_short}, S. 7)`.
- KEIN `## Quellen`-Block am Body-Ende — der wird vom System automatisch aus den Metadaten generiert.

## Body-Format (Pflicht)

### H1 — prägnanter Satz (PFLICHT, erste Body-Zeile)
Body MUSS mit einer H1-Zeile beginnen die das Thema in einem prägnanten Satz benennt.
- Format: `# {{Title}}: {{Kerncharakteristik in einem Satz}}`
- Kerncharakteristik = was das Konzept inhaltlich beschreibt, NICHT Modell-Verortung.
- Beispiel falsch: `# ADKAR Awareness`
- Beispiel richtig: `# ADKAR Awareness: Bewusstsein für die Notwendigkeit und das Risiko der Veränderung`
- Beispiel richtig: `# Information Search Process: Sechsstufiges Modell der Informationssuche aus Sicht des Suchenden`

### Body-Struktur
- **25–40 Zeilen Markdown, HARDLIMIT 30 Zeilen** für den Fließtext (Footnote-Defs + Block-Quote-Callouts zählen NICHT). Destilliert in deinen Worten, KEIN TOC-Nachbau. Body MUSS vollständig sein — wenn du das Limit erreichst, KÜRZE inhaltlich, brich KEINEN Satz ab.
- **3-Phasen-Struktur** (knapp, ohne sichtbare Header — Übergänge im Fließtext):
  1. **Definition + Einordnung** (3–6 Zeilen): Erster Satz = **Kerncharakteristik** (was es inhaltlich ausmacht / welches Phänomen es beschreibt), NICHT Modell-Verortung. Modell/Rahmen/Author folgen erst im 2.–3. Satz.
  2. **Substanz** (Hauptteil, 15–25 Zeilen): Mechanismus, Eigenschaften, Dimensionen, Phasen.
  3. **Empirie / Begründung** (2–5 Zeilen): wie wurde das Konzept gewonnen / gestützt (Methode, Studienart, Stichprobe).

### Seiten-Anker (Pflicht)
- **JEDE inhaltliche Aussage** trägt einen Seiten-Anker `(S. N)` direkt am Satzende. KEINE Aussage ohne Anker.
- Ausnahme: rein definitorische Einleitungssätze (allgemeine Einordnung) — bei Zweifel Anker setzen.
- Der Renderer konvertiert die `(S. N)`-Marker später deterministisch in Footnotes `[^N]` mit Definitionsblock am Body-Ende. Du schreibst nur die Inline-Marker.
- Wenn mehrere Sätze auf dieselbe Seite verweisen: jeweils eigener `(S. N)`-Marker, der Renderer baut daraus zwei separate Footnotes.

### Block-Quote-Callouts (1–3 pro Note)
- Wähle 1–3 markante **Originalzitate** aus der Quelle und bette sie thematisch passend in den Fließtext ein (an der Stelle wo sie das umgebende Argument stützen, NICHT am Body-Ende gesammelt).
- Format:
  ```
  > [!quote]- {author_short} {{year}}, S. 13
  > „Originalzitat im Quellen-Wortlaut."
  ```
- Direktzitate kommen ausschließlich in Block-Quote-Callouts vor — KEIN inline `„..." (S. 13)` mehr.
- Anti-Deixis-Regel: EN-Originalzitate mit zeitlicher Deixis (`currently`, `recently`, `now`, `today`, `nowadays`, `at present`, `lately`, `new`) NICHT übernehmen — stattdessen Konzept im Fließtext paraphrasieren (Footnote!).
- Stilkonvention für Anführungszeichen im Quote: deutsche Guillemets `„..."` (U+201E und U+201C).
- **Deutsch** — auch wenn Quelle Englisch ist. Block-Quote-Callouts dürfen das EN-Originalzitat enthalten; im Fließtext davor wird auf Deutsch eingeleitet/paraphrasiert.
- **Future-Self-Hard-Gate**: zeitliche Deixis wird im **gesamten** Body geprüft, also auch in Block-Quote-Callouts. Englische Quotes mit `currently`/`recent`/etc. werden NICHT als Quote übernommen — stattdessen den Inhalt im Fließtext paraphrasieren und auf den Quote verzichten.
- **Akronyme beim ERSTEN Vorkommen auflösen**: `ELIS (Everyday Life Information Seeking)`, `ISP (Information Search Process)`, `CERQual (...)`. Auch quelleneigene Abkürzungen wie „BP5" auflösen.
- **Fachbegriffe inline kurz erklären** beim ersten Vorkommen — Note muss in 5 Jahren ohne Original-PDF verständlich sein
- **Verboten**: Sektionen wie „BA-Relevanz", „Anwendung in X", „Fazit für meine Arbeit" — Anwendung gehört in andere Notes, nicht in die Konzept-Note
- **Verboten**: Verweise auf das Originaldokument — Note muss selbsttragend sein. Konkret verboten: „in Kapitel 3", „siehe Abschnitt 5", „vgl. Kapitel 2", „in Chapter 4", „wie im nächsten Kapitel". Stattdessen: Inhalt direkt formulieren oder Seitenzahl angeben.
- **Verboten**: zeitliche Deixis JEDER Inflection — `neu`/`neuer`/`neuere`/`neuerdings`/`neueste`, `aktuell`/`aktueller`, `jüngst`/`jüngere`, `kürzlich`/`unlängst`, `derzeit`/`gegenwärtig`, „in letzter Zeit", „heutzutage". Wird beim späteren Lesen falsch verstanden, weil „neu" relativ zum Quellen-Erscheinungsdatum ist. Ersatzformen: konkrete Jahresangabe (`seit 1990ern`), Prozess-Sprache (`die Begriffsverschiebung von X zu Y`) oder schlicht weglassen.
- **Verboten**: hängende Pronomen am Absatzbeginn ohne nominales Subjekt davor
- **KEINEN `## Quellen`-Block schreiben** — Renderer hängt den deterministisch an. Du schreibst nur den Konzept-Body.

## Aliases (Pflicht)
2–4 Schreibvarianten für Wikilink-Auflösung — DE-Synonyme + EN-Originalbegriff wenn die Quelle Englisch ist.
Beispiel für „Information Search Process (Kuhlthau)": `["ISP", "Information Search Process", "Informationssuchprozess"]`.

## Author-Einführung (F7 — Pflicht beim ersten Vorkommen)
Beim ersten Vorkommen von `{author_short}` im Body kurz einordnen (1 Beisatz, NICHT mehr als ein halber Satz), sonst hält ein kontextfreier Leser den Author für eine weitere zitierte Stimme. Beispiele:
- „Die Informationsforscherin Kuhlthau beschreibt ..." (S. X)
- „Bates, Begründerin des Berrypicking-Modells, argumentiert ..." (S. Y)

Nur bei der ersten Erwähnung. Danach reicht der Nachname.

## Tags (F10 — STRIKT aus Whitelist für `tags`, NIE erfinden)
Wähle 1–3 `tags` AUSSCHLIESSLICH aus der untenstehenden Whitelist. Diese Tags steuern Auto-Note-Mover-Routing und sind autoritativ — Erfindung ist verboten.

{tag_whitelist}

Faustregel: Domain-passende Tags aus dem **Quellnah**-Block bevorzugen wenn vorhanden, **Übrige**-Block nur wenn dort ein wirklich passender Tag steht. KEINEN `uni/ibi/konzept` oder `bachelorarbeit` als Default — nur wenn die Quelle tatsächlich aus einem IBI-/BA-Kontext stammt. Lieber 1 thematisch korrekter Tag als 2–3 mit fehlpassendem Domain-Tag. Wenn nichts in der Whitelist wirklich passt: `tags:` leer lassen.

## Proposed-Tags (Bootstrap für neue Domains)
Wenn KEIN passender Tag in der Whitelist existiert UND die Quelle klar eine neue Domain markiert (z.B. Change-Management ohne `change-management`-Tag im Vault), darfst du in `proposed-tags` 1–2 Vorschläge machen. Strikte Konvention:
- kebab-case, englisch, hierarchisch via `/`
- Maximal 3 Hierarchie-Ebenen (`change-management/adkar/awareness` ist die Obergrenze)
- Aus quellnahen Wörtern abgeleitet (kein erfundener Vault-übergreifender Tag)
- **Kein Routing** — Proposed-Tags lösen kein Auto-Note-Mover aus. User reviewed beim Inbox-Triage.
- Wenn Whitelist-Tags ausreichen: `proposed-tags:` leer lassen.

## Ziel-Konzepte (vom Planner)
{concepts}

## Bereits existierende Notes (nicht duplizieren — bei starker Überschneidung action="extend" mit extend_path)
{existing}

{background_block}{related_mentions_block}## Output — NUR dieses Format, kein erklärender Text, KEINE JSON-Codeblöcke:

<!--NOTE-->
title: Konzeptname (knapp, EINE Idee)
aliases: DE-Variante, EN-Variante, Akronym
tags: domain-tag-aus-whitelist
proposed_tags:
synthesis_confidence: low
action: create
extend_path:
<!--BODY-->
# {{Title}}: {{Kerncharakteristik in einem prägnanten Satz}}

Fließtext-Body, 25–40 Zeilen (HARDLIMIT 30, ohne Block-Quote-Callouts).
Jede Aussage mit `(S. N)`-Anker am Satzende. 1–3 Block-Quote-Callouts
(`> [!quote]- {author_short} {{year}}, S. N` + nächste Zeile `> „Originalzitat."`)
thematisch eingebettet im Fließtext.
<!--ANCHOR-->
page: S. 42
<!--QUOTE-->
wörtliches Zitat oder Paraphrase wie im Body
<!--END-->

**Format-Regeln (strikt):**
- Sentinels exakt wie oben: `<!--NOTE-->`, `<!--BODY-->`, `<!--ANCHOR-->`, `<!--QUOTE-->`, `<!--END-->`. ALL_CAPS, kein Whitespace im Sentinel.
- Header-Lines `key: value` einzeilig vor dem ersten `<!--BODY-->`. Lists comma-separated.
- `extend_path:` leer lassen wenn nicht zutreffend (entspricht null).
- Pro Note ein `<!--NOTE-->` Block — bei mehreren Konzepten weitere Notes anhängen, **EIN** finaler `<!--END-->` schließt den Output.
- Bei nur einem Konzept (Single-Concept-Modus, der Regelfall hier): genau ein `<!--NOTE-->` Block + `<!--END-->`.

**Wichtig**: synthesis_confidence Default ist "low" — ein einzelner Quellen-Lauf produziert primär monoquellige Notes (CERQual-Adequacy-Defizit). Der Confidence-Agent korrigiert das später ggf. auf "medium" wenn Vault-Belege gefunden werden.

Wenn ein Konzept aus der Liste nicht im Text vorkommt: keinen <!--NOTE-->-Block ausgeben. Direkt zum nächsten Konzept oder zum finalen <!--END-->. Kein Kommentar, keine Erklärung, keine Abwesenheits-Notiz — stummes Weglassen.

## Textabschnitt: {chunk_title}
{chunk_text}
"""


def _short_author(meta: dict[str, str]) -> str:
    """Kurzform für Inline-Zitate: 'Schlebbe & Greifeneder' oder 'Bertram'.
    Erkennt CrossRef-Format 'Lastname, Firstname; Lastname, Firstname' und
    einfaches 'A and B' / 'A; B'."""
    a = (meta.get("Author") or "").strip()
    if not a:
        return "Autor"
    # Autor-Separator: Semikolon oder ' und ' / ' and '
    authors = [p.strip() for p in re.split(r"\s*;\s*|\s+(?:und|and)\s+", a) if p.strip()]
    surnames = [_surname(p) for p in authors]
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]}"
    return f"{surnames[0]} et al."


def _surname(full_name: str) -> str:
    """Nachname aus '<Lastname>, <Firstname>' (CrossRef) oder '<Firstname> <Lastname>'."""
    if "," in full_name:
        return full_name.split(",", 1)[0].strip()
    tokens = full_name.split()
    return tokens[-1] if tokens else full_name


_BODY_END_OK = set('.!?…")”』」])')


def _ends_complete(body: str) -> bool:
    """Heuristik für Trunkierungs-Erkennung (B2). Body gilt als vollständig
    wenn das letzte nicht-Whitespace-Zeichen ein Satzendzeichen, Klammer oder
    Anführungszeichen ist. Mid-sentence-Cuts enden auf Buchstabe/Komma/Doppelpunkt.
    """
    s = body.rstrip()
    if not s:
        return False
    return s[-1] in _BODY_END_OK


# Schema-Konvention für proposed Tags (Bootstrap-Pfad). Strikt segmentbasiert:
# - jedes Segment startet mit Buchstabe, endet mit Buchstabe/Ziffer
# - Bindestriche nur als Worttrenner (kein `bad-`, kein `a--b`)
# - max 3 Hierarchie-Ebenen via "/"
# - Codex-Finding 2: vorherige Regex erlaubte trailing/double hyphen
_PROPOSED_TAG_RE = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?:/[a-z][a-z0-9]*(?:-[a-z0-9]+)*){0,2}$"
)


def is_valid_schema_tag(tag: str) -> bool:
    """Public Schema-Validator — auch von tag_registry-Loader genutzt damit User
    nicht versehentlich `a//b` oder `bad-` in die Registry einträgt."""
    return bool(tag and _PROPOSED_TAG_RE.match(tag))


def _validate_proposed_tags(raw: list[str], whitelist: set[str], max_count: int = 2) -> list[str]:
    """Validiert vom Extractor vorgeschlagene Tags gegen Schema-Konvention.

    Filter-Kriterien:
    - Schema-Match `_PROPOSED_TAG_RE` (kebab-case, max 3 Hierarchien)
    - NICHT bereits in Whitelist (sonst wäre es ein regulärer `tags:`-Eintrag)
    - Max max_count Tags (gegen Tag-Wildwuchs)

    Ungültige Vorschläge werden silent gedroppt — Pipeline soll nicht durch
    LLM-Tippfehler crashen. Validation-Drops sind sichtbar im Frontmatter (keiner
    der proposed-tags ist da → keine Note-Review-Status, normal verarbeitet).
    """
    if not isinstance(raw, list):
        return []
    valid: list[str] = []
    schema_valid_count = 0  # für Truncation-Warning
    for t in raw:
        if not isinstance(t, str):
            continue
        ts = t.strip().lstrip("#")
        if not ts:
            continue
        # Schema-Match strikt — Uppercase, Whitespace etc. werden NICHT durch
        # silent-lower geheilt. Wenn LLM Convention bricht, Tag droppen.
        if not _PROPOSED_TAG_RE.match(ts):
            continue
        if ts in whitelist:
            continue
        schema_valid_count += 1
        if len(valid) < max_count:
            valid.append(ts)
    # Codex-Finding 6: Truncation sichtbar machen — Hinweis auf Prompt-Drift /
    # Modell-Overproduction. Pipeline crasht nicht, aber Drop wird geloggt.
    if schema_valid_count > max_count:
        import sys
        print(f"      [proposed-tags-truncate] {schema_valid_count} valide Vorschläge "
              f"→ auf {max_count} gekürzt", file=sys.stderr)
    return valid


def _format_tag_whitelist(tags: list[str] | None,
                            source_text: str | None = None) -> str:
    """Whitelist als Bullet-Liste fürs Prompt. Leere Liste → Tags strikt
    leer lassen. Bei vorhandenem source_text wird zweigeteilt: priorisierte
    Tags (Token-Match) + übrige. Bias-Schutz: Vault-häufige aber quellfremde
    Tags landen im 'übrige'-Block und sind so nicht prompt-dominant."""
    if not tags:
        return "(keine Whitelist verfügbar — Tag-Feld leer lassen)"
    if source_text:
        from generative.agents.context_builder import score_tags_for_source
        prio, rest = score_tags_for_source(tags, source_text)
        if prio:
            blocks = [
                "**Quellnah (Token-Match mit Source) — prefer diese wenn passend:**",
                "\n".join(f"- {t}" for t in sorted(prio)),
            ]
            if rest:
                blocks.append("\n**Übrige Vault-Tags (kein lexikalischer Source-Match — meist NICHT passend):**")
                blocks.append("\n".join(f"- {t}" for t in sorted(rest)))
            return "\n".join(blocks)
    return "\n".join(f"- {t}" for t in tags)


def _clean_source_file_display(source_file: str) -> str:
    """Gibt den Dateinamen für die Prompt-`Datei:`-Zeile mit gesäubertem Autor
    zurück. Der rohe Zotero-Dateiname ('Mahmood und University of the Punjab -
    2016 - …') leakt den Affiliations-Koautor sonst trotz gesäubertem Autor-Feld
    in LLM-Sekundärzitate ('zit. n. Mahmood & Punjab') — das ' und ' liest sich
    als Zwei-Autoren-Trenner. Drei­ter Geschwister-Kanal der Issue-41/PR-71-Klasse.
    Nicht-parsbare Namen bleiben unverändert."""
    from generative.pipeline.vault_writer import _parse_filename_fallback
    fb = _parse_filename_fallback(source_file)
    author = fb.get("Author")
    if not author:
        return source_file
    ext = Path(source_file).suffix
    title = fb.get("Title", "")
    year = fb.get("Year")
    core = f"{author} - {year} - {title}" if year else f"{author} - {title}"
    return f"{core}{ext}"


def _format_source_meta(meta: dict[str, str], source_file: str) -> str:
    parts = []
    if meta.get("Author"): parts.append(f"Autor: {meta['Author']}")
    if meta.get("Title"):  parts.append(f"Titel: {meta['Title']}")
    if meta.get("Year"):   parts.append(f"Jahr: {meta['Year']}")
    parts.append(f"Datei: {_clean_source_file_display(source_file)}")
    return "\n".join(f"- {p}" for p in parts)


def _format_background_block(background_context: list[str] | None) -> str:
    """Optionaler Prompt-Block für stilles Kontext-Wissen. Leer wenn keine Claims.

    Hintergrundwissen hilft dem Extractor Konzepte einzuordnen — es darf NICHT
    sichtbar in den Note-Body. Keine [Trainingswissen]-Marker, keine Erwähnung
    dass dieses Wissen aus Training stammt. Nur Quelltextaussagen mit (S. N)-Ankern.
    """
    if not background_context:
        return ""
    claims = "\n".join(f"- {c}" for c in background_context)
    return (
        "## Kontext-Wissen (nur zur Einordnung — NICHT in den Body schreiben)\n"
        "Folgende Hintergrundinformationen helfen dir das Konzept einzuordnen.\n"
        "WICHTIG: Schreibe NICHTS davon in den Note-Body. Kein '[Trainingswissen]'-Marker.\n"
        "Der Body enthält NUR Aussagen die direkt aus dem Quelltext stammen mit (S. N)-Anker.\n\n"
        f"{claims}\n\n"
    )


def _format_related_mentions(mentions: list[str] | None) -> str:
    """Kontext-Block für sekundär erwähnte Konzepte aus der Quelle.

    Der Extractor emittiert nur die Konzept-Namen — kein Link-Format.
    Der vault_writer entscheidet später ob ein Wikilink gesetzt wird
    (abhängig von OUTPUT_FORMAT und ob das Konzept im Vault existiert).
    """
    if not mentions:
        return ""
    items = "\n".join(f"- {m}" for m in mentions[:10])
    return (
        "## In dieser Quelle zitierte Konzepte (nur zur Orientierung)\n"
        "Diese Konzepte wurden in der Quelle erwähnt, aber nicht primär behandelt.\n"
        "Falls thematisch passend: kurz im Fließtext erwähnen (nur als Begriff, kein Link).\n"
        f"{items}\n\n"
    )


async def run_per_concept(concept, concept_text: str,
                          existing_concepts: dict[str, str],
                          source_meta: dict[str, str] | None = None,
                          source_file: str = "",
                          revision_hint: str | None = None,
                          tag_whitelist: list[str] | None = None,
                          background_context: list[str] | None = None,
                          related_mentions: list[str] | None = None,
                          current_draft_body: str | None = None) -> AtomicNoteDraft | None:
    """Extrahiere genau eine Note für ein konkretes Konzept aus den relevanten
    Textstellen (gesammelt via pdf_chunker.concept_text_window).

    Returns None wenn Modell keinen verwertbaren Output liefert (Konzept zu schwach im Text).

    Bei Self-Refine (Milestone 3.6): revision_hint vom Critic wird dem Prompt vorangestellt
    als zusätzliches Constraint. Cache-Miss garantiert (anderer Prompt).
    """
    if not concept_text.strip():
        return None

    source_meta = source_meta or {}
    _etoks = set(concept.title.lower().split())
    _esorted = sorted(existing_concepts, key=lambda k: len(_etoks & set(k.lower().split())), reverse=True)
    existing_str = "\n".join(f"- {k}" for k in _esorted[:75])

    # Prompt: nur dieses eine Konzept fokussieren — Modell soll die ganze Textstelle dafür nutzen
    concepts_str = f"- {concept.title} (Priorität: {concept.priority}, action: {concept.action})"

    refine_block = ""
    if revision_hint and current_draft_body:
        # Bug #1: gezieltes Überarbeiten statt Neugenerierung — alter Body + Quellentext + Hint
        refine_block = (
            "## Gezielte Überarbeitung (höchste Priorität — kein Neuschreiben)\n\n"
            "Bestehende Note (Ausgangspunkt — behalte alles Korrekte):\n"
            "---\n"
            f"{current_draft_body}\n"
            "---\n\n"
            "Critic-Feedback (nur diese Punkte ändern):\n"
            f"{revision_hint}\n\n"
            "Regeln:\n"
            "- Ändere MINIMAL: nur was der Critic bemängelt\n"
            "- Behalte alle korrekten Aussagen, Zitate, Anker, Aliases, Tags\n"
            "- Füge nichts hinzu das nicht im Quellentext unten steht\n"
            "- Prüfe jeden behaltenen Satz gegen den Quellentext\n"
            "- Wenn eine Aussage im alten Body keine Entsprechung im Quellentext hat und der Critic sie nicht bemängelt hat: trotzdem entfernen\n\n"
        )
    elif revision_hint:
        refine_block = (
            "## Revision-Hinweis (Self-Refine — vom Critic, höchste Priorität)\n"
            f"{revision_hint}\n"
            "Adressiere diesen Punkt direkt in der neuen Version.\n\n"
        )

    prompt = refine_block + _PROMPT.format(
        source_meta=_format_source_meta(source_meta, source_file),
        author_short=_short_author(source_meta),
        concepts=concepts_str,
        existing=existing_str or "(noch keine)",
        background_block=_format_background_block(background_context),
        related_mentions_block=_format_related_mentions(related_mentions),
        tag_whitelist=_format_tag_whitelist(tag_whitelist, source_text=concept_text),
        chunk_title=concept.title,
        chunk_text=concept_text,  # pdf_chunker.concept_text_window liefert bereits gerankte Top-Fenster (Option D, max_chars=8000)
    )

    raw = await call_claude_async(prompt, model=MODEL_EXTRACTOR, agent="extractor")

    items, parse_warnings = parse_extractor_output(raw)
    if parse_warnings:
        import sys
        for w in parse_warnings:
            print(f"      [extractor-warn] '{concept.title}': {w}", file=sys.stderr)

    if not items:
        import sys
        print(f"      [extractor-empty] '{concept.title}' kein verwertbarer Output (raw[:120]={raw[:120]!r})",
              file=sys.stderr)
        return None

    item = items[0]

    # B2-Fix: Trunkierungs-Retry. Output-Token-Cap von `claude -p` ist nicht
    # steuerbar — wenn body mid-sentence abgebrochen wird, Re-Call mit Hint
    # „kürzer schreiben". Empirisch beobachtet: Body endet auf Buchstabe/Komma/
    # Halbsatz statt Satzendzeichen. Ohne Retry → Critic-Quellen-Hard-Gate-Fail.
    body = (item.get("body") or "").rstrip()
    if body and not _ends_complete(body) and revision_hint is None:
        # nur außerhalb des Self-Refine-Loops, sonst Retry-Kaskade möglich
        tail = body[-30:]
        trunc_hint = (
            "Vorheriger Output endete unvollständig (Body brach mitten im Satz "
            f"ab bei: '...{tail}'). Schreibe die Note kürzer, sodass der gesamte "
            "Body in der Antwort Platz hat: maximal **25 Zeilen** Body, alle "
            "Sätze vollständig mit Satzendzeichen. Empirie-Phase auf 1–2 Sätze "
            "kürzen wenn Platzdruck. Definition + Substanz haben Vorrang."
        )
        retry_prompt = (
            "## Trunkierungs-Hinweis (höchste Priorität)\n"
            f"{trunc_hint}\n\n"
        ) + _PROMPT.format(
            source_meta=_format_source_meta(source_meta, source_file),
            author_short=_short_author(source_meta),
            concepts=concepts_str,
            existing=existing_str or "(noch keine)",
            background_block=_format_background_block(background_context),
            related_mentions_block=_format_related_mentions(related_mentions),
            tag_whitelist=_format_tag_whitelist(tag_whitelist, source_text=concept_text),
            chunk_title=concept.title,
            chunk_text=concept_text[:8000],
        )
        raw2 = await call_claude_async(retry_prompt, model=MODEL_EXTRACTOR, agent="extractor")
        items2, _ = parse_extractor_output(raw2)
        if items2:
            cand = items2[0]
            cand_body = (cand.get("body") or "").rstrip()
            # Übernahme nur wenn Retry sauber endet — sonst Original behalten
            if cand_body and _ends_complete(cand_body):
                item = cand
    anchors = [
        TextAnchor(quote=a.get("quote", ""), page=a.get("page"))
        for a in item.get("source_anchors", [])
    ]
    proposed = _validate_proposed_tags(item.get("proposed_tags", []),
                                         whitelist=set(tag_whitelist or []))
    return AtomicNoteDraft(
        title=item.get("title", concept.title),
        body=item.get("body", ""),
        source_anchors=anchors,
        related=[],
        tags=item.get("tags", []),
        aliases=item.get("aliases", []),
        synthesis_confidence=item.get("synthesis_confidence", "low"),
        action=item.get("action", concept.action),
        extend_path=item.get("extend_path") or concept.extend_path,
        proposed_tags=proposed,
        tag_review_status="needs-review" if proposed else None,
    )


async def run(chunk_title: str, chunk_text: str,
              concept_plan: ConceptPlan,
              existing_concepts: dict[str, str],
              source_meta: dict[str, str] | None = None,
              source_file: str = "",
              tag_whitelist: list[str] | None = None) -> list[AtomicNoteDraft]:

    source_meta = source_meta or {}
    target_concepts = [
        c for c in concept_plan.concepts
        if c.action in ("create", "extend") and c.chapter.lower() in chunk_title.lower()
        or c.priority == "high"  # High-Priority-Konzepte in jedem Chunk suchen
    ]

    if not target_concepts:
        return []

    concepts_str = "\n".join(
        f"- {c.title} (Priorität: {c.priority}, action: {c.action})"
        for c in target_concepts
    )
    _etoks2 = set(chunk_title.lower().split())
    _esorted2 = sorted(existing_concepts, key=lambda k: len(_etoks2 & set(k.lower().split())), reverse=True)
    existing_str = "\n".join(f"- {k}" for k in _esorted2[:75])

    prompt = _PROMPT.format(
        source_meta=_format_source_meta(source_meta, source_file),
        author_short=_short_author(source_meta),
        concepts=concepts_str,
        existing=existing_str or "(noch keine)",
        background_block="",           # run() hat kein background_context — legacy-Pfad
        related_mentions_block="",     # run() hat kein related_mentions — legacy-Pfad
        tag_whitelist=_format_tag_whitelist(tag_whitelist, source_text=chunk_text),
        chunk_title=chunk_title,
        chunk_text=chunk_text[:8000],
    )

    raw = await call_claude_async(prompt, model=MODEL_EXTRACTOR, agent="extractor")

    items, parse_warnings = parse_extractor_output(raw)
    if parse_warnings:
        import sys
        for w in parse_warnings:
            print(f"      [extractor-warn] {w}", file=sys.stderr)

    drafts: list[AtomicNoteDraft] = []
    for item in items:
        anchors = [
            TextAnchor(quote=a.get("quote", ""), page=a.get("page"))
            for a in item.get("source_anchors", [])
        ]
        drafts.append(AtomicNoteDraft(
            title=item.get("title", ""),
            body=item.get("body", ""),
            source_anchors=anchors,
            related=[],  # wird vom Cross-Reference-Agent gefüllt
            tags=item.get("tags", []),
            aliases=item.get("aliases", []),
            synthesis_confidence=item.get("synthesis_confidence", "low"),
            action=item.get("action", "create"),
            extend_path=item.get("extend_path"),
        ))
        proposed = _validate_proposed_tags(
            item.get("proposed_tags", []), whitelist=set(tag_whitelist or []))
        drafts[-1].proposed_tags = proposed
        if proposed:
            drafts[-1].tag_review_status = "needs-review"

    return drafts
