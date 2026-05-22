"""Verifier-Agent: prüft jede Aussage in der Draft-Note gegen den Originaltext."""
from __future__ import annotations
import re

from rapidfuzz import fuzz

from agents.base import call_claude, trace_event
from agents.structured_output import parse_verifier_output
import config as _config
from config import MODEL_VERIFIER, SEMANTIC_PREPASS_THRESHOLD
from schemas.atomic_note import AtomicNoteDraft, TextAnchor

FUZZY_THRESHOLD = 85
# Pre-Pass vor LLM-Call: hoch-konfidente deterministische Auflösung. Cutoff 98 (statt
# 85 wie im Post-LLM-Fallback), Mindestlänge 30 Zeichen — kurze/generische Phrasen
# matchen sonst trivial. Cross-Model-Konsens Codex/Gemini 2026-05-11.
FUZZY_PREPASS_THRESHOLD = 98
MIN_QUOTE_LEN_FOR_PREPASS = 30
from pipeline.anchor_patterns import PAGE_MARKER_RE, PAGE_ANCHOR_NUMS_RE as _NEAR_PAGE_RE

# Hebel #3: Body-Anker-Sync. Direktzitate im Body stehen typischerweise als
# „..." gefolgt von einer Seitenangabe `(S. N)`. Opening: U+201E (vom Extractor
# erzwungen). Closing: in der Praxis U+201C, U+201D oder ASCII U+0022 — alle
# zulassen, weil der Extractor-Output trotz Prompt-Verbot ASCII enthält.
_BODY_QUOTE_RE = re.compile("„([^„“”\"]{4,200})[“”\"]")

_PROMPT = """Du prüfst ob Aussagen in einer Atomic Note tatsächlich im Originaltext stehen.

Der Originaltext ist mit `[S. N]`-Markern an Seitenanfängen versehen. Beim Verifizieren eines Ankers musst du die korrekte Seitenzahl aus dem Marker ablesen, nicht raten.

## Deine Aufgabe
Pro Source-Anchor:
- Suche das Zitat/die Paraphrase im Originaltext
- Wenn gefunden: `verified: true`, setze `page` auf den letzten `[S. N]`-Marker VOR der Fundstelle (z.B. `"S. 5"`)
- Wenn nicht gefunden: `verified: false`, page bleibt null

Ändere NICHTS am Body der Note. Nur Anker prüfen.

## Output — NUR dieses Format, kein erklärender Text, KEIN JSON:

all_verified: true
<!--ANCHOR-->
page: S. 5
verified: true
<!--QUOTE-->
exakt das Quote-Zitat aus den Anchors zur Prüfung
<!--ANCHOR-->
page:
verified: false
<!--QUOTE-->
nicht gefundenes Zitat
<!--END-->

**Format-Regeln**: Top-Level Header `all_verified: true|false` einzeilig vor erstem `<!--ANCHOR-->`. Pro geprüftem Anker ein `<!--ANCHOR-->`-Block mit `page:` (oder leer wenn nicht gefunden) und `verified: true|false`. Quote als Heredoc nach `<!--QUOTE-->`.

## Draft-Note Titel: {title}

## Source Anchors zu prüfen:
{anchors}

## Originaltext (mit Seiten-Markern):
{chunk_text}
"""


def sync_anchors_from_body(draft: AtomicNoteDraft) -> AtomicNoteDraft:
    """Hebel #3: Body-Anker-Sync.

    Englische Original-Phrasen erscheinen oft im Body als `„Quote" ... (S. N)`
    aber fehlen in `source_anchors` oder stehen dort ohne Seite. Critic flaggt
    dann fälschlich Quellen-Test-Fail. Diese Stage scannt den Body, findet
    Quote+Page-Paare und (a) ergänzt fehlende Anker, (b) füllt Pages bei
    bestehenden Ankern mit Quote-Match nach.

    Konservativ: Nur wenn `(S. N)` innerhalb von 120 Zeichen NACH dem Quote
    folgt (typische Inline-Belegform). Erste passende Seite gewinnt.
    """
    body = draft.body
    existing_quotes = {a.quote: a for a in draft.source_anchors}
    added = 0
    filled = 0

    for qm in _BODY_QUOTE_RE.finditer(body):
        quote = qm.group(1).strip()
        if not quote:
            continue
        # Look-ahead-Fenster für (S. N)
        window = body[qm.end(): qm.end() + 120]
        pm = _NEAR_PAGE_RE.search(window)
        if not pm:
            continue
        first_page = pm.group(1).split(",")[0].strip()
        page_str = f"S. {first_page}"

        existing = existing_quotes.get(quote)
        if existing is None:
            new_anchor = TextAnchor(quote=quote, page=page_str, fuzzy_page=None)
            draft.source_anchors.append(new_anchor)
            existing_quotes[quote] = new_anchor
            added += 1
        elif not existing.page and not existing.fuzzy_page:
            existing.fuzzy_page = page_str
            filled += 1

    if added or filled:
        import sys
        print(f"      [anker-sync] {added} ergänzt, {filled} Page nachgetragen",
              file=sys.stderr)
    return draft


def _log_anchor_stats(title: str, total_in: int, final_anchors: list) -> None:
    confirmed = sum(1 for a in final_anchors if a.page or a.fuzzy_page)
    trace_event("verifier", "anchor_stats", {
        "title": title,
        "total_in": total_in,
        "confirmed": confirmed,
        "confirmation_rate": round(confirmed / total_in, 3) if total_in > 0 else 0.0,
    })


def run(draft: AtomicNoteDraft, chunk_text: str) -> AtomicNoteDraft:
    _total_in = len(draft.source_anchors)
    try:
        return _run_inner(draft, chunk_text)
    finally:
        _log_anchor_stats(draft.title, _total_in, draft.source_anchors)


def _run_inner(draft: AtomicNoteDraft, chunk_text: str) -> AtomicNoteDraft:
    if not draft.source_anchors:
        # Auch wenn bisher keine Anker da sind: Body-Sync kann welche aus
        # `„..." (S. N)`-Zitaten ableiten.
        sync_anchors_from_body(draft)
        if not draft.source_anchors:
            return draft

    # Deterministischer Pre-Pass: Anker mit hohem Fuzzy-Score sofort auflösen,
    # nur unresolved gehen an LLM. Quality-Sicherheit:
    #   - Exact-Substring (normalisiert) → page (cache-stabiler Critic-Input)
    #   - Fuzzy ≥98 + Mindestlänge → fuzzy_page (Critic sieht es nicht als verifiziert)
    # LLM bleibt Safety-Net für Edge-Cases. Wenn alles pre-resolved → kein LLM-Call.
    original_count = len(draft.source_anchors)
    pre_resolved: list[TextAnchor] = []
    unresolved: list[TextAnchor] = []
    for orig in draft.source_anchors:
        quote_clean = orig.quote.strip().strip('„"\'""')
        page: str | None = None
        is_exact = False
        if len(quote_clean) >= MIN_QUOTE_LEN_FOR_PREPASS:
            if quote_clean in chunk_text:
                # Exact-Substring → autoritative Page-Ableitung
                pos = chunk_text.find(quote_clean)
                last_marker = None
                for m in PAGE_MARKER_RE.finditer(chunk_text):
                    if m.start() > pos:
                        break
                    last_marker = m.group(1)
                if last_marker:
                    page = f"S. {last_marker}"
                    is_exact = True
            if page is None:
                page = _fuzzy_find_page(
                    orig.quote, chunk_text, threshold=FUZZY_PREPASS_THRESHOLD
                )
        if page:
            pre_resolved.append(TextAnchor(
                quote=orig.quote,
                page=page if is_exact else None,
                fuzzy_page=None if is_exact else page,
            ))
        else:
            unresolved.append(orig)

    if not unresolved:
        import sys
        print(
            f"      [verifier-prepass] '{draft.title}' "
            f"{len(pre_resolved)}/{original_count} fuzzy≥{FUZZY_PREPASS_THRESHOLD} — LLM skip",
            file=sys.stderr,
        )
        draft.source_anchors = pre_resolved
        sync_anchors_from_body(draft)
        return draft

    if pre_resolved:
        import sys
        print(
            f"      [verifier-prepass] '{draft.title}' "
            f"{len(pre_resolved)}/{original_count} fuzzy≥{FUZZY_PREPASS_THRESHOLD}, "
            f"{len(unresolved)} weiter",
            file=sys.stderr,
        )

    # Tier-3: semantischer Pre-Pass via sentence-transformers.
    # Paraphrasen die rapidfuzz≥98 verfehlt (Wortwahl-Varianten, DE/EN-Mix).
    # Chunk-Embeddings einmal berechnen und an alle Anker-Checks weitergeben.
    cached_page_sections = _build_page_sections(chunk_text)
    semantic_resolved: list[TextAnchor] = []
    remaining: list[TextAnchor] = []
    for anc in unresolved:
        sp = _semantic_find_page(anc.quote, chunk_text,
                                 cached_sections=cached_page_sections)
        if sp:
            semantic_resolved.append(TextAnchor(quote=anc.quote, page=None, fuzzy_page=sp))
        else:
            remaining.append(anc)
    if semantic_resolved:
        import sys
        print(
            f"      [verifier-semantic] '{draft.title}' "
            f"{len(semantic_resolved)}/{len(unresolved)} semantic≥{SEMANTIC_PREPASS_THRESHOLD:.2f} — "
            f"{len(remaining)} an LLM",
            file=sys.stderr,
        )
    pre_resolved = pre_resolved + semantic_resolved
    unresolved = remaining

    if not unresolved:
        import sys
        print(
            f"      [verifier-prepass+semantic] '{draft.title}' "
            f"{original_count}/{original_count} Anker ohne LLM",
            file=sys.stderr,
        )
        draft.source_anchors = pre_resolved
        sync_anchors_from_body(draft)
        return draft

    # no-LLM-Modus: verbleibende unresolved Anker unverifiziert lassen.
    if not _config.ENABLE_LLM:
        import sys
        print(
            f"      [verifier-nollm] '{draft.title}' "
            f"{len(unresolved)} Anker unverifiziert (no-LLM-Modus)",
            file=sys.stderr,
        )
        unresolved = [
            TextAnchor(quote=anc.quote, page=None, fuzzy_page=None)
            for anc in unresolved
        ]
        draft.source_anchors = pre_resolved + unresolved
        draft.quality_flags.append(
            f"ℹ️ Verifier: {len(unresolved)} Anker unverifiziert (no-LLM-Modus)"
        )
        sync_anchors_from_body(draft)
        return draft

    anchors_str = "\n".join(
        f"- \"{a.quote}\" (Seite: {a.page or 'unbekannt'})"
        for a in unresolved
    )

    prompt = _PROMPT.format(
        title=draft.title,
        anchors=anchors_str,
        chunk_text=chunk_text[:6000],
    )

    try:
        raw = call_claude(prompt, model=MODEL_VERIFIER, agent="verifier")
        data, parse_warnings = parse_verifier_output(raw)
        if parse_warnings:
            import sys
            for w in parse_warnings:
                print(f"      [verifier-warn] '{draft.title}': {w}", file=sys.stderr)
    except RuntimeError as e:
        import sys
        print(f"      [verifier-fail] '{draft.title}' LLM-Call fehlgeschlagen: {str(e)[:80]}",
              file=sys.stderr)
        # Pre-resolved Anker behalten, unresolved bleiben mit Original-Page (oder None)
        draft.source_anchors = pre_resolved + unresolved
        draft.quality_flags.append("⚠️ Verifier nicht ausgeführt — Anker unverifiziert")
        return draft

    # Heredoc-Parser raised nicht bei malformed Output (anders als das alte
    # parse_json), sondern returnt anchors=[]. Wenn der Draft Anker hatte und
    # der Parser KEINE liefert, ist das ein Parse-Fail des LLM-Outputs — Rebuild
    # würde sonst alle Original-Anker stumm löschen (v16-Silent-Destructive-Pattern).
    if unresolved and not data["anchors"]:
        import sys
        print(f"      [verifier-fail] '{draft.title}' LLM-Output ohne ANCHOR-Block "
              f"(raw[:120]={raw[:120]!r}) — Anker bleiben unverifiziert",
              file=sys.stderr)
        draft.source_anchors = pre_resolved + unresolved
        draft.quality_flags.append("⚠️ Verifier-Output unparsbar — Anker unverifiziert")
        return draft

    # F8: Anker-Übernahme mit Fuzzy-Fallback. Wenn LLM-Verifier exact-match
    # findet, übernehmen wir mit page (Cache-stabiler Critic-Input). Wenn LLM
    # nicht findet, aber rapidfuzz partial_ratio ≥ FUZZY_THRESHOLD greift,
    # übernehmen wir den Anker mit page=None und fuzzy_page=<match>. Renderer
    # nutzt `page or fuzzy_page` für den Quellen-Block — Critic sieht nur page,
    # bleibt also cache-stabil gegenüber dem LLM-Verdikt.
    llm_results = {a.get("quote", ""): a for a in data.get("anchors", [])}
    rebuilt: list[TextAnchor] = list(pre_resolved)
    for orig in unresolved:
        llm = llm_results.get(orig.quote, {})
        if llm.get("verified", False):
            rebuilt.append(TextAnchor(
                quote=orig.quote,
                page=llm.get("page"),
                fuzzy_page=None,
            ))
            continue
        fp = _fuzzy_find_page(orig.quote, chunk_text)
        if fp:
            rebuilt.append(TextAnchor(
                quote=orig.quote,
                page=None,
                fuzzy_page=fp,
            ))
    draft.source_anchors = rebuilt

    # 50%-Schwelle bezieht sich auf alle Anker mit irgendeiner Page-Bestätigung
    # (Pre-Pass ODER LLM-exact ODER fuzzy-Post). Schärft die ursprüngliche Logik.
    bestätigt = sum(1 for a in rebuilt if a.page or a.fuzzy_page)
    if original_count > 0 and bestätigt < original_count / 2:
        draft.synthesis_confidence = "low"
        draft.quality_flags.append("⚠️ weniger als 50% der Text-Anker verifiziert")

    # Hebel #3: Body-Sync nach LLM-Rebuild — nimmt englische Original-Phrasen
    # `„..." (S. N)` aus dem Body als zusätzliche Anker mit Page auf, ergänzt
    # Pages bei bestehenden Ankern ohne Page.
    sync_anchors_from_body(draft)

    return draft


def _build_page_sections(chunk_text: str) -> list[tuple[str, list[str]]] | None:
    """Splittet chunk_text nach [S. N]-Markern und embedded Sätze pro Seite.

    Gibt [(page_num, sent_embeddings)] zurück oder None wenn Modell nicht verfügbar.
    Einmalig pro Note aufrufbar — Ergebnis wird an alle _semantic_find_page-Calls
    weitergereicht (kein redundantes Re-Embedding).

    Kurzschluss wenn Modell noch nicht geladen: vermeidet Cold-Start durch CrossRef/Critic.
    In Tests bleibt _MODEL=None → kein 3s-Download im Unit-Test.
    """
    try:
        from pipeline import embeddings as _emb_mod
        if _emb_mod._MODEL is None:
            return None  # Modell nicht geladen → Tier-3 überspringen
        from pipeline.embeddings import _sentences as _split_sents, _model
        model = _model()
    except Exception:
        return None

    result: list[tuple[str, object]] = []  # (page_num, sent_embs_array)
    prev_end = 0
    prev_page: str | None = None
    for m in PAGE_MARKER_RE.finditer(chunk_text):
        if prev_page is not None:
            sents = _split_sents(chunk_text[prev_end:m.start()])
            if sents:
                embs = model.encode(sents, show_progress_bar=False, normalize_embeddings=True)
                result.append((prev_page, embs))
        prev_page = m.group(1)
        prev_end = m.end()
    if prev_page is not None:
        sents = _split_sents(chunk_text[prev_end:])
        if sents:
            embs = model.encode(sents, show_progress_bar=False, normalize_embeddings=True)
            result.append((prev_page, embs))
    return result or None


def _semantic_find_page(quote: str, chunk_text: str,
                        threshold: float = SEMANTIC_PREPASS_THRESHOLD,
                        cached_sections: list | None = None) -> str | None:
    """Tier-3 Pre-Pass: semantisches Satz-Matching via sentence-transformers.

    cached_sections: vorab berechnete [(page_num, sent_embs)] aus _build_page_sections().
    Wenn None, werden Embeddings hier berechnet (Fallback für Einzelaufruf).
    """
    quote_clean = quote.strip().strip('„"\'""')
    if not quote_clean or len(quote_clean) < 20:
        return None

    try:
        from pipeline.embeddings import embed_title
        page_sections = cached_sections or _build_page_sections(chunk_text)
    except Exception:
        return None

    if not page_sections:
        return None

    quote_emb = embed_title(quote_clean)

    best_score = -1.0
    best_page: str | None = None
    for page_num, sent_embs in page_sections:
        page_best = float(sent_embs.dot(quote_emb).max())
        if page_best > best_score:
            best_score = page_best
            best_page = page_num

    if best_score < threshold or best_page is None:
        return None
    return f"S. {best_page}"


def _fuzzy_find_page(quote: str, text: str, threshold: int = FUZZY_THRESHOLD) -> str | None:
    """Sucht das Quote per partial_ratio im Volltext. Bei Treffer ≥ threshold:
    finde die Position des Best-Matches und gib den letzten [S. N]-Marker davor zurück.
    """
    if not quote or len(quote) < 15:
        return None
    # Cleanup für robusteres Matching
    quote_clean = quote.strip().strip('„"\'""')
    # Token-Set-Score auf Wort-Ebene wäre robuster gegen Reihenfolge; für Direktzitate
    # ist partial_ratio (Substring-Match mit Edit-Distance) die richtige Wahl
    score = fuzz.partial_ratio(quote_clean, text)
    if score < threshold:
        return None
    # Beste Position via partial_ratio_alignment
    align = fuzz.partial_ratio_alignment(quote_clean, text)
    if align is None:
        return None
    pos = align.dest_start
    # Letzten [S. N]-Marker VOR pos finden
    last_page = None
    for m in PAGE_MARKER_RE.finditer(text):
        if m.start() > pos:
            break
        last_page = m.group(1)
    if last_page is None:
        return None
    return f"S. {last_page}"
