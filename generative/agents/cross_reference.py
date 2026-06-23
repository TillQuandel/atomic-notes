"""Cross-Reference-Agent: prüft Aussagen gegen existierende Vault-Notes."""
from __future__ import annotations
import re
from pathlib import Path

from generative.agents.base import call_claude
from generative.agents.structured_output import parse_cross_reference_output
from generative import config as _config
from generative.config import VAULT, MODEL_CROSS_REF, ENABLE_NLI_VALIDATION, NLI_MODEL_NAME, NLI_CONTRADICTION_THRESHOLD
from generative.schemas.atomic_note import AtomicNoteDraft

# Mindest-Anzahl `related`-Wikilinks für eine Schema-konforme Note (siehe Schema-Konzept §5)
MIN_RELATED = 2
MAX_RELATED = 4


def _clean_wikilink(s: str) -> str:
    """Reiner innerer Titel eines (evtl. verschachtelten) Wikilink-Strings, ohne
    eckige Klammern: '[[[[A]]]]' -> 'A', '[[A]]' -> 'A', 'A' -> 'A', '[[A|x]]' -> 'A|x'.

    Härtet gegen vom LLM gelieferte oder doppelt gewrappte Strings, die sonst zu
    '[[[[..]]]]' führen (Muster wie vault_writer.rewrite_merged_related_links)."""
    return (s or "").strip().strip("[]").strip()


def _resolve_vault_path(dup_path: str, existing_concepts: dict[str, str] | None) -> Path | None:
    """Löst einen vom LLM gelieferten duplicate_path auf eine reale Vault-Datei auf.

    Kandidaten werden dem LLM als `Path(p).stem` präsentiert (siehe Prompt-Bau), daher
    kommt dup_path meist als Datei-Stem zurück (z.B. 'ba-lit-ebner-gegenfurtner-2019'),
    gelegentlich als Titel. Beide Wege werden gegen existing_concepts (Titel/Alias→Pfad)
    aufgelöst. None, wenn kein Vault-Treffer (z.B. Intra-Run-Sibling — bisheriges Verhalten).
    """
    from generative.agents.context_builder import resolve_vault_relpath
    rel = resolve_vault_relpath(dup_path, existing_concepts)
    return VAULT / rel if rel else None


def _dup_target_eligible(dup_path: str, existing_concepts: dict[str, str] | None) -> bool:
    """False, wenn dup_path auf eine Note zeigt, deren Typ per Vault-Design mit
    Konzept-Notes KOEXISTIERT (literature/moc/merge-stub) → kein echtes Duplikat.

    Verhindert den False-Positive „Konzept-Note ist Dup ihrer eigenen Lit-Note"
    (Ebner-Run 2026-06-23). Unauflösbarer/nicht-Vault dup_path → True (bisheriges
    Verhalten bleibt; Intra-Run-Siblings regelt resolve_sibling_dups).
    """
    target = _resolve_vault_path(dup_path, existing_concepts)
    if target is None:
        return True
    from generative.agents.context_builder import is_dedup_eligible
    return is_dedup_eligible(target)

# Lazy-loaded NLI CrossEncoder — wird nur bei ENABLE_NLI_VALIDATION=1 geladen
_nli_encoder = None
_nli_lock = __import__("threading").Lock()  # Thread-Safety bei parallelem Stage-6-Load

# BM25-Index-Cache: pro Run wird existing_concepts einmal aufgebaut und nicht verändert.
# id(existing_concepts) als Key — gleicher Dict-Pointer → Cache-Hit.
_bm25_cache: dict = {}  # {id: (keys, paths, bm25_instance)}


def _nli_validate_contradictions(
    note_body: str,
    vault_excerpts: list[str],
) -> bool:
    """Prüft ob NLI-DeBERTa einen Widerspruch zwischen der neuen Note und den
    Vault-Kandidaten-Excerpts bestätigt. AND-Kombination mit Haiku:
    gibt True zurück wenn DeBERTa contradiction_score ≥ NLI_CONTRADICTION_THRESHOLD
    für mindestens ein Vault-Excerpt. Bei Fehler: True (Haiku-Urteil beibehalten).

    Gemma-4-Ansatz: Note-Excerpt × Vault-Excerpt — kein all-pairs Satz-Brute-Force.
    Cosine-Prefilter (gpt-oss-120b): nur topisch ähnliche Excerpts per NLI prüfen.
    """
    global _nli_encoder
    try:
        from sentence_transformers import CrossEncoder
        from generative.pipeline.embeddings import embed_body, cosine as cos_sim
        if _nli_encoder is None:
            with _nli_lock:
                if _nli_encoder is None:  # Double-Checked Locking
                    import sys
                    print(f"      [nli] Lade {NLI_MODEL_NAME} (einmalig ~70MB)…", file=sys.stderr)
                    _nli_encoder = CrossEncoder(NLI_MODEL_NAME)

        note_short = note_body[:600].strip()
        for vault_exc in vault_excerpts:
            vault_short = vault_exc[:600].strip()
            if not vault_short:
                continue
            # Cosine-Prefilter (gpt-oss-120b): nur bei topischer Ähnlichkeit NLI aufrufen
            try:
                sim = cos_sim(embed_body(note_short), embed_body(vault_short))
                if sim < 0.3:
                    continue
            except Exception:
                pass  # Prefilter nicht kritisch — NLI trotzdem laufen lassen

            scores = _nli_encoder.predict(
                [(note_short, vault_short)], apply_softmax=True
            )
            # DeBERTa NLI Labels: [contradiction, entailment, neutral]
            contradiction_score = float(scores[0][0])
            import sys
            print(
                f"      [nli] contradiction={contradiction_score:.2f} "
                f"(threshold={NLI_CONTRADICTION_THRESHOLD})",
                file=sys.stderr,
            )
            if contradiction_score >= NLI_CONTRADICTION_THRESHOLD:
                return True
        return False
    except Exception as e:
        import sys
        print(f"      [nli] Fehler bei Validation: {e}", file=sys.stderr)
        return True  # Fallback: Haiku-Urteil beibehalten

# Stoppwörter (DE+EN) für Tokenize-Match — verhindern False-Positives auf Füllwörtern
_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "aber", "doch", "von", "vom", "zum", "zur", "als", "wie", "für", "mit", "bei",
    "auf", "in", "im", "an", "am", "zu", "über", "unter", "vor", "nach", "ist", "sind", "war",
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "as", "from", "that", "this", "it",
}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Lower-case Content-Tokens, ohne Stoppwörter und kurze (<3) Tokens."""
    return {t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
            if len(t) >= 3 and t not in _STOPWORDS}


def _matches(query_tokens: set[str], concept_key: str) -> bool:
    """≥1 Content-Token-Overlap mit Wortgrenze (kein Substring-Bleed)."""
    return bool(query_tokens & _tokens(concept_key))

_PROMPT = """Du prüfst eine neue Atomic Note gegen verwandte Vault-Notes — drei Aufgaben:

1. **Widersprüche**: Gibt es inhaltliche Konflikte zwischen neuer Note und bestehenden? (max 1 Satz pro Widerspruch). KEIN Widerspruch ist: (a) eine Note erwähnt ein Konzept kurz, eine andere vertieft es (Atomic-Notes-Hierarchie); (b) zwei Notes betrachten dasselbe Thema aus verschiedenen Perspektiven ohne sich zu widersprechen; (c) eine Note ist allgemein, die andere ist spezifischer Sub-Aspekt. Ein echter Widerspruch verlangt unvereinbare Aussagen über denselben Sachverhalt (z.B. „X verursacht Y" vs. „X verursacht Y nicht").
2. **Duplikat-Risiko**: Gibt es starke Überschneidungen mit einer bestehenden Note? (none/low/high)
3. **`related`-Links**: Wähle {min_related}–{max_related} echte Wikilinks zu **inhaltlich verwandten** existierenden Notes. Format pro Link: `[[<Note-Titel>]]` mit GENAU einem Klammer-Paar — Titel wie in den `###`-Überschriften unten, OHNE zusätzliche eckige Klammern (die Headers oben enthalten den reinen Titel). Kein Themen-Drift, keine reinen Stichwort-Treffer. Wenn weniger als {min_related} sinnvolle Links existieren: keine RELATED-Sektion, Note geht zur manuellen Review.

## Neue Note: {title}
{body}

## Existierende Vault-Notes (Auszüge — Kandidaten für `related`):
{existing_excerpts}

## Output — NUR dieses Format, kein erklärender Text, KEIN JSON:

duplicate_risk: none
duplicate_path:
<!--CONTRADICTION-->
Knapper Widerspruch (1 Satz, darf beliebige Zeichen enthalten — auch Quotes/Doppelpunkte).
<!--CONTRADICTION-->
Weiterer Widerspruch falls vorhanden.
<!--RELATED-->
[[Note-Titel-1]]
[[Note-Titel-2]]
<!--END-->

**Format-Regeln (strikt):**
- Sentinels exakt `<!--CONTRADICTION-->`, `<!--RELATED-->`, `<!--END-->`. ALL_CAPS.
- `duplicate_risk` und `duplicate_path` als Header-Lines VOR dem ersten Sentinel. `duplicate_path:` leer lassen wenn duplicate_risk != "high".
- Bei keinen Widersprüchen: `<!--CONTRADICTION-->`-Block weglassen (nicht leer ausgeben).
- Bei <{min_related} sinnvollen Links: `<!--RELATED-->`-Block weglassen (nicht leer ausgeben).
- Im RELATED-Block EIN Wikilink pro Zeile, exakt `[[Titel]]` ohne weitere Zeichen in der Zeile.
- **EIN** finaler `<!--END-->` schließt den Output.
"""


def _rank_vault_candidates(
    query_title: str,
    query_tokens: set[str],
    existing_concepts: dict[str, str],
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """Hybrid BM25 + Embedding RRF für Vault-Kandidaten-Ranking.

    Ersetzt einfachen Token-Overlap durch zwei-stufiges Retrieval:
    - BM25Okapi (rank_bm25) über tokenisierte Konzept-Titel — IDF-gewichtet,
      robuster als Count-Overlap bei unterschiedlichen Formulierungen.
    - Embedding-Cosine (paraphrase-multilingual-MiniLM-L12-v2): Draft-Titel
      vs. Konzept-Titel — deckt Synonyme, EN/DE-Varianten, Paraphrasen ab.
      Wird nur eingebunden wenn das Modell bereits geladen ist (kein cold-start).
    - RRF (k=60, Cormack et al.): kombiniert beide Rankings verlustfrei.

    Dhakal et al. 2026: Hybrid liefert +41% Precision@10 vs. BM25-allein.
    Praktisch: bei 5 Kandidaten ist der Latenz-Nachteil von Embedding irrelevant.
    """
    import numpy as np

    valid_items = [
        (k, p) for k, p in existing_concepts.items()
        if (VAULT / p).exists()
    ]
    if not valid_items:
        return []
    n = len(valid_items)
    if n <= top_n:
        return [(k, p) for k, p in valid_items]

    keys = [k for k, _ in valid_items]
    paths = [p for _, p in valid_items]

    # BM25 über tokenisierte Konzept-Titel — Index einmal pro Run via id()-Cache
    try:
        from rank_bm25 import BM25Okapi
        cache_key = id(existing_concepts)
        if cache_key not in _bm25_cache or _bm25_cache[cache_key][0] is not keys:
            tokenized_keys = [list(_tokens(k)) or ["_"] for k in keys]
            _bm25_cache[cache_key] = (keys, paths, BM25Okapi(tokenized_keys), tokenized_keys)
        cached_keys, cached_paths, bm25, tokenized_keys = _bm25_cache[cache_key]
        query_toks = list(query_tokens) if query_tokens else list(_tokens(query_title)) or ["_"]
        bm25_scores = np.array(bm25.get_scores(query_toks), dtype=float)
    except Exception:
        # Fallback auf einfachen Token-Overlap
        bm25_scores = np.array([
            float(len(query_tokens & _tokens(k))) for k in keys
        ], dtype=float)

    # Embedding-Cosine nur wenn Modell bereits geladen (kein cold-start durch CrossRef)
    cos_scores: "np.ndarray | None" = None
    try:
        from generative.pipeline import embeddings as _emb_mod
        if _emb_mod._MODEL is not None:
            query_emb = _emb_mod.embed_title(query_title)
            key_embs = _emb_mod._model().encode(keys, show_progress_bar=False,
                                                 normalize_embeddings=True)
            cos_scores = np.array(key_embs.dot(query_emb), dtype=float)
    except Exception:
        pass

    # RRF: Score = Σ 1/(k + rank_i) — addiert Beiträge beider Retriever
    RRF_K = 60
    # argsort().argsort() → Ränge (0 = schlechtester); umkehren → 1 = bester
    bm25_ranks = n - bm25_scores.argsort().argsort()
    rrf = 1.0 / (RRF_K + bm25_ranks)
    if cos_scores is not None:
        cos_ranks = n - cos_scores.argsort().argsort()
        rrf = rrf + 1.0 / (RRF_K + cos_ranks)

    top_idx = rrf.argsort()[::-1][:top_n]
    return [(keys[i], paths[i]) for i in top_idx]


def _read_excerpt(path: Path, max_words: int = 150) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Frontmatter überspringen
        if text.startswith("---"):
            end = text.find("---", 3)
            text = text[end + 3:] if end != -1 else text
        words = text.split()
        return " ".join(words[:max_words])
    except OSError:
        return ""


def _excerpt_from_body(body: str, max_words: int = 150) -> str:
    """Excerpt aus In-Memory-Draft-Body (ohne Frontmatter, da Drafts keinen haben)."""
    return " ".join(body.split()[:max_words])


def run(draft: AtomicNoteDraft, existing_concepts: dict[str, str],
        siblings: dict[str, AtomicNoteDraft] | None = None) -> AtomicNoteDraft:
    # Relevante existierende Notes finden via Content-Token-Overlap.
    # Aliases der draft mit-suchen, weil draft.title oft generisch ("Information Need")
    # ist und die spezifischeren Aliase zusätzliche Match-Tokens liefern.
    query_tokens = _tokens(draft.title)
    for alias in draft.aliases:
        query_tokens |= _tokens(alias)
    for r in draft.related:
        query_tokens |= _tokens(r.strip("[]"))

    # Stage A: BM25 + Embedding RRF Kandidaten-Ranking (FOSS-NLP-Recherche 2026-05-12).
    # Ersetzt einfachen Token-Overlap — deckt Synonyme, EN/DE-Varianten, Paraphrasen ab.
    vault_candidates = _rank_vault_candidates(
        query_title=draft.title,
        query_tokens=query_tokens,
        existing_concepts=existing_concepts,
        top_n=5,
    )

    # Stage B (F5): Pipeline-Sibling-Drafts als zusätzliche Kandidaten.
    # Drafts vom selben PDF-Lauf kennen sich nicht — Cross-Reference sah bisher nur
    # Vault. Bei Kuhlthau-ISP-Phasen führt das zu 0-1 related → Hard-Gate-Fail.
    # Sibling-Drafts werden mit gleicher Token-Overlap-Heuristik geranked, Top 5.
    sibling_candidates: list[tuple[str, AtomicNoteDraft]] = []
    if siblings:
        scored_sib: list[tuple[int, str, AtomicNoteDraft]] = []
        for sib_title, sib_draft in siblings.items():
            if sib_title == draft.title:
                continue  # self
            sib_keys = _tokens(sib_title)
            for alias in sib_draft.aliases:
                sib_keys |= _tokens(alias)
            overlap = len(query_tokens & sib_keys)
            if overlap >= 1:
                scored_sib.append((overlap, sib_title, sib_draft))
        scored_sib.sort(key=lambda t: -t[0])
        sibling_candidates = [(t, d) for _, t, d in scored_sib[:5]]

    total_candidates = len(vault_candidates) + len(sibling_candidates)
    if total_candidates == 0:
        draft.quality_flags.append("⚠️ Keine verwandten Vault- oder Sibling-Notes gefunden — manuell prüfen")
        return draft

    # Short-Circuit: wenn Kandidaten-Pool zu klein für sinnvolle LLM-Auswahl →
    # direkt die verfügbaren Kandidaten als related-Links nutzen, LLM-Call sparen.
    if total_candidates < MIN_RELATED:
        related = []
        for _, p in vault_candidates:
            related.append(f"[[{Path(p).stem}]]")
        for sib_title, _ in sibling_candidates:
            related.append(f"[[{sib_title}]]")
        draft.related = related[:MAX_RELATED]
        draft.quality_flags.append(
            f"⚠️ CrossRef Short-Circuit: nur {total_candidates} Kandidat(en) — "
            f"LLM übersprungen, {len(draft.related)} related-Link(s) direkt gesetzt"
        )
        return draft

    # no-LLM-Modus: BM25+RRF Top-N direkt als related-Links, NLI für Widersprüche.
    if not _config.ENABLE_LLM:
        # Bug #9: deterministischer Duplicate-Check via Token-Overlap
        draft_tokens = _tokens(draft.title)
        for _, cand_path in vault_candidates[:3]:
            cand_title = Path(cand_path).stem
            cand_tokens = _tokens(cand_title)
            if not draft_tokens or not cand_tokens:
                continue
            overlap = len(draft_tokens & cand_tokens) / max(len(draft_tokens), len(cand_tokens))
            if overlap >= 0.7:  # ≥70% Token-Überlap → wahrscheinliches Duplikat (0.8 war zu hoch: "Konzept (Autor)"-Pattern ergibt nur 75%)
                draft.action = "extend"
                draft.extend_path = str(VAULT / cand_path)
                draft.quality_flags.append(f"⚠️ Duplikat-Risiko (no-LLM, overlap={overlap:.0%}) — prüfe: {cand_title}")
                break

        related = []
        for _, p in vault_candidates:
            related.append(f"[[{Path(p).stem}]]")
        for sib_title, _ in sibling_candidates:
            related.append(f"[[{sib_title}]]")
        draft.related = related[:MAX_RELATED]
        if len(draft.related) < MIN_RELATED:
            draft.quality_flags.append(
                f"⚠️ Nur {len(draft.related)} related-Links (no-LLM-Modus) — manuell prüfen"
            )
        vault_excerpts_for_nli = [_read_excerpt(VAULT / p) for _, p in vault_candidates]
        if ENABLE_NLI_VALIDATION:
            nli_confirmed = _nli_validate_contradictions(draft.body, vault_excerpts_for_nli)
            if nli_confirmed:
                draft.quality_flags.append("⚠️ Möglicher Widerspruch (NLI, no-LLM-Modus)")
        draft.quality_flags.append("ℹ️ CrossRef: no-LLM-Modus (BM25+RRF Top-N)")
        return draft

    # Hebel #2: Header OHNE `[[...]]`-Brackets, damit das LLM nicht
    # `[[[[..]]]]` produziert. Sibling-Marker als separate Zeile, damit das
    # LLM den Marker nicht als Teil des Titels in den Wikilink zieht
    # (sonst entsteht `[[Title (Pipeline-Sibling)]]`).
    excerpt_parts = []
    for _, p in vault_candidates:
        excerpt_parts.append(f"### {Path(p).stem}\n{_read_excerpt(VAULT / p)}")
    for sib_title, sib_draft in sibling_candidates:
        excerpt_parts.append(
            f"### {sib_title}\n_(Pipeline-Sibling — gleichgewichtet wie Vault-Notes)_\n"
            f"{_excerpt_from_body(sib_draft.body)}"
        )
    excerpts = "\n\n".join(excerpt_parts)

    prompt = _PROMPT.format(
        min_related=MIN_RELATED,
        max_related=MAX_RELATED,
        title=draft.title,
        body=draft.body[:4000],
        existing_excerpts=excerpts,
    )

    try:
        raw = call_claude(prompt, model=MODEL_CROSS_REF, agent="cross_reference")
    except (RuntimeError) as e:
        import sys
        print(f"      [cross-ref-fail] '{draft.title}' LLM-Call fehlgeschlagen: {str(e)[:80]}",
              file=sys.stderr)
        draft.quality_flags.append("⚠️ Cross-Reference nicht ausgeführt — related-Links manuell prüfen")
        return draft

    data, parse_warnings = parse_cross_reference_output(raw)
    if parse_warnings:
        import sys
        for w in parse_warnings:
            print(f"      [cross-ref-warn] '{draft.title}': {w}", file=sys.stderr)

    # AND-Kombination: Haiku-Widersprüche per NLI-DeBERTa validieren wenn aktiviert.
    # Nur bestätigte Widersprüche → quality_flag; unbestätigte → Soft-Warning.
    # Vault-Excerpts werden an NLI übergeben (Gemma-4-Ansatz: Haiku-Passagen direkt).
    vault_excerpts_for_nli = [_read_excerpt(VAULT / p) for _, p in vault_candidates]
    for contradiction in data.get("contradictions", []):
        if ENABLE_NLI_VALIDATION:
            nli_confirmed = _nli_validate_contradictions(draft.body, vault_excerpts_for_nli)
            if nli_confirmed:
                draft.quality_flags.append(f"⚠️ Widerspruch (Haiku+NLI bestätigt): {contradiction}")
            else:
                draft.quality_flags.append(f"ℹ️ Möglicher Widerspruch (nur Haiku, NLI unbestätigt): {contradiction}")
        else:
            draft.quality_flags.append(f"⚠️ Widerspruch: {contradiction}")

    dup_risk = data.get("duplicate_risk", "none")
    # LLM liefert duplicate_path gelegentlich als "[[Titel]]" oder "Titel|alias" statt
    # reinem Pfad/Titel — normalisieren, sonst entsteht unten "[[[[Titel]]]]" und ein
    # verklammertes Duplikat-Flag (beobachtet im Ebner-Run 2026-06-22).
    dup_path = _clean_wikilink(data.get("duplicate_path") or "").split("|", 1)[0].strip()
    # Typ-bewusstes Blocking: ein Dup-Treffer gegen eine koexistierende Lit-/MoC-/Stub-Note
    # ist KEIN echtes Duplikat (Schema-Lit ≠ Schema-Konzept) — sonst würde eine Konzept-Note
    # fälschlich in ihre eigene Lit-Note gemergt (Ebner-Run 2026-06-23). Dann: kein
    # action=extend, aber den Bezug als related-Link erhalten (Konzept SOLL auf Quelle linken).
    if dup_risk == "high" and not _dup_target_eligible(dup_path, existing_concepts):
        draft.quality_flags.append(f"ℹ️ Verwandte Quelle/Note (kein Konzept-Duplikat): {dup_path}")
        if dup_path:
            dup_link = f"[[{Path(dup_path).stem}]]"
            if dup_link not in data["related"]:
                data["related"].insert(0, dup_link)
    elif dup_risk == "high":
        # Duplikat ist der stärkste mögliche Vault-Beleg — confidence NICHT runter,
        # stattdessen action='extend' setzen + Duplikat als related-Link aufnehmen,
        # damit Confidence-Agent has_vault_corroboration=True erkennt
        draft.quality_flags.append(f"⚠️ Duplikat-Risiko hoch — prüfe: {dup_path}")
        draft.action = "extend"
        if dup_path:
            draft.extend_path = dup_path
            # Wikilink-Form aus dem Pfad bauen (Stem ohne .md)
            from pathlib import Path as _P
            stem = _P(dup_path).stem
            dup_link = f"[[{stem}]]"
            if dup_link not in data["related"]:
                data["related"].insert(0, dup_link)

    # related-Links setzen (nicht nur ergänzen — Cross-Reference ist die Autorität dafür).
    # Defense-in-depth: überschüssige Klammern normalisieren, damit nachträglich (nach
    # parse_cross_reference_output, das _WIKILINK_RE prüft) injizierte Links nie als
    # "[[[[..]]]]" durchrutschen.
    related = []
    for link in data.get("related", []):
        if not link.strip().startswith("[["):
            continue
        inner = _clean_wikilink(link)
        if inner:
            related.append(f"[[{inner}]]")
    draft.related = related[:MAX_RELATED]

    if len(draft.related) < MIN_RELATED:
        draft.quality_flags.append(
            f"⚠️ Nur {len(draft.related)} sinnvolle related-Links — Schema verlangt {MIN_RELATED}–{MAX_RELATED}, manuell prüfen"
        )

    return draft
