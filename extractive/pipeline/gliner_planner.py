from __future__ import annotations
from functools import lru_cache
from rapidfuzz import fuzz
import re as _re

_ANCHOR_RE_PLAN = _re.compile(r"\s*\(S\.\s*\d+(?:-\d+)?\)")


def _strip_name(text: str) -> str:
    return _ANCHOR_RE_PLAN.sub("", text).strip().lower()


CONCEPT_TYPES = ["Theory", "Concept", "Method", "Metric", "Model", "Framework", "Phenomenon"]
_MODEL_NAME = "urchade/gliner_medium-v2.1"

_GENERIC_BLACKLIST = frozenset(
    {
        # Abstrakte Generika
        "information",
        "system",
        "process",
        "method",
        "model",
        "data",
        "analysis",
        "management",
        "mean",
        "average",
        "advantage",
        "result",
        "approach",
        "aspect",
        "concept",
        "theory",
        "issue",
        "factor",
        "element",
        "component",
        "feature",
        "problem",
        "solution",
        "area",
        "level",
        "type",
        "form",
        "role",
        "ability",
        "use",
        "need",
        "way",
        "part",
        "point",
        "case",
        "end",
        "set",
        # Plural-Generika
        "methods",
        "models",
        "surveys",
        "rubrics",
        "metrics",
        "factors",
        "systems",
        "concepts",
        "aspects",
        "results",
        "studies",
        "issues",
        "elements",
        # Fachfremde Einzelbegriffe
        "tinnitus",
    }
)


def _is_specific_concept(name: str) -> bool:
    """Prueft ob ein Konzept spezifisch genug ist (Blacklist + Eigennamen-Regel).

    Regeln (in dieser Reihenfolge):
    1. Blacklist-Treffer → False
    2. Mehrwort-Begriff → True
    3. Einwort mit Grossbuchstabe (Eigenname/Akronym) → True
    4. Einwort rein lowercase → False

    Zu 4: Rein-lowercase Einzelwoerter sind fast immer GLiNER-Artefakte und keine
    Atomic-Note-Titel (die reale Note 'chemistry' handelte von Atomicity). Der
    fruehere >=8-Zeichen-Proxy liess 'chemistry'/'knowledge'/'retrieval'/'conceptual'
    passieren (#167a). Eigennamen ueberleben ueber den Grossbuchstaben (Regel 3).
    """
    stripped = name.strip()
    normalized = stripped.lower()
    if normalized in _GENERIC_BLACKLIST:
        return False
    words = stripped.split()
    if len(words) >= 2:
        return True
    # Einwort: Eigenname/Akronym (mind. ein Grossbuchstabe) ist spezifisch
    if any(c.isupper() for c in stripped):
        return True
    # Rein-lowercase Einzelwoerter sind zu generisch
    return False


_GERMANIC_NON_ENGLISH = frozenset({"de", "da", "nl", "af", "sv", "no", "lb", "fy"})


def _matches_language(text: str, main_language: str) -> bool:
    """Prueft ob Konzept-Text in der Hauptsprache ist. Bei Unsicherheit: True (behalten).

    Besonderheit: langdetect verwechselt verwandte germanische Sprachen (de/da/nl).
    - main_language='en': alle germanischen Erkennungen werden gefiltert (nicht englisch)
    - main_language='de': germanische Erkennungen (de/da/nl) zaehlen als Match
      (langdetect erkennt deutsche Komposita manchmal als 'da')
    """
    if len(text.split()) < 2 and len(text) < 10:
        return True
    try:
        from langdetect import detect

        detected = detect(text)
        if detected == main_language:
            return True
        if main_language == "en":
            # Germanische Erkennungen sind definitiv nicht-englisch → filtern
            return detected not in _GERMANIC_NON_ENGLISH
        if main_language == "de":
            # Germanische Verwechslungen (da/nl) fuer deutsche Komposita tolerieren
            return detected in _GERMANIC_NON_ENGLISH
        return False
    except Exception:
        return True


@lru_cache(maxsize=2)
def _get_model(device: str = "cpu"):
    from gliner import GLiNER

    # #167d: --device einloesen — gliner 0.2.27 laedt die Gewichte via map_location.
    return GLiNER.from_pretrained(_MODEL_NAME, map_location=device)


def extract_concepts(
    text: str, page: int = 1, threshold: float = 0.75, main_language: str = "en", device: str = "cpu"
) -> list[dict]:
    model = _get_model(device)
    entities = model.predict_entities(text, CONCEPT_TYPES, threshold=threshold)
    return [
        {"name": name, "type": e["label"], "page": page, "score": e["score"]}
        for e in entities
        if (name := e["text"].strip())
        and len(name) >= 3
        and _is_specific_concept(name)
        and _matches_language(name, main_language)
    ]


def deduplicate_concepts(concepts: list[dict], threshold: int = 90) -> list[dict]:
    seen: list[dict] = []
    for c in sorted(concepts, key=lambda x: -x.get("score", 0)):
        name = _strip_name(c["name"])
        if not any(fuzz.ratio(name, _strip_name(s["name"])) >= threshold for s in seen):
            seen.append(c)
    return seen


def plan_concepts(
    chunks,
    min_concepts: int = 3,
    min_chunk_count: int = 2,
    max_concepts: int = 20,
    main_language: str = "en",
    device: str = "cpu",
) -> list[dict]:
    """Extrahiert Konzepte. Filtert Konzepte die nur in 1 Chunk vorkommen (zu spezifisch)."""
    # Sicherstellen dass max_concepts nicht kleiner als min_concepts ist
    max_concepts = max(max_concepts, min_concepts)
    from collections import Counter

    all_concepts: list[dict] = []
    for chunk in chunks:
        all_concepts.extend(extract_concepts(chunk.text, page=chunk.page, main_language=main_language, device=device))

    # Prominenz-Filter: Konzept muss in >= min_chunk_count Chunks vorkommen
    # (verhindert Einzel-Chunk-Artefakte wie "avoidance", "blunting")
    name_counts: Counter = Counter()
    for c in all_concepts:
        name_counts[_strip_name(c["name"])] += 1
    prominent = [c for c in all_concepts if name_counts[_strip_name(c["name"])] >= min_chunk_count]

    # Fallback auf alle wenn zu wenige *distinkte* prominente Konzepte (nicht Instanzen).
    # len(prominent) zaehlte Instanzen: 2 Konzepte in je 8/3 Chunks = 11 >= 3 blockierte
    # den Fallback, obwohl nur 2 distinkte prominent sind. Frueher stopfte der KeyBERT-
    # Fallback dann lowercase-Einzelwoerter nach; nach #167a surft dieser Zweig echte
    # 1-Chunk-Mehrwortkonzepte an (bates: 2 -> 6 statt 6 Schrott-Notes).
    n_prominent = len({_strip_name(c["name"]) for c in prominent})
    source = prominent if n_prominent >= min_concepts else all_concepts
    result = deduplicate_concepts(source)
    if len(result) < min_concepts:
        result = _keybert_fallback(chunks, result, main_language=main_language)
    result = sorted(result, key=lambda x: -x.get("score", 0))[:max_concepts]
    return result


def _keybert_fallback(chunks, existing: list[dict], main_language: str = "en") -> list[dict]:
    try:
        from keybert import KeyBERT
    except ImportError:
        return existing
    model = KeyBERT()
    fulltext = " ".join(c.text for c in chunks)
    stop_words = "english" if main_language == "en" else None
    keywords = model.extract_keywords(fulltext, top_n=8, stop_words=stop_words)
    new = [
        {"name": kw, "type": "Concept", "page": 1, "score": score}
        for kw, score in keywords
        if _is_specific_concept(kw) and _matches_language(kw, main_language)
    ]
    return deduplicate_concepts(existing + new)
