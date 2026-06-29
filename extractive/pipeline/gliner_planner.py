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
    """Prueft ob ein Konzept spezifisch genug ist (Blacklist + Subword-Proxy).

    Regeln (in dieser Reihenfolge):
    1. Blacklist-Treffer → False
    2. Mehrwort-Begriff → True
    3. Einwort mit Grossbuchstabe (Eigenname/Akronym) → True
    4. Einwort < 8 Zeichen (rein lowercase) → False
    5. Sonst → True
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
    # Kurze rein-lowercase Einzelwoerter sind zu generisch
    if len(normalized) < 8:
        return False
    return True


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


@lru_cache(maxsize=1)
def _get_model():
    from gliner import GLiNER

    return GLiNER.from_pretrained(_MODEL_NAME)


def extract_concepts(text: str, page: int = 1, threshold: float = 0.75, main_language: str = "en") -> list[dict]:
    model = _get_model()
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
) -> list[dict]:
    """Extrahiert Konzepte. Filtert Konzepte die nur in 1 Chunk vorkommen (zu spezifisch)."""
    # Sicherstellen dass max_concepts nicht kleiner als min_concepts ist
    max_concepts = max(max_concepts, min_concepts)
    from collections import Counter

    all_concepts: list[dict] = []
    for chunk in chunks:
        all_concepts.extend(extract_concepts(chunk.text, page=chunk.page, main_language=main_language))

    # Prominenz-Filter: Konzept muss in >= min_chunk_count Chunks vorkommen
    # (verhindert Einzel-Chunk-Artefakte wie "avoidance", "blunting")
    name_counts: Counter = Counter()
    for c in all_concepts:
        name_counts[_strip_name(c["name"])] += 1
    prominent = [c for c in all_concepts if name_counts[_strip_name(c["name"])] >= min_chunk_count]

    # Fallback auf alle wenn zu wenige prominent
    source = prominent if len(prominent) >= min_concepts else all_concepts
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
