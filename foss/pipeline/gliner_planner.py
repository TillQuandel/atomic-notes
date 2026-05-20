from __future__ import annotations
from functools import lru_cache
from rapidfuzz import fuzz
import re as _re

_ANCHOR_RE_PLAN = _re.compile(r"\s*\(S\.\s*\d+(?:-\d+)?\)")


def _strip_name(text: str) -> str:
    return _ANCHOR_RE_PLAN.sub("", text).strip().lower()

CONCEPT_TYPES = ["Theory", "Concept", "Method", "Metric", "Model", "Framework", "Phenomenon"]
_MODEL_NAME = "urchade/gliner_medium-v2.1"


@lru_cache(maxsize=1)
def _get_model():
    from gliner import GLiNER
    return GLiNER.from_pretrained(_MODEL_NAME)


def extract_concepts(text: str, pages=None, threshold: float = 0.55) -> list[dict]:
    model = _get_model()
    entities = model.predict_entities(text, CONCEPT_TYPES, threshold=threshold)
    page = pages[0] if pages else 1
    return [
        {"name": e["text"].strip(), "type": e["label"], "page": page, "score": e["score"]}
        for e in entities if len(e["text"].strip()) >= 3
    ]


def deduplicate_concepts(concepts: list[dict], threshold: int = 90) -> list[dict]:
    seen: list[dict] = []
    for c in sorted(concepts, key=lambda x: -x.get("score", 0)):
        name = _strip_name(c["name"])
        if not any(fuzz.ratio(name, _strip_name(s["name"])) >= threshold for s in seen):
            seen.append(c)
    return seen


def plan_concepts(chunks, min_concepts: int = 3, min_chunk_count: int = 2) -> list[dict]:
    """Extrahiert Konzepte. Filtert Konzepte die nur in 1 Chunk vorkommen (zu spezifisch)."""
    from collections import Counter
    all_concepts: list[dict] = []
    for chunk in chunks:
        all_concepts.extend(extract_concepts(chunk.text, pages=[chunk.page]))

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
        result = _keybert_fallback(chunks, result)
    return result


def _keybert_fallback(chunks, existing: list[dict]) -> list[dict]:
    try:
        from keybert import KeyBERT
    except ImportError:
        return existing
    model = KeyBERT()
    fulltext = " ".join(c.text for c in chunks)
    keywords = model.extract_keywords(fulltext, top_n=8, stop_words="english")
    new = [{"name": kw, "type": "Concept", "page": 1, "score": score} for kw, score in keywords]
    return deduplicate_concepts(existing + new)
