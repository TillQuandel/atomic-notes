# foss-Pipeline Quality Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce hallucination rate from ~17% to <10% and eliminate generic/language-drift concepts without introducing LLMs.

**Architecture:** Four independent improvements to `gliner_planner.py` and `sentence_extractor.py`. Each can be toggled independently. No new dependencies except `langdetect` (already in many NLP setups). Improvements are additive — baseline eval runs first, then each fix measured separately.

**Tech Stack:** GLiNER, sumy LexRank, rapidfuzz, langdetect, spaCy (optional for POS), sentence-transformers (already installed)

---

## Background

Eval on 3 PDFs (eval_version=4.1, LLM-Judge):

| PDF | Chunks | Notes | Ø hall |
|-----|--------|-------|--------|
| Bates 2017 | 15 | 4 | 18.8% |
| Hiatt ADKAR | 39 | 13 | 16.0% |
| Case 2002 | 2 | 8 | 2.5% |

`sentence_extractor.py` already does concept-scoped sentence selection via `find_concept_sentences()`. The fallback `or sent_tokenize(text)` on line 64 is dangerous — when no sentences contain the concept, it falls back to the full text. Root cause of hallucinations: **poor concept quality** (generic, off-language, low-confidence) forces the fallback, extracting unrelated sentences.

## File Map

| File | Change |
|------|--------|
| `foss/pipeline/gliner_planner.py` | Task 1, 2, 3 |
| `foss/pipeline/sentence_extractor.py` | Task 4 |
| `foss/tests/test_gliner_planner.py` | Task 1, 2, 3 |
| `foss/tests/test_sentence_extractor.py` | Task 4 |
| `foss/requirements.txt` | Task 3 (langdetect) |

---

## Task 1: GLiNER-Threshold + max_concepts-Cap

**Problem:** `extract_concepts()` uses `threshold=0.55` — abstrakte Entity-Typen (`Concept`, `Method`) erreichen systematisch niedrigere Scores. Cap fehlt: Beutelspacher → 56 Konzepte.

**Files:**
- Modify: `foss/pipeline/gliner_planner.py:22-29` (extract_concepts), `foss/pipeline/gliner_planner.py:41-60` (plan_concepts)
- Test: `foss/tests/test_gliner_planner.py`

- [ ] **Step 1: Failing test schreiben**

```python
# foss/tests/test_gliner_planner.py
import pytest
from unittest.mock import patch, MagicMock

def test_extract_concepts_default_threshold_is_higher():
    """Threshold muss >= 0.7 sein."""
    import inspect
    from foss.pipeline.gliner_planner import extract_concepts
    sig = inspect.signature(extract_concepts)
    assert sig.parameters["threshold"].default >= 0.7

def test_plan_concepts_respects_max_cap():
    """plan_concepts darf max_concepts nicht ueberschreiten."""
    from foss.pipeline.gliner_planner import plan_concepts, Chunk
    # 5 Chunks mit vielen Konzepten simulieren
    chunks = [MagicMock(text=f"Information literacy framework concept model {i}", page=1) for i in range(5)]
    with patch("foss.pipeline.gliner_planner.extract_concepts") as mock_ec:
        mock_ec.return_value = [
            {"name": f"concept_{i}", "type": "Concept", "page": 1, "score": 0.9}
            for i in range(20)
        ]
        result = plan_concepts(chunks, max_concepts=10)
    assert len(result) <= 10
```

- [ ] **Step 2: Test laufen lassen (muss FAIL)**

```bash
cd C:/Users/tillq/source/repos/atomic-notes
python -m pytest foss/tests/test_gliner_planner.py::test_extract_concepts_default_threshold_is_higher foss/tests/test_gliner_planner.py::test_plan_concepts_respects_max_cap -v
```

Expected: FAIL — `AssertionError` weil threshold=0.55 und max_concepts-Parameter fehlt.

- [ ] **Step 3: Implementierung**

`foss/pipeline/gliner_planner.py` — zwei Änderungen:

```python
def extract_concepts(text: str, pages=None, threshold: float = 0.75) -> list[dict]:
    # threshold: 0.55 -> 0.75 (abstrakte Konzepte haben systematisch niedrigere Scores)
    model = _get_model()
    entities = model.predict_entities(text, CONCEPT_TYPES, threshold=threshold)
    page = pages[0] if pages else 1
    return [
        {"name": e["text"].strip(), "type": e["label"], "page": page, "score": e["score"]}
        for e in entities if len(e["text"].strip()) >= 3
    ]


def plan_concepts(
    chunks,
    min_concepts: int = 3,
    min_chunk_count: int = 2,
    max_concepts: int = 20,
) -> list[dict]:
    """Extrahiert Konzepte. Cap bei max_concepts verhindert Explosion bei grossen PDFs."""
    from collections import Counter
    all_concepts: list[dict] = []
    for chunk in chunks:
        all_concepts.extend(extract_concepts(chunk.text, pages=[chunk.page]))

    name_counts: Counter = Counter()
    for c in all_concepts:
        name_counts[_strip_name(c["name"])] += 1
    prominent = [c for c in all_concepts if name_counts[_strip_name(c["name"])] >= min_chunk_count]

    source = prominent if len(prominent) >= min_concepts else all_concepts
    result = deduplicate_concepts(source)
    if len(result) < min_concepts:
        result = _keybert_fallback(chunks, result)

    # Cap: nach Dedup + Fallback auf Score abschneiden
    result = sorted(result, key=lambda x: -x.get("score", 0))[:max_concepts]
    return result
```

- [ ] **Step 4: Tests laufen lassen (müssen PASS)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_extract_concepts_default_threshold_is_higher foss/tests/test_gliner_planner.py::test_plan_concepts_respects_max_cap -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add foss/pipeline/gliner_planner.py foss/tests/test_gliner_planner.py
git commit -m "feat(foss): GLiNER-Threshold 0.55->0.75, max_concepts-Cap (default=20)"
```

---

## Task 2: Konzept-Spezifizitäts-Filter (Blacklist + Subword-Proxy)

**Problem:** Generika wie `mean`, `management`, `advantage`, `tinnitus` passieren GLiNER. Zwei komplementäre Filter ohne Embeddings.

**Files:**
- Modify: `foss/pipeline/gliner_planner.py` — neue Funktion `_is_specific_concept()`
- Test: `foss/tests/test_gliner_planner.py`

- [ ] **Step 1: Failing tests schreiben**

```python
def test_generic_concepts_filtered():
    from foss.pipeline.gliner_planner import _is_specific_concept
    assert _is_specific_concept("mean") is False
    assert _is_specific_concept("management") is False
    assert _is_specific_concept("advantage") is False
    assert _is_specific_concept("information") is False

def test_specific_concepts_pass():
    from foss.pipeline.gliner_planner import _is_specific_concept
    assert _is_specific_concept("information literacy framework") is True
    assert _is_specific_concept("ADKAR model") is True
    assert _is_specific_concept("LexRank") is True
    assert _is_specific_concept("constructivism") is True
```

- [ ] **Step 2: Test laufen lassen (muss FAIL)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_generic_concepts_filtered foss/tests/test_gliner_planner.py::test_specific_concepts_pass -v
```

Expected: FAIL — `ImportError: cannot import name '_is_specific_concept'`

- [ ] **Step 3: Implementierung**

In `foss/pipeline/gliner_planner.py` nach den Konstanten einfügen:

```python
_GENERIC_BLACKLIST = frozenset({
    # Abstrakte Generika
    "information", "system", "process", "method", "model", "data", "analysis",
    "management", "mean", "average", "advantage", "result", "approach", "aspect",
    "concept", "theory", "issue", "factor", "element", "component", "feature",
    "problem", "solution", "area", "level", "type", "form", "role",
    "ability", "use", "need", "way", "part", "point", "case", "end", "set",
    # Plural-Generika (Gemini-Finding: 7-8 Zeichen aber trotzdem generisch)
    "methods", "models", "surveys", "rubrics", "metrics", "factors", "systems",
    "concepts", "aspects", "results", "studies", "issues", "elements",
    # Fachfremde Einzelbegriffe die durchrutschen koennen
    "tinnitus",
})

def _is_specific_concept(name: str) -> bool:
    """Prüft ob ein Konzept spezifisch genug ist (Blacklist + Subword-Proxy).

    Subword-Proxy: kurze, häufige Wörter haben wenige Subword-Tokens.
    'sun' = 1 Token (generisch), 'constructivism' = 4+ Tokens (spezifisch).
    Wir approximieren das mit Zeichenlänge des Einzelworts.
    """
    normalized = name.strip().lower()

    # 1. Blacklist
    if normalized in _GENERIC_BLACKLIST:
        return False

    words = normalized.split()

    # 2. Einzelwörter unter 8 Zeichen sind oft generisch
    if len(words) == 1 and len(normalized) < 8:
        return False

    # 3. Mehrwörter-Ausdrücke sind fast immer spezifisch
    if len(words) >= 2:
        return True

    # 4. Lange Einzelwörter (>=8 Zeichen) gelten als spezifisch
    return True
```

Dann in `extract_concepts()` nach dem List-Comprehension-Filter ergänzen:

```python
def extract_concepts(text: str, pages=None, threshold: float = 0.75) -> list[dict]:
    model = _get_model()
    entities = model.predict_entities(text, CONCEPT_TYPES, threshold=threshold)
    page = pages[0] if pages else 1
    return [
        {"name": e["text"].strip(), "type": e["label"], "page": page, "score": e["score"]}
        for e in entities
        if len(e["text"].strip()) >= 3 and _is_specific_concept(e["text"].strip())
    ]
```

- [ ] **Step 4: Tests laufen lassen (müssen PASS)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_generic_concepts_filtered foss/tests/test_gliner_planner.py::test_specific_concepts_pass -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add foss/pipeline/gliner_planner.py foss/tests/test_gliner_planner.py
git commit -m "feat(foss): Konzept-Spezifizitaets-Filter (Blacklist + Subword-Proxy)"
```

---

## Task 3: Sprach-Filter (langdetect)

**Problem:** Mehrsprachige Sammelbände (Beutelspacher) liefern deutsche Konzepte (`bekanntheitsgrad`, `datenschutz`) in englischem Haupttext.

**Files:**
- Modify: `foss/pipeline/gliner_planner.py` — Sprach-Filter in `extract_concepts()`
- Modify: `foss/requirements.txt`
- Test: `foss/tests/test_gliner_planner.py`

- [ ] **Step 1: langdetect zu requirements.txt**

```
langdetect>=1.0.9
```

```bash
cd C:/Users/tillq/source/repos/atomic-notes
pip install langdetect
```

- [ ] **Step 2: Failing test**

```python
def test_language_drift_filtered():
    from foss.pipeline.gliner_planner import _matches_language
    assert _matches_language("bekanntheitsgrad", "en") is False
    assert _matches_language("datenschutz", "en") is False
    assert _matches_language("information literacy", "en") is True
    assert _matches_language("informationskompetenz", "de") is True

def test_extract_concepts_filters_language():
    from foss.pipeline.gliner_planner import extract_concepts
    # Simuliere: Konzept-Name in falscher Sprache wird gefiltert
    from unittest.mock import patch, MagicMock
    mock_entity = MagicMock()
    mock_entity.__getitem__ = lambda self, k: {
        "text": "bekanntheitsgrad", "label": "Concept", "score": 0.9
    }[k]
    with patch("foss.pipeline.gliner_planner._get_model") as mock_model:
        mock_model.return_value.predict_entities.return_value = [
            {"text": "bekanntheitsgrad", "label": "Concept", "score": 0.9}
        ]
        result = extract_concepts("some english text", main_language="en")
    assert len(result) == 0
```

- [ ] **Step 3: Test laufen lassen (muss FAIL)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_language_drift_filtered foss/tests/test_gliner_planner.py::test_extract_concepts_filters_language -v
```

Expected: FAIL — `ImportError: cannot import name '_matches_language'`

- [ ] **Step 4: Implementierung**

In `foss/pipeline/gliner_planner.py`:

```python
def _matches_language(text: str, main_language: str) -> bool:
    """Prüft ob Konzept-Text in der Hauptsprache ist. Bei Unsicherheit: True (behalten)."""
    if len(text.split()) < 2 and len(text) < 10:
        # Zu kurz für zuverlässige Erkennung -> behalten
        return True
    try:
        from langdetect import detect, LangDetectException
        detected = detect(text)
        return detected == main_language
    except Exception:
        return True  # Im Zweifel behalten


def extract_concepts(
    text: str, pages=None, threshold: float = 0.75, main_language: str = "en"
) -> list[dict]:
    model = _get_model()
    entities = model.predict_entities(text, CONCEPT_TYPES, threshold=threshold)
    page = pages[0] if pages else 1
    return [
        {"name": e["text"].strip(), "type": e["label"], "page": page, "score": e["score"]}
        for e in entities
        if len(e["text"].strip()) >= 3
        and _is_specific_concept(e["text"].strip())
        and _matches_language(e["text"].strip(), main_language)
    ]
```

Dann `plan_concepts()` aktualisieren um `main_language` durchzureichen:

```python
def plan_concepts(
    chunks,
    min_concepts: int = 3,
    min_chunk_count: int = 2,
    max_concepts: int = 20,
    main_language: str = "en",
) -> list[dict]:
    from collections import Counter
    all_concepts: list[dict] = []
    for chunk in chunks:
        all_concepts.extend(
            extract_concepts(chunk.text, pages=[chunk.page], main_language=main_language)
        )

    name_counts: Counter = Counter()
    for c in all_concepts:
        name_counts[_strip_name(c["name"])] += 1
    prominent = [c for c in all_concepts if name_counts[_strip_name(c["name"])] >= min_chunk_count]

    source = prominent if len(prominent) >= min_concepts else all_concepts
    result = deduplicate_concepts(source)
    if len(result) < min_concepts:
        result = _keybert_fallback(chunks, result)

    return sorted(result, key=lambda x: -x.get("score", 0))[:max_concepts]
```

`orchestrator.py` — `lang` an `plan_concepts` übergeben:

```python
concepts = plan_concepts(chunks, main_language=lang)
```

- [ ] **Step 5: Tests laufen lassen (müssen PASS)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_language_drift_filtered foss/tests/test_gliner_planner.py::test_extract_concepts_filters_language -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add foss/pipeline/gliner_planner.py foss/orchestrator.py foss/requirements.txt foss/tests/test_gliner_planner.py
git commit -m "feat(foss): Sprach-Filter via langdetect (verhindert Konzept-Drift bei mehrsprachigen PDFs)"
```

---

## Task 4: Fallback-Entfernung in sentence_extractor

**Problem:** `extract_body_for_concept()` fällt auf `sent_tokenize(text)` (ganzer Text!) zurück wenn kein Satz das Konzept erwähnt. Das erzeugt thematisch irrelevante Sätze.

**Files:**
- Modify: `foss/pipeline/sentence_extractor.py:64`
- Test: `foss/tests/test_sentence_extractor.py`

- [ ] **Step 1: Failing test**

```python
# foss/tests/test_sentence_extractor.py
def test_extract_body_returns_empty_when_no_concept_sentences():
    """Kein Fallback auf ganzen Text wenn Konzept nicht vorkommt."""
    from foss.pipeline.sentence_extractor import extract_body_for_concept
    text = "This sentence is about cats. Another sentence about dogs. A third about fish."
    result = extract_body_for_concept("information literacy", text)
    assert result == [], f"Expected [], got {result}"
```

- [ ] **Step 2: Test laufen lassen (muss FAIL)**

```bash
python -m pytest foss/tests/test_sentence_extractor.py::test_extract_body_returns_empty_when_no_concept_sentences -v
```

Expected: FAIL — gibt Sätze zurück wegen `or sent_tokenize(text)`-Fallback.

- [ ] **Step 3: Implementierung**

In `foss/pipeline/sentence_extractor.py` Zeile 64:

```python
def extract_body_for_concept(
    concept: str, text: str, n: int = 4, language: str = "english"
) -> list[str]:
    """LexRank über Konzept-Sentence-Cluster -> Top-n Sätze.
    Gibt leere Liste zurück wenn kein Satz das Konzept erwähnt (kein Fallback auf Volltext).
    """
    sentences = find_concept_sentences(concept, text)
    if not sentences:
        return []
    if len(sentences) <= n:
        return sentences
    cluster = " ".join(sentences)
    parser = PlaintextParser.from_string(cluster, Tokenizer(language))
    return [str(s) for s in LexRankSummarizer()(parser.document, sentences_count=n)]
```

Orchestrator muss leere Bodies überspringen — in `orchestrator.py`:

```python
for c in concepts:
    body = add_page_anchors(
        extract_body_for_concept(c["name"], fulltext),
        [c.get("page", 1)]
    )
    if not body:
        continue  # Konzept ohne belegbare Sätze überspringen
    note = AtomicNoteFoss(...)
    notes.append(note)
```

- [ ] **Step 4: Tests laufen lassen (müssen PASS)**

```bash
python -m pytest foss/tests/test_sentence_extractor.py::test_extract_body_returns_empty_when_no_concept_sentences -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add foss/pipeline/sentence_extractor.py foss/orchestrator.py foss/tests/test_sentence_extractor.py
git commit -m "fix(foss): Fallback-Entfernung in extract_body_for_concept (kein sent_tokenize Volltext-Fallback)"
```

---

---

## Task 5: KeyBERT-Fallback durch Filter laufen lassen

**Problem (Gemini-Finding):** `_keybert_fallback()` umgeht alle neuen Filter aus Task 2+3. Wenn GLiNER nichts findet, kann KeyBERT generische oder fremdsprachige Begriffe einschleusen.

**Files:**
- Modify: `foss/pipeline/gliner_planner.py` — `_keybert_fallback()` um Filter erweitern
- Test: `foss/tests/test_gliner_planner.py`

- [ ] **Step 1: Failing test**

```python
def test_keybert_fallback_filtered():
    """KeyBERT-Fallback muss Spezifizitaets- und Sprach-Filter durchlaufen."""
    from foss.pipeline.gliner_planner import _keybert_fallback
    # "management" ist auf Blacklist, soll nicht im Output erscheinen
    chunks = [MagicMock(text="change management process", page=1)]
    with patch("foss.pipeline.gliner_planner.KeyBERT") as mock_kb:
        mock_kb.return_value.extract_keywords.return_value = [
            ("management", 0.9), ("information literacy framework", 0.8)
        ]
        result = _keybert_fallback(chunks, [], main_language="en")
    names = [r["name"] for r in result]
    assert "management" not in names
    assert "information literacy framework" in names
```

- [ ] **Step 2: Test laufen lassen (muss FAIL)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_keybert_fallback_filtered -v
```

- [ ] **Step 3: Implementierung**

```python
def _keybert_fallback(
    chunks, existing: list[dict], main_language: str = "en"
) -> list[dict]:
    try:
        from keybert import KeyBERT
    except ImportError:
        return existing
    model = KeyBERT()
    fulltext = " ".join(c.text for c in chunks)
    keywords = model.extract_keywords(fulltext, top_n=8, stop_words="english")
    new = [
        {"name": kw, "type": "Concept", "page": 1, "score": score}
        for kw, score in keywords
        if _is_specific_concept(kw) and _matches_language(kw, main_language)
    ]
    return deduplicate_concepts(existing + new)
```

`plan_concepts()` — `main_language` an Fallback weitergeben:

```python
    if len(result) < min_concepts:
        result = _keybert_fallback(chunks, result, main_language=main_language)
```

- [ ] **Step 4: Test laufen lassen (muss PASS)**

```bash
python -m pytest foss/tests/test_gliner_planner.py::test_keybert_fallback_filtered -v
```

- [ ] **Step 5: Commit**

```bash
git add foss/pipeline/gliner_planner.py foss/tests/test_gliner_planner.py
git commit -m "fix(foss): KeyBERT-Fallback laeuft jetzt durch Spezifizitaets- und Sprach-Filter"
```

---

## Eval nach allen Tasks

Nach Task 1-4: Bates 2017 + Hiatt ADKAR neu laufen und vergleichen.

```bash
# foss run
python -m foss.orchestrator --source "<pdf>" --out-dir "./tmp/foss-<name>-v2" --eval-db ".cache/atomic_analytics.db"

# LLM-Judge eval
python run_eval.py --run-id <uuid> --notes "./tmp/foss-<name>-v2" --pdf "<pdf>"
```

Ziel: hallucination_rate < 10% auf Bates + Hiatt.

---

## Self-Review

**Spec-Coverage:**
- [x] Threshold-Erhöhung → Task 1
- [x] max_concepts-Cap → Task 1
- [x] Generika-Filter (Blacklist + Subword) → Task 2
- [x] Sprach-Filter → Task 3
- [x] Fallback-Entfernung → Task 4
- [x] Eval-Vergleich → Eval-Abschnitt

**Placeholders:** keine — alle Code-Blöcke vollständig.

**Typ-Konsistenz:** `main_language: str` wird in Task 3 eingeführt und in Task 3 Schritt 4 (orchestrator) konsistent verwendet. `max_concepts` in Task 1 eingeführt und in Task 3 beibehalten.
