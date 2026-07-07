"""Body-Embeddings für Cluster-Erkennung (4-Stage ER-Pipeline, Stage 2).

Nutzt sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` lokal.
Modell ist paraphrase-trained → optimiert für Cosine-Similarity zwischen
semantisch ähnlichen Texten verschiedener Wortwahl. Multilingual (50+
Sprachen) → passt zu DE-Bodies mit eingestreuten EN-Direktzitaten.

Token-Limit 128 wird durch Sentence-Chunking + Mean-Pooling überbrückt.
"""

from __future__ import annotations
import re
import threading

# Singleton-Modell — Lade-Zeit ~3s, danach pro Embedding ~10ms
_MODEL = None
_MODEL_LOCK = threading.Lock()

_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_ml_model(factory, model_name: str):
    """Führt einen ML-Modell-Load aus und übersetzt Low-Level-Ladefehler in
    eine actionable Fehlermeldung (#146 H2).

    torch/sentence-transformers scheitern unter RAM-Druck typischerweise schon
    beim DLL-Load (`OSError [WinError 1455]: Die Auslagerungsdatei ist zu
    klein`) oder mit MemoryError/RuntimeError — als nackter Traceback mit rc=1
    ist die Ursache für Nutzer nicht erkennbar. Kein fail-soft: der Lauf darf
    weiterhin sterben (Embeddings sind funktional nötig), aber mit klarer
    Ursache. ImportError (fehlende Dependency) propagiert unverändert —
    das ist kein RAM-Problem.
    """
    try:
        return factory()
    except (OSError, MemoryError, RuntimeError) as exc:
        raise RuntimeError(
            f"ML-Modell '{model_name}' konnte nicht geladen werden "
            f"({type(exc).__name__}: {exc}) — vermutlich RAM-Druck "
            "(typisch: WinError 1455, Auslagerungsdatei zu klein). "
            "Andere Programme schließen und erneut versuchen."
        ) from exc


def _load_sentence_transformer():
    """Import + Konstruktion zusammen: der torch-DLL-Load beim ersten
    `import sentence_transformers` ist genau der Schritt, der unter
    RAM-Druck mit WinError 1455 scheitert — muss mit in den try-Scope."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_ST_MODEL_NAME)


def _model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:  # Double-Checked Locking
                _MODEL = load_ml_model(_load_sentence_transformer, _ST_MODEL_NAME)
    return _MODEL


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _sentences(body: str) -> list[str]:
    """Split Body in Sätze. Heuristik reicht für Cosine-Similarity-Use-Case
    — keine perfekte Tokenisierung nötig, nur Chunks unter dem 128-Token-Limit."""
    raw = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    # Markdown-Header und Bullet-Points sind eigene Chunks
    expanded: list[str] = []
    for s in raw:
        # Bei Zeilenumbrüchen weiter splitten (Bold-Header etc.)
        for part in s.split("\n"):
            part = part.strip("*-> \t")
            if part:
                expanded.append(part)
    return expanded


def embed_body(body: str):
    """Body → 384-dim numpy-Array. Mean-Pooling über Satz-Embeddings.

    Bei langen Bodies (>128 Tokens) liefert Sentence-Mean-Pooling stabilere
    Cosine-Werte als naives Truncation. Pattern aus sentence-transformers-Doku
    für Long-Document-Cosine-Similarity.
    """
    import numpy as np

    sents = _sentences(body)
    if not sents:
        return np.zeros(_model().get_sentence_embedding_dimension())
    embs = _model().encode(sents, show_progress_bar=False, normalize_embeddings=True)
    # Mean-Pooling über Sätze, dann Re-Normalisieren für Cosine
    mean = embs.mean(axis=0)
    norm = (mean**2).sum() ** 0.5
    return mean / norm if norm > 0 else mean


def embed_title(title: str):
    """Titel → 384-dim numpy-Array. Direkte Kodierung ohne Satz-Splitting.
    Optimiert für Kurz-Texte (V35 semantisches Titel-Matching)."""
    import numpy as np

    t = title.strip()
    if not t:
        return np.zeros(_model().get_sentence_embedding_dimension())
    # Direkte Kodierung eines einzelnen Strings
    emb = _model().encode([t], show_progress_bar=False, normalize_embeddings=True)[0]
    return emb


def cosine(a, b) -> float:
    """Cosine zwischen normalisierten Embeddings = Dot-Product."""
    return float((a * b).sum())
