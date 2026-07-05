"""Shared NLI-Adapter (mDeBERTa/XNLI, Faithfulness-Gate E4a, #69).

Pure, wiederverwendbare Batching-Schicht rund um `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
— löst die Code-Duplikation zwischen `eval_quality.py` (`_nli_score`/`_nli_best_window`)
und `eval_quality_v2.py` (`_nli_batch`) auf, ohne von dort zu importieren (beide Module
bleiben unverändert). Der dritte NLI-Nutzer `agents.cross_reference._nli_validate_contradictions`
nutzt ein anderes Modell (CrossEncoder `NLI_MODEL_NAME`) mit anderer Semantik (Contradiction-
Gate gegen Vault-Kandidaten) und wird bewusst NICHT migriert.

Keine Pipeline-Verdrahtung, kein Gate, kein ENV-Flag-Auswertung hier — `ENABLE_FAITHFULNESS_GATE`
wird erst vom Faithfulness-Gate (Etappe E6) konsultiert. Dieses Modul ist flag-frei nutzbar.

**Abstain-Vertrag:** Ist das Modell nicht ladbar (fehlende `transformers`/`torch`,
fehlende Model-Files im Cache, offline — kein Download-Zwang), liefert `score_pairs`
`None` zurück statt zu crashen. Der Fehlerfall wird EINMALIG geloggt (kein Spam bei
wiederholten Aufrufen) und danach aus dem Cache beantwortet. Aufrufer werten `None`
als „abstain" (kein Urteil möglich), nicht als „alles unentailed".
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

from generative.config import MDEBERTA_NLI_MODEL

_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class NliScores:
    entailment: float
    neutral: float
    contradiction: float


def _import_and_load():
    """Importiert transformers/torch und lädt Tokenizer+Modell.

    Eigene Funktion (statt inline in `_load_model`), damit Tests den
    Ladevorgang isoliert mocken können, ohne echte ML-Deps zu brauchen.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MDEBERTA_NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MDEBERTA_NLI_MODEL)
    model.eval()
    return tokenizer, model, torch


def _load_model():
    """Lazy-Singleton-Load des mDeBERTa-Modells (Double-Checked-Locking).

    Gibt `(tokenizer, model, torch)` zurück, oder `(None, None, None)` wenn
    das Modell nicht ladbar ist. Cached auch den Fehlerfall — kein wiederholter
    Ladeversuch, kein Log-Spam (siehe Modul-Docstring, Abstain-Vertrag).
    """
    if "loaded" in _MODEL_CACHE:
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"], _MODEL_CACHE["torch"]
    with _MODEL_LOCK:
        if "loaded" in _MODEL_CACHE:  # Double-Checked Locking
            return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"], _MODEL_CACHE["torch"]
        try:
            tokenizer, model, torch_mod = _import_and_load()
            _MODEL_CACHE.update({"tokenizer": tokenizer, "model": model, "torch": torch_mod})
        except Exception as exc:
            print(f"  [nli] mDeBERTa-Modell nicht ladbar, Abstain: {exc}", file=sys.stderr)
            _MODEL_CACHE.update({"tokenizer": None, "model": None, "torch": None})
        # "loaded" erst NACH dem Daten-Update setzen: der lock-freie Fast-Path
        # oben liest die Daten-Keys, sobald er "loaded" sieht — stünde "loaded"
        # vor dem Update, läse ein zweiter Thread während des Modell-Downloads
        # einen leeren Cache (KeyError-Race, Qwen-Review E4).
        _MODEL_CACHE["loaded"] = True
    return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"], _MODEL_CACHE["torch"]


def nli_available() -> bool:
    """True wenn `transformers`/`torch` importierbar sind.

    Reiner Import-Check — KEIN Download-Versuch, kein Modell-Load. Für einen
    schnellen Vorab-Check durch Aufrufer, bevor `score_pairs` (das den echten,
    ggf. langsamen Modell-Load anstößt) aufgerufen wird.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _label_index_map(model) -> dict[str, int]:
    """`id2label` → `{label.lower(): index}` — NIE hartcodierte Indizes.

    Fallen-Hinweis aus den Eval-Vorlagen: `eval_quality.py` kommentiert die
    Standard-Reihenfolge (entailment=0, neutral=1, contradiction=2) explizit
    als NUR "Standard bei Laurer-Modellen" — keine Garantie. Deshalb wird die
    Reihenfolge hier bei jedem Load aus `model.config.id2label` gelesen.
    """
    labels = getattr(model.config, "id2label", {}) or {}
    return {str(label).lower(): int(idx) for idx, label in labels.items()}


def score_pairs(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[NliScores] | None:
    """Batched NLI-Scoring für `(premise, hypothesis)`-Paare.

    Verarbeitet in Chunks von `batch_size` (ein Modell-Call pro Chunk — ein
    einziger Tensor über beliebig viele Paare wäre ein OOM-Risiko bei langen
    Quellfenstern). Truncation auf 512 Tokens (Modell-Limit). Gibt `None`
    zurück (Abstain), wenn das Modell nicht ladbar ist, ein Kern-Label im
    `id2label`-Mapping fehlt oder die Inferenz fehlschlägt — siehe Abstain-
    Vertrag im Modul-Docstring.

    Die Ergebnis-Reihenfolge entspricht exakt der Eingabe-Reihenfolge; das
    Mapping auf Claims liegt beim Aufrufer. Aufrufer sollten Claims, die schon
    deterministisch entschieden sind (z. B. `attribution.check_attribution` →
    `author_missing`/`no_window`), vorab herausfiltern statt teure Inferenz
    zu verschwenden.
    """
    if not pairs:
        return []

    tokenizer, model, torch = _load_model()
    if tokenizer is None or model is None or torch is None:
        return None

    label_map = _label_index_map(model)
    missing = {"entailment", "neutral", "contradiction"} - set(label_map)
    if missing:
        # Kein stiller Index-Fallback: falsches Label-Mapping produzierte
        # lautlos vertauschte Scores (Qwen-Review E4) — lieber Abstain.
        print(f"  [nli] id2label ohne Kern-Label {sorted(missing)}, Abstain", file=sys.stderr)
        return None
    e_idx, n_idx, c_idx = label_map["entailment"], label_map["neutral"], label_map["contradiction"]

    results: list[NliScores] = []
    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            inputs = tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).tolist()
            results.extend(
                NliScores(
                    entailment=float(row[e_idx]),
                    neutral=float(row[n_idx]),
                    contradiction=float(row[c_idx]),
                )
                for row in probs
            )
    except Exception as exc:
        # Abstain-Vertrag gilt auch zur Laufzeit (OOM, Shape-Mismatch, …) —
        # das Gate soll sich enthalten, nicht die Note-Pipeline crashen.
        print(f"  [nli] Inferenz fehlgeschlagen, Abstain: {exc}", file=sys.stderr)
        return None

    return results
