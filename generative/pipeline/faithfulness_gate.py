"""Gate-Logik des Faithfulness-Gates (Faithfulness-Gate E5a, #69).

Pure Kombination der bereits gemergten Bausteine zu einem Verdikt pro
High-Risk-Claim — keine neue ML-Logik, keine Pipeline-Verdrahtung (die kommt
erst in E6):

- `page_index.claim_source_window` liefert das Seitenfenster
- `claims.decompose_claims` liefert die High-Risk-Claims
- `attribution.check_attribution` prüft Fremd-Autoren-Präsenz (Stufe 2a)
- `nli.score_pairs` (mDeBERTa) prüft Entailment (Stufe 2b)

**Ablauf pro Claim:**

1. `is_quote=True` → übersprungen, kein Verdict (Blockquotes sind
   anker-verifiziert).
2. Kein `anchor_page` oder kein Seitenfenster im Index → `abstain_no_window`
   (zählt als abstained, NIE als Fail).
3. Nur bei `"attribution" in risk_types`: `check_attribution` — `author_missing`
   → `failed_attribution`. Sonst weiter zu Schritt 4 (`no_window` kann hier
   nicht mehr auftreten, Schritt 2 hat das schon gefiltert).
4. Entailment für ALLE verbliebenen Claims: die Top-`top_k`-Sätze des Fensters
   (via `top_k_sentences`, MiniLM-Cosine-Ranking) plus deren kumulative
   Prefix-Konkatenationen top1..j als zusätzliche Premises (Deckung kann über
   mehrere Sätze verteilt sein — reduziert False-Fails bei legitimer Synthese;
   Prefix- statt nur Voll-Konkat, siehe `_claim_premises`). Maximaler
   Entailment-Score über alle Premises ≥ `entail_threshold` → `supported`,
   sonst `failed_entailment`. `score_pairs` → `None` → `abstain_nli`.
4b. Unverifizierbare-Zahlen-Abstain (Kalibrierung E5b, 2026-07-05): ein Claim
   mit `number`-Risk, der an 4. scheitern würde, dessen sämtliche Zahlen aber
   (kanonisiert, Seiten-Refs/Jahre/Footnote-Marker maskiert) im Fenster
   vorkommen, wird `abstain_unverifiable_numbers` statt `failed_entailment`:
   pdftotext zerlegt Tabellen in zuordnungslose Zahlen-Fragmente (`369 (99%)`),
   und Synthese-Claims über mehrere Absätze entailen satz-basiert nicht (der
   perfekte Stütz-Satz „group sizes (8 versus 19)" ergab e=0.000 am echten
   Methodik-Claim). Tabellen-Zeilen als Pseudo-Satz-Premise wurden empirisch
   verworfen: mDeBERTa entailt daraus VERTAUSCHTE Aussagen mit e≈0.97–0.99
   und widerspricht wahren mit c=0.996. Erfundene Zahlen (nicht im Fenster)
   bleiben `failed_entailment`; Claims ohne Zahlen (causal/attribution)
   erreichen diesen Zweig nie — der Zeitzonen-Pflichtfall bleibt gefangen.

**Batching-Entscheidung:** Alle NLI-Paare ALLER pending Claims werden in
EINEM `score_pairs`-Aufruf gesammelt (statt ein Call pro Claim) und danach
anhand eines Index-Mappings pro Claim zurücksortiert — spart wiederholte
Tokenizer-/Modell-Overhead-Kosten pro Note, `score_pairs` batched intern
ohnehin schon (siehe `nli.py`).

5. `failed = n_failed >= 1` — harter any-fail (der 1/12-Hrastinski-Fall muss
   blocken; `decompose_claims` liefert ohnehin nur High-Risk-Claims, ein
   Toleranz-Anteil ist nicht nötig).

`contra_threshold` bleibt als Parameter für die E5b-Kalibrierung dokumentiert
und überschreibbar, wird im MVP-Verdikt aber nicht als eigener Status
gebraucht (siehe Plan #69).

Rein pure Funktionen — keine Draft-Mutation, kein I/O außer den Modell-Loads
der importierten Bausteine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from generative.config import MDEBERTA_THRESHOLD_CONFIRMED, MDEBERTA_THRESHOLD_CONTRA
from generative.pipeline.anchor_patterns import PAGE_MARKER_LINE_RE
from generative.pipeline.attribution import check_attribution
from generative.pipeline.citation_check import _primary_surnames
from generative.pipeline.claims import (
    FOOTNOTE_MARKER_RE,
    PAGE_REF_RE,
    YEAR_TOKEN_RE,
    Claim,
    decompose_claims,
)
from generative.pipeline.embeddings import _model, _sentences, cosine
from generative.pipeline.nli import score_pairs
from generative.pipeline.page_index import claim_source_window
from generative.schemas.citation import CitationMeta


@dataclass(frozen=True)
class ClaimVerdict:
    claim: Claim
    # "supported" | "failed_attribution" | "failed_entailment"
    # | "abstain_no_window" | "abstain_nli" | "abstain_unverifiable_numbers"
    status: str
    evidence: str | None  # bester Stütz-Satz aus dem Fenster (bei NLI-Prüfung), sonst None
    entailment: float | None  # max-Entailment-Score, sonst None


@dataclass(frozen=True)
class GateResult:
    verdicts: list[ClaimVerdict]
    failed: bool  # True sobald >=1 Claim failed_* hat (any-high-risk-fail)
    n_supported: int
    n_failed: int
    n_abstained: int


def _window_sentence_embeddings(window_text: str, cache: dict | None = None):
    """Sätze + Embeddings eines Fensters — optional per-Lauf-gecacht.

    Mehrere Claims einer Note ankern oft auf derselben Seite; ohne Cache würde
    dasselbe Fenster pro Claim erneut durchs MiniLM encodet (Mistral-Review
    E5a, Performance). Der Cache lebt beim Aufrufer (ein Lauf), kein globaler
    Zustand — die Funktionen bleiben pure.
    """
    if cache is not None and window_text in cache:
        return cache[window_text]
    stripped = PAGE_MARKER_LINE_RE.sub("", window_text)
    sentences = _sentences(stripped)
    if not sentences:
        result = ([], None)
    else:
        embs = _model().encode(sentences, show_progress_bar=False, normalize_embeddings=True)
        result = (sentences, embs)
    if cache is not None:
        cache[window_text] = result
    return result


def top_k_sentences(window_text: str, query: str, k: int, *, _cache: dict | None = None) -> list[str]:
    """Rankt Sätze aus `window_text` per MiniLM-Cosine-Similarity zu `query`.

    Entfernt vorher `[S. N]`-Markerzeilen (`PAGE_MARKER_LINE_RE`) — sonst
    würden sie als eigener Pseudo-Satz mitgerankt. Nutzt `_model`/`_sentences`
    aus `embeddings.py` (Cross-Modul-Präzedenz bereits in `agents/planner.py`).
    """
    if not window_text or not query:
        return []
    sentences, sentence_embs = _window_sentence_embeddings(window_text, _cache)
    if not sentences:
        return []

    query_emb = _model().encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
    ranked = sorted(zip(sentences, sentence_embs), key=lambda pair: -cosine(pair[1], query_emb))
    return [sentence for sentence, _ in ranked[:k]]


# Seiten-Anker-Klammern im Claim-Text: "(S. 2)", "(zit. n. X, S. 2)". Für das
# NLI ist der Anker Metadatum, keine prüfbare Behauptung — mit Anker kippt
# mDeBERTa auf neutral (empirisch: identischer Claim e=0.998 ohne vs. e=0.002
# MIT "(S. 2)" — Kalibrierungs-Befund E5b, 2026-07-05).
_ANCHOR_PAREN_RE = re.compile(r"\s*\([^()]*?S\.\s*\d+[^()]*?\)")


def _nli_hypothesis(claim_text: str) -> str:
    """Claim-Text ohne Seiten-Anker-Klammern — die NLI-Hypothese."""
    return _ANCHOR_PAREN_RE.sub("", claim_text).strip()


def _claim_premises(window: str, claim: Claim, top_k: int, cache: dict | None = None) -> list[str]:
    """Top-k Fenster-Sätze plus kumulative Prefix-Konkatenationen top1..j.

    Prefix-Konkate statt nur der Voll-Konkatenation: Rausch-Sätze am Ende der
    Top-k verwässern das Entailment der Voll-Konkat (Kalibrierung E5b am
    Hrastinski-PDF: top1+top2 e=0.998, top1..3 e=0.993, top1..5 e=0.149 am
    identischen Claim) — die Präfixe geben legitimer Zwei-Satz-Synthese eine
    unverdünnte Premise. Max-über-Premises bleibt die Verdikt-Regel.
    """
    top_sentences = top_k_sentences(window, _nli_hypothesis(claim.text), top_k, _cache=cache)
    premises = list(top_sentences)
    for j in range(2, len(top_sentences) + 1):
        premises.append(" ".join(top_sentences[:j]))
    return premises


# Zahlen-Token für den Unverifizierbar-Abstain (Schritt 4b): Digit-Gruppen
# inkl. Tausender-/Dezimaltrenner ("1,507", "0,59") als EIN Token.
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _canonical_number(token: str) -> str:
    """Trenner-invariante Vergleichsform: "1,507" == "1.507" == "1507"."""
    return token.replace(".", "").replace(",", "")


def _numbers_present_in_window(claim_text: str, window: str) -> bool:
    """True wenn der Claim ≥1 echte Zahl trägt UND alle davon im Fenster stehen.

    Claim-Seite: Seiten-Refs, Jahre und Footnote-Marker werden wie in
    `claims.number_risk` maskiert — sonst würde jeder Claim abstainen, dessen
    Seitenzahl im Fenster auftaucht. Fenster-Seite: Digit-Tokens, die direkt
    an einem Buchstaben kleben ("hypothesis19" — Footnote-Superscript aus
    pdftotext), zählen nicht als Beleg — konservativ Richtung Fail.
    """
    masked = PAGE_REF_RE.sub("", claim_text)
    masked = YEAR_TOKEN_RE.sub("", masked)
    masked = FOOTNOTE_MARKER_RE.sub("", masked)
    claim_numbers = {_canonical_number(m.group(0)) for m in _NUMBER_TOKEN_RE.finditer(masked)}
    if not claim_numbers:
        return False

    window_numbers: set[str] = set()
    for match in _NUMBER_TOKEN_RE.finditer(window):
        before = window[match.start() - 1] if match.start() > 0 else ""
        after = window[match.end()] if match.end() < len(window) else ""
        if before.isalpha() or after.isalpha():
            continue
        window_numbers.add(_canonical_number(match.group(0)))

    return claim_numbers <= window_numbers


def run_faithfulness_gate(
    body: str,
    page_index: dict[int, str],
    citation: CitationMeta,
    *,
    neighbors: int = 1,
    top_k: int = 5,
    entail_threshold: float = MDEBERTA_THRESHOLD_CONFIRMED,
    contra_threshold: float = MDEBERTA_THRESHOLD_CONTRA,
) -> GateResult:
    """Kombiniert Claim-Dekomposition, Attribution-Check und NLI zu einem
    Verdikt pro High-Risk-Claim. Siehe Modul-Docstring für den vollen Ablauf."""
    claims = [c for c in decompose_claims(body) if not c.is_quote]
    if not claims:
        return GateResult(verdicts=[], failed=False, n_supported=0, n_failed=0, n_abstained=0)

    primary_surnames = _primary_surnames(citation.author) if citation is not None else []

    verdicts: list[ClaimVerdict | None] = [None] * len(claims)
    pending: list[int] = []
    windows: dict[int, str] = {}

    for i, claim in enumerate(claims):
        window = (
            claim_source_window(page_index, claim.anchor_page, neighbors) if claim.anchor_page is not None else None
        )
        if window is None:
            verdicts[i] = ClaimVerdict(claim, "abstain_no_window", None, None)
            continue

        if "attribution" in claim.risk_types:
            attribution_status = check_attribution(claim, window, primary_surnames)
            if attribution_status == "author_missing":
                verdicts[i] = ClaimVerdict(claim, "failed_attribution", None, None)
                continue
            # "supported"/"not_applicable" -> weiter zu Schritt 4 (Entailment)

        windows[i] = window
        pending.append(i)

    # Schritt 4: alle NLI-Paare ALLER pending Claims in EINEM score_pairs-Aufruf
    # sammeln (Batching-Entscheidung, siehe Modul-Docstring), danach über ein
    # Index-Mapping pro Claim zurücksortieren.
    claim_premises: dict[int, list[str]] = {}
    pair_owner: list[int] = []
    all_pairs: list[tuple[str, str]] = []

    window_cache: dict = {}
    for i in pending:
        claim = claims[i]
        premises = _claim_premises(windows[i], claim, top_k, cache=window_cache)
        claim_premises[i] = premises
        for premise in premises:
            pair_owner.append(i)
            all_pairs.append((premise, _nli_hypothesis(claim.text)))

    scores = score_pairs(all_pairs) if all_pairs else []

    for i in pending:
        claim = claims[i]
        premises = claim_premises[i]
        if not premises or scores is None:
            verdicts[i] = ClaimVerdict(claim, "abstain_nli", None, None)
            continue

        claim_scores = [scores[j] for j, owner in enumerate(pair_owner) if owner == i]
        best_pos = max(range(len(claim_scores)), key=lambda pos: claim_scores[pos].entailment)
        best_score = claim_scores[best_pos]
        best_premise = premises[best_pos]

        if best_score.entailment >= entail_threshold:
            verdicts[i] = ClaimVerdict(claim, "supported", best_premise, best_score.entailment)
        elif "number" in claim.risk_types and _numbers_present_in_window(claim.text, windows[i]):
            # Schritt 4b: Zahlen belegt, Aussage satz-NLI-unverifizierbar
            # (Tabellen-Fragmente/Absatz-Synthese) — Abstain, nie Fail.
            verdicts[i] = ClaimVerdict(claim, "abstain_unverifiable_numbers", best_premise, best_score.entailment)
        else:
            verdicts[i] = ClaimVerdict(claim, "failed_entailment", best_premise, best_score.entailment)

    final_verdicts = [v for v in verdicts if v is not None]
    n_supported = sum(1 for v in final_verdicts if v.status == "supported")
    n_failed = sum(1 for v in final_verdicts if v.status.startswith("failed_"))
    n_abstained = len(final_verdicts) - n_supported - n_failed

    return GateResult(
        verdicts=final_verdicts,
        failed=n_failed >= 1,
        n_supported=n_supported,
        n_failed=n_failed,
        n_abstained=n_abstained,
    )
