from __future__ import annotations
import re
from rapidfuzz import fuzz

try:
    from nltk.tokenize import sent_tokenize
except Exception:
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

_ANCHOR_RE = re.compile(r"\s*\(S\.\s*\d+(?:-\d+)?\)")
_CITATION_FRAG_RE = re.compile(r"^[;,\s]*[A-Z][a-z]+,?\s+[A-Z]\..*?\(S\.\s*\d+\)\s*$")
_MERGED_WORDS_RE = re.compile(r"([a-z])([A-Z])")
_BIBREF_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")  # [21] [66] [21,22] strippen
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s*\n?\s*(\w+)")  # "interac- tion" -> "interaction"


def strip_anchors(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


# #167b: langdetect-ISO-Code -> sumy-Tokenizer-Sprachname. Fallback english,
# damit LexRank auch bei nicht abgedeckten/unklaren Sprachen laeuft.
_SUMY_LANGUAGES = {"en": "english", "de": "german"}


def sumy_language(code: str) -> str:
    """Mappt den langdetect-Code auf den sumy-Tokenizer-Sprachnamen."""
    return _SUMY_LANGUAGES.get((code or "").lower(), "english")


_NON_MATCH_RE = re.compile(r"[^0-9a-zäöüß ]+")


def _normalize_for_match(text: str) -> str:
    """Vereinheitlicht Text fuer den Seitenabgleich (Silbentrennung, Kleinschreibung)."""
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text.lower())
    return re.sub(r"\s+", " ", _NON_MATCH_RE.sub(" ", text)).strip()


def map_sentences_to_pages(
    sentences: list[str], page_texts: list[str], fallback_page: int = 1, threshold: int = 85
) -> list[int]:
    """Ordnet jeden Satz der physischen Seite zu, deren Text ihn am besten enthaelt.

    Nutzt den vorhandenen Seitenindex (`page_texts[i]` = physische Seite i+1) und
    einen Fuzzy-Abgleich ueber ein distinktes Satzpraefix (#167c). Faellt ein Satz
    unter `threshold`, wird `fallback_page` (i.d.R. die Chunk-Seite) genutzt.
    """
    norm_pages = [(i + 1, _normalize_for_match(pt)) for i, pt in enumerate(page_texts) if pt.strip()]
    result: list[int] = []
    for sent in sentences:
        probe = _normalize_for_match(strip_anchors(sent))[:60]
        best_page, best_score = fallback_page, 0.0
        if probe:
            for page_no, npage in norm_pages:
                score = fuzz.partial_ratio(probe, npage)
                if score > best_score:
                    best_score, best_page = score, page_no
        result.append(best_page if best_score >= threshold else fallback_page)
    return result


def clean_sentence(text: str) -> str:
    """Bereinigt PDF-Extraktions-Artefakte."""
    # Silbentrennung zusammenfuehren: "interac- tion" -> "interaction"
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    # Bibliografische Referenz-Nummern entfernen: [21], [66], [21,22]
    text = _BIBREF_RE.sub("", text)
    # Zitationsfragmente wie "; Allard, S.L. (S. 8)" entfernen
    if _CITATION_FRAG_RE.match(text.strip()):
        return ""
    # Sehr kurze Fragmente (< 20 Zeichen ohne Anker) skippen
    if len(strip_anchors(text).strip()) < 20:
        return ""
    return text.strip()


def find_concept_sentences(concept: str, text: str, context: int = 1, threshold: int = 70) -> list[str]:
    """Findet Saetze mit Konzept-Erwaehnung + Kontext-Fenster. Bereinigt Artefakte."""
    sentences = sent_tokenize(text)
    result, seen = [], set()
    for i, sent in enumerate(sentences):
        if fuzz.partial_ratio(concept.lower(), sent.lower()) >= threshold:
            for j in range(max(0, i - context), min(len(sentences), i + context + 1)):
                if j not in seen:
                    seen.add(j)
                    cleaned = clean_sentence(sentences[j])
                    if cleaned:
                        result.append(cleaned)
    return result


def extract_body_for_concept(concept: str, text: str, n: int = 4, language: str = "english") -> list[str]:
    """LexRank ueber Konzept-Sentence-Cluster -> Top-n Saetze.
    Gibt leere Liste zurueck wenn kein Satz das Konzept erwaehnt (kein Fallback auf Volltext).
    """
    sentences = find_concept_sentences(concept, text)
    if not sentences:
        return []
    if len(sentences) <= n:
        return sentences
    cluster = " ".join(sentences)
    parser = PlaintextParser.from_string(cluster, Tokenizer(language))
    return [str(s) for s in LexRankSummarizer()(parser.document, sentences_count=n)]


def add_page_anchors(sentences: list[str], pages: list[int]) -> list[str]:
    """Fuegt (S. N)-Anker an jeden Satz (strippt vorherige Anker)."""
    result = []
    for i, sent in enumerate(sentences):
        page = pages[min(i, len(pages) - 1)] if pages else 1
        result.append(f"{strip_anchors(sent)} (S. {page})")
    return result
