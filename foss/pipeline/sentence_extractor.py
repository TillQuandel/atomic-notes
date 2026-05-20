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


def strip_anchors(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


def find_concept_sentences(
    concept: str, text: str, context: int = 1, threshold: int = 70
) -> list[str]:
    """Findet Saetze mit Konzept-Erwaehnung + Kontext-Fenster."""
    sentences = sent_tokenize(text)
    result, seen = [], set()
    for i, sent in enumerate(sentences):
        if fuzz.partial_ratio(concept.lower(), sent.lower()) >= threshold:
            for j in range(max(0, i - context), min(len(sentences), i + context + 1)):
                if j not in seen:
                    seen.add(j)
                    result.append(sentences[j])
    return result


def extract_body_for_concept(
    concept: str, text: str, n: int = 4, language: str = "english"
) -> list[str]:
    """LexRank ueber Konzept-Sentence-Cluster -> Top-n Saetze."""
    sentences = find_concept_sentences(concept, text) or sent_tokenize(text)
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
