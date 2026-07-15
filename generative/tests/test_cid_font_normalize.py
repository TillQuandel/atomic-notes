# -*- coding: utf-8 -*-
"""Fix fuer #278: CID-Font-Artefakte im Jockisch (2010)-PDF korrumpieren den
Eval-Grounder-Quelltext -- quellentreue Notes werden faelschlich als
Halluzination markiert.

Root Cause (per PyMuPDF/fitz direkt am Original-PDF verifiziert, nicht
angenommen -- die urspruengliche Annahme "Whitespace-Kollaps" aus dem
Issue-Titel war beim Nachmessen nicht die tatsaechliche Ursache): Die
eingebettete Schriftart des PDFs mappt ihr Leerzeichen-Glyph auf U+0231
("ȱ") und ihr Trennstrich-Glyph auf U+022C ("Ȭ") statt auf ASCII-Space
bzw. ASCII-Hyphen. `eval_common._extract_page_text`/`_raw_page_lines` lesen
das buchstabengetreu -- der resultierende Text besteht fast nur noch aus
zusammengeschriebenen Woertern ("DieȱAkzeptanzȱistȱGegenstandȱ..."), wodurch
sowohl das Cosine-Chunk-Matching (`_retrieve_claim_contexts`) als auch die
Zitat-Fuzzy-Verifikation (`_verify_evidence_normalized`) an einer Quelle
scheitern, die das generierende LLM anhand sauber rekonstruierter Notes
korrekt gelesen hat. Repro-Runs (Halluzinationsrate 57,1%/42,9%, Coverage
0,0%, Opus-Tiefenreview bestaetigt ~100% Quellentreue bei 10/16/14 geprueften
Ankern): 20260714-185639, 20260714-215345.

Haeufigkeitsbeleg (ganzes PDF, PyMuPDF-Extraktion): U+0231 5173x, U+022C
240x -- alle anderen Nicht-ASCII-Zeichen (echte Umlaute, Anfuehrungszeichen,
eine Handvoll Formel-/Griechisch-Symbole je <=8x) sind gegenueber diesen zwei
dominanten Artefakten vernachlaessigbar und bleiben bewusst ungefixt (kein
messbarer Beitrag zum 0%-Coverage-Symptom).

Die Fragmente unten sind woertliche PyMuPDF-Extrakte aus
"Jockisch - 2010 - Das Technologieakzeptanzmodell.pdf" (S. 1 bzw. S. 7),
keine nachgebauten Beispiele.
"""

from generative.eval_common import _normalize
from generative.eval_quality_v4 import _normalize_for_evidence, _verify_evidence_normalized

CORRUPTED_SENTENCE = "DieȱAkzeptanzȱistȱGegenstandȱzahlreicherȱwissenschaftlicherȱUntersuchungen."
CLEAN_SENTENCE = "Die Akzeptanz ist Gegenstand zahlreicher wissenschaftlicher Untersuchungen."

# Zeilenumbruch-Trennung ("einfaȬ\nche") -- Trennstrich-Glyph ebenfalls korrumpiert,
# die bestehende Silbentrennungs-Normalisierung (ASCII "-\n") greift deshalb nicht.
CORRUPTED_HYPHENATION = "aufȱdieȱwahrgenommeneȱeinfaȬ\ncheȱ Bedienbarkeitȱ inȱ dasȱ Modellȱ integriert."
CLEAN_HYPHENATION = "auf die wahrgenommene einfache Bedienbarkeit in das Modell integriert."


def test_normalize_repairs_cid_space_artifact():
    assert _normalize(CORRUPTED_SENTENCE) == CLEAN_SENTENCE


def test_normalize_repairs_cid_hyphenation_artifact():
    assert _normalize(CORRUPTED_HYPHENATION) == CLEAN_HYPHENATION


def test_normalize_for_evidence_matches_corrupted_corpus():
    """Der eigentliche Bug: das LLM-generierte, saubere Zitat matcht nach der
    Normalisierung gegen den CID-korrumpierten Quelltext. Ohne Fix faellt der
    Score unter die 0.90-Schwelle und der Claim wird faelschlich als
    evidence_unverified/not_in_context gezaehlt (False-Positive-Halluzination,
    genau das in #278 belegte Symptom)."""
    corpus = _normalize_for_evidence(CORRUPTED_SENTENCE)
    verified, score = _verify_evidence_normalized(CLEAN_SENTENCE, corpus)
    assert verified is True, f"Erwartet verifiziert, war {verified} (Score={score})"


def test_normalize_leaves_clean_text_untouched():
    """Regressionsschutz: PDFs ohne CID-Artefakte duerfen durch den Fix nicht
    veraendert werden (die Artefakt-Codepoints kommen in gesundem Text nicht vor)."""
    clean = "Die Akzeptanz neuer Informationstechnologien wird durch mehrere Faktoren beeinflusst."
    assert _normalize(clean) == clean
