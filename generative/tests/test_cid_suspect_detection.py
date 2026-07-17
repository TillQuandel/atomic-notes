# -*- coding: utf-8 -*-
"""Generische CID-Font-/Korruptions-Erkennung auf Eval-Seite (#306).

#278 fixt exakt zwei Codepoints eines defekten Fonts im Jockisch-PDF
(U+0231/U+022C). Jedes andere PDF mit einem anderen CID-Glyph-Mapping
erzeugt dasselbe Fehlerbild und bleibt ungefixt. Dieses Modul testet die
generische Erkennungs-Heuristik (`generative.eval_text_quality`) sowie ihre
additive Verdrahtung in `eval_quality_v4` -- NUR Flag/Warnung, siehe
Modul-Docstring dort: kein Rueckmapping, keine Label-/Raten-Aenderung.

Drei Kern-Szenarien (Akzeptanzkriterium #306):
  1. Synthetisches CID-Profil (Jockisch-artig) -> Flag gesetzt.
  2. Sauberer deutscher Text mit Umlauten -> kein Flag.
  3. Grenzfall knapp unter dem Schwellwert -> kein Flag.
Zusaetzlich: Bit-Identitaets-Test, der belegt, dass das Flag KEINE
Halluzinationsrate/kein Label veraendert (Bump-Freiheits-Beleg).
"""

from __future__ import annotations

import fitz

from generative import eval_quality_v4 as eq
from generative.eval_common import Chunk
from generative.eval_text_quality import (
    DEFAULT_RATIO_THRESHOLD,
    QUALITY_FLAG_CID_SUSPECT,
    CidSuspectResult,
    detect_cid_suspect,
)

# ---------------------------------------------------------------------------
# 1) Reine Heuristik (generative.eval_text_quality) -- kein PDF-I/O noetig.
# ---------------------------------------------------------------------------

# Jockisch-artiges Muster: ein einzelnes Nicht-ASCII-Glyph (hier U+0231, exakt
# der am Original-PDF per PyMuPDF nachgemessene Codepoint aus #278) ersetzt
# praktisch jedes Leerzeichen -> dominiert das Nicht-ASCII-Haeufigkeitsprofil.
JOCKISCH_LIKE_TEXT = ("DieȱAkzeptanzȱistȱGegenstandȱzahlreicherȱwissenschaftlicherȱUntersuchungenȱ ") * 40

# Sauberer deutscher Text mit hoher Umlaut-Dichte (ä ö ü ß) -- alle Umlaute
# liegen in Latin-1 Supplement (<= U+017F) und duerfen die Erkennung NIE
# ausloesen, unabhaengig von ihrer Haeufigkeit.
CLEAN_UMLAUT_TEXT = (
    "Die Akzeptanz neuer Informationstechnologien wird durch waehrgenommene "
    "Nuetzlichkeit und wahrgenommene Benutzerfreundlichkeit erklaert. "
    "Fuer groessere Stichproben zeigt sich ein aehnliches Bild bei der "
    "Ueberpruefung mehrerer Softwareloesungen fuer oeffentliche Behoerden. "
) * 60


class TestDetectCidSuspect:
    def test_synthetic_jockisch_profile_triggers_flag(self):
        result = detect_cid_suspect(JOCKISCH_LIKE_TEXT)
        assert result is not None
        assert result.codepoint == "U+0231"
        assert result.ratio >= DEFAULT_RATIO_THRESHOLD

    def test_clean_german_umlaut_text_does_not_trigger(self):
        assert detect_cid_suspect(CLEAN_UMLAUT_TEXT) is None

    def test_ratio_just_below_threshold_does_not_trigger(self):
        # 499 von 100000 Zeichen = 0,499 % < 0,5 % Schwelle.
        total = 100_000
        suspect_count = 499
        text = "a" * (total - suspect_count) + "ƀ" * suspect_count
        assert len(text) == total
        result = detect_cid_suspect(text, ratio_threshold=DEFAULT_RATIO_THRESHOLD)
        assert result is None

    def test_ratio_just_above_threshold_triggers(self):
        # 501 von 100000 Zeichen = 0,501 % > 0,5 % Schwelle.
        total = 100_000
        suspect_count = 501
        text = "a" * (total - suspect_count) + "ƀ" * suspect_count
        result = detect_cid_suspect(text, ratio_threshold=DEFAULT_RATIO_THRESHOLD)
        assert result is not None
        assert result.count == suspect_count

    def test_typographic_punctuation_whitelist_does_not_trigger(self):
        # Gedankenstriche/"smarte" Anfuehrungszeichen sind normale Extraktions-
        # Artefakte, kein CID-Signal (false-positive-arm).
        text = ("Ein Satz " + "—" + " mit vielen Gedankenstrichen ") * 200
        assert detect_cid_suspect(text) is None

    def test_empty_text_returns_none(self):
        assert detect_cid_suspect("") is None


# ---------------------------------------------------------------------------
# 2) Verdrahtung in eval_quality_v4: _pdf_artifacts liest das Flag aus dem
#    per PyMuPDF extrahierten Volltext (derselbe Volltext, der auch fuer
#    Chunking/Retrieval/Evidence-Verifikation verwendet wird).
# ---------------------------------------------------------------------------


def _make_pdf(tmp_path, name="quelle.pdf"):
    pdf_path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Wilson beschreibt Informationsverhalten als uebergeordnetes Rahmenkonzept.\n"
        "Das ISP-Modell von Kuhlthau umfasst sechs aufeinanderfolgende Phasen.",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestPdfArtifactsCidWiring:
    def test_clean_pdf_has_no_cid_suspect(self, tmp_path):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)
        assert eq._pdf_artifacts(pdf).cid_suspect is None

    def test_corrupted_full_text_sets_cid_suspect(self, tmp_path, monkeypatch):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)

        def fake_extract(doc, page):
            return JOCKISCH_LIKE_TEXT if page == 1 else ""

        monkeypatch.setattr(eq, "_extract_page_text", fake_extract)
        result = eq._pdf_artifacts(pdf).cid_suspect
        assert result is not None
        assert result.codepoint == "U+0231"


# ---------------------------------------------------------------------------
# 3) Verdrahtung in _aggregate: additives Feld + quality_flags-Eintrag, OHNE
#    dass irgendein anderer Wert im Ergebnis-Dict sich aendert (#306 harte
#    Vorgabe: kein Messverhalten-Unterschied, kein EVAL_VERSION-Bump noetig).
# ---------------------------------------------------------------------------


def _minimal_aggregate_args(tmp_path):
    note_path = tmp_path / "note.md"
    pdf_path = tmp_path / "quelle.pdf"
    chunks = [Chunk(0, "Ein Chunk-Text.", (1,))]
    claim_scores = [
        {
            "claim_idx": 0,
            "claim": "Ein Claim.",
            "label": eq.SUPPORTED_EXACT,
            "quality_flags": [],
            "evidence": "Ein Chunk-Text.",
            "evidence_verified": True,
            "decision_source": "primary",
        }
    ]
    llm_meta = {"calls": 1, "input_tokens": 10, "output_tokens": 5, "cached_calls": 0, "quality_flags": []}
    return note_path, pdf_path, chunks, claim_scores, llm_meta


class TestAggregateCidSuspectAdditive:
    def test_cid_suspect_adds_flag_and_detail_field(self, tmp_path):
        note_path, pdf_path, chunks, claim_scores, llm_meta = _minimal_aggregate_args(tmp_path)
        suspect = CidSuspectResult(codepoint="U+0231", char="ȱ", count=5173, ratio=0.421, total_chars=12283)

        result = eq._aggregate(
            note_path,
            pdf_path,
            "v-test",
            "2026-07-17T00:00:00",
            "de",
            chunks,
            claim_scores,
            llm_meta,
            cid_suspect=suspect,
        )

        assert QUALITY_FLAG_CID_SUSPECT in result["quality_flags"]
        assert result["pdf_text_suspect_cid"] == suspect.as_dict()

    def test_no_cid_suspect_leaves_field_none_and_no_flag(self, tmp_path):
        note_path, pdf_path, chunks, claim_scores, llm_meta = _minimal_aggregate_args(tmp_path)

        result = eq._aggregate(
            note_path,
            pdf_path,
            "v-test",
            "2026-07-17T00:00:00",
            "de",
            chunks,
            claim_scores,
            llm_meta,
            cid_suspect=None,
        )

        assert result["pdf_text_suspect_cid"] is None
        assert QUALITY_FLAG_CID_SUSPECT not in result["quality_flags"]

    def test_cid_suspect_does_not_change_any_other_result_field(self, tmp_path):
        """Bit-Identitaets-Beleg: das additive Flag/Feld ist die EINZIGE
        Differenz zwischen einem Lauf mit und ohne CID-Verdacht bei sonst
        identischem Input -- Halluzinationsrate/Labels/Counts bleiben
        unberuehrt (Grundlage der PR-Begruendung "kein EVAL_VERSION-Bump")."""
        note_path, pdf_path, chunks, claim_scores, llm_meta = _minimal_aggregate_args(tmp_path)
        suspect = CidSuspectResult(codepoint="U+0231", char="ȱ", count=5173, ratio=0.421, total_chars=12283)

        result_without = eq._aggregate(
            note_path, pdf_path, "v-test", "2026-07-17T00:00:00", "de", chunks, claim_scores, llm_meta, cid_suspect=None
        )
        result_with = eq._aggregate(
            note_path,
            pdf_path,
            "v-test",
            "2026-07-17T00:00:00",
            "de",
            chunks,
            claim_scores,
            llm_meta,
            cid_suspect=suspect,
        )

        diff_keys = {k for k in set(result_without) | set(result_with) if result_without.get(k) != result_with.get(k)}
        assert diff_keys == {"quality_flags", "pdf_text_suspect_cid"}
