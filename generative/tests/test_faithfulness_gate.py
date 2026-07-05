"""Tests für die Gate-Logik des Faithfulness-Gates (Faithfulness-Gate E5a, #69).

Reine Kombination der vorhandenen Bausteine (page_index, claims, attribution,
nli) zu einem Verdikt pro Claim. Gemockt werden nur die zwei ML-Bausteine:
`score_pairs` (NLI) und das Satz-Retrieval `top_k_sentences` (MiniLM-Embedding) —
alles andere (Claim-Dekomposition, Attribution-Praesenz-Check, Seitenfenster)
ist deterministisch und laeuft echt mit, wie in den Nachbar-Testdateien
(`test_attribution.py`, `test_claims.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

from generative.pipeline import faithfulness_gate as fg
from generative.pipeline.nli import NliScores
from generative.schemas.citation import CitationMeta


def _citation(author: str | None = "Hrastinski", source_file: str = "hrastinski.pdf") -> CitationMeta:
    return CitationMeta(author=author, year="2008", title=None, doi=None, source_file=source_file)


@pytest.fixture
def stub_top_k_sentences(monkeypatch):
    """Default-Stub: entfernt `[S. N]`-Markerzeilen (wie die echte Funktion) und
    liefert die ersten `k` Saetze aus dem Fenster (Substring-Split auf Punkt) —
    verhindert echten Modell-Load in Tests, die die Ranking-Details nicht selbst
    prüfen. NICHT autouse: `TestTopKSentencesHelper` testet die echte Funktion
    und darf sie nicht ueberschrieben bekommen."""

    def _fake(window_text: str, query: str, k: int, **_kw) -> list[str]:
        stripped = fg.PAGE_MARKER_LINE_RE.sub("", window_text)
        sents = [s.strip() for s in stripped.replace("\n", " ").split(".") if s.strip()]
        return sents[:k]

    monkeypatch.setattr(fg, "top_k_sentences", _fake)
    return _fake


# ---- Attribution-Fail --------------------------------------------------------


class TestFailedAttribution:
    def test_foreign_author_missing_from_window_fails_attribution(self):
        body = "Laut Haythornthwaite sind drei Kommunikationstypen zentral (S. 2)."
        page_index = {2: "Drei Kommunikationstypen werden anhand ihrer Synchronitaet unterschieden."}
        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        assert result.verdicts[0].status == "failed_attribution"
        assert result.verdicts[0].evidence is None
        assert result.verdicts[0].entailment is None
        assert result.failed is True
        assert result.n_failed == 1


# ---- Entailment-Fail (number-Risk, neutral-dominant) -------------------------


class TestFailedEntailment:
    def test_number_risk_neutral_dominant_fails_entailment(self, monkeypatch, stub_top_k_sentences):
        body = "Die Effektstaerke lag bei r = 0,59 (S. 12)."
        page_index = {12: "Die Studie berichtet einen kleinen bis mittleren Zusammenhang."}

        monkeypatch.setattr(fg, "score_pairs", lambda pairs, batch_size=32: [NliScores(0.05, 0.9, 0.05)])

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "failed_entailment"
        assert v.entailment == pytest.approx(0.05)
        assert result.failed is True


# ---- Entailment-Support -------------------------------------------------------


class TestSupported:
    def test_high_entailment_is_supported_with_evidence(self, monkeypatch, stub_top_k_sentences):
        body = "Die Rueckmeldung verursacht eine hoehere Motivation (S. 4)."
        page_index = {4: "Regelmaessige Rueckmeldung fuehrt zu hoeherer Lernmotivation."}

        monkeypatch.setattr(fg, "score_pairs", lambda pairs, batch_size=32: [NliScores(0.92, 0.05, 0.03)])

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "supported"
        assert v.entailment == pytest.approx(0.92)
        assert v.evidence is not None
        assert result.failed is False
        assert result.n_supported == 1

    def test_entailment_only_in_concat_premise_is_supported_synthesis_case(self, monkeypatch):
        # Zwei Einzelsaetze im Fenster, beide schwach fuer sich, aber zusammen
        # (Konkat-Premise) tragen sie den Claim -- Synthese-Fall aus dem Plan.
        body = "Blended Learning ist effektiver als reines Praesenzlernen (S. 7)."
        page_index = {7: "Satz eins gibt Kontext. Satz zwei liefert den Beleg."}

        def _two_sentences(window_text, query, k, **_kw):
            return ["Satz eins gibt Kontext", "Satz zwei liefert den Beleg"]

        monkeypatch.setattr(fg, "top_k_sentences", _two_sentences)

        # Reihenfolge der Paare: sent1, sent2, konkat -- nur die Konkat-Premise
        # (letzter Eintrag) hat hohes Entailment.
        def _fake_score_pairs(pairs, batch_size=32):
            assert len(pairs) == 3
            return [NliScores(0.1, 0.8, 0.1), NliScores(0.2, 0.7, 0.1), NliScores(0.85, 0.1, 0.05)]

        monkeypatch.setattr(fg, "score_pairs", _fake_score_pairs)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "supported"
        assert v.entailment == pytest.approx(0.85)
        assert v.evidence == "Satz eins gibt Kontext Satz zwei liefert den Beleg"
        assert result.failed is False


# ---- Abstain: kein Quellfenster ----------------------------------------------


class TestAbstainNoWindow:
    def test_anchor_page_none_abstains_without_failing(self):
        body = "Die Rueckmeldung verursacht eine hoehere Motivation."
        page_index = {4: "irrelevanter Fenstertext"}
        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "abstain_no_window"
        assert v.evidence is None
        assert v.entailment is None
        assert result.failed is False
        assert result.n_abstained == 1

    def test_anchor_page_not_in_index_abstains_without_failing(self):
        body = "Die Rueckmeldung verursacht eine hoehere Motivation (S. 99)."
        page_index = {4: "irrelevanter Fenstertext"}
        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        assert result.verdicts[0].status == "abstain_no_window"
        assert result.failed is False


# ---- Abstain: NLI nicht verfuegbar --------------------------------------------


class TestAbstainNli:
    def test_score_pairs_none_abstains_without_failing(self, monkeypatch, stub_top_k_sentences):
        body = "Die Rueckmeldung verursacht eine hoehere Motivation (S. 4)."
        page_index = {4: "Regelmaessige Rueckmeldung foerdert die Motivation."}

        monkeypatch.setattr(fg, "score_pairs", lambda pairs, batch_size=32: None)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "abstain_nli"
        assert v.evidence is None
        assert v.entailment is None
        assert result.failed is False
        assert result.n_abstained == 1


# ---- Blockquote wird uebersprungen --------------------------------------------


class TestBlockquoteSkipped:
    def test_blockquote_claim_produces_no_verdict(self):
        body = "> Laut Haythornthwaite sind drei Kommunikationstypen zentral (S. 2)."
        page_index = {2: "Drei Kommunikationstypen werden unterschieden."}
        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert result.verdicts == []
        assert result.failed is False
        assert result.n_supported == 0
        assert result.n_failed == 0
        assert result.n_abstained == 0


# ---- Any-High-Risk-Fail: 1 von 12 muss blocken --------------------------------


class TestAnyFail:
    def test_twelve_claims_one_failed_blocks_the_gate(self, monkeypatch, stub_top_k_sentences):
        # 11 unproblematische number-Claims (hohes Entailment) + 1 Claim mit
        # fehlender Attribution -- der 1/12-Hrastinski-Fall aus dem Plan.
        ok_sentences = [f"Die Kennzahl {i} lag bei {i} Prozent (S. {i})." for i in range(1, 12)]
        fail_sentence = "Laut Haythornthwaite sind drei Kommunikationstypen zentral (S. 2)."
        body = " ".join(ok_sentences) + " " + fail_sentence

        page_index = {i: f"Kontexttext zur Kennzahl {i}." for i in range(1, 12)}
        page_index[2] = "Drei Kommunikationstypen werden anhand ihrer Synchronitaet unterschieden."

        monkeypatch.setattr(fg, "score_pairs", lambda pairs, batch_size=32: [NliScores(0.9, 0.05, 0.05)] * len(pairs))

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 12
        assert result.n_failed == 1
        assert result.failed is True


# ---- Leerer Body ---------------------------------------------------------------


class TestEmptyBody:
    def test_empty_body_yields_no_verdicts_and_not_failed(self):
        result = fg.run_faithfulness_gate("", {}, _citation())

        assert result.verdicts == []
        assert result.failed is False
        assert result.n_supported == 0
        assert result.n_failed == 0
        assert result.n_abstained == 0


# ---- Schwellen ueberschreibbar --------------------------------------------------


class TestThresholdOverride:
    def test_entail_threshold_override_turns_08_claim_into_failed(self, monkeypatch, stub_top_k_sentences):
        body = "Die Rueckmeldung verursacht eine hoehere Motivation (S. 4)."
        page_index = {4: "Regelmaessige Rueckmeldung foerdert die Motivation."}

        monkeypatch.setattr(fg, "score_pairs", lambda pairs, batch_size=32: [NliScores(0.8, 0.15, 0.05)])

        default_result = fg.run_faithfulness_gate(body, page_index, _citation())
        assert default_result.verdicts[0].status == "supported"

        strict_result = fg.run_faithfulness_gate(body, page_index, _citation(), entail_threshold=0.9)
        assert strict_result.verdicts[0].status == "failed_entailment"
        assert strict_result.failed is True


# ---- top_k_sentences: eigener Helper (Marker-Strip + Ranking) ------------------


class _FakeEmbedModel:
    """Ordnet jedem Satz einen 1-D-Fake-Embedding-Wert nach einer festen Tabelle
    zu, damit die Rangfolge deterministisch pruefbar ist -- kein echtes Modell."""

    def __init__(self, table: dict[str, float]):
        self._table = table

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        return np.array([[self._table.get(t, 0.0)] for t in texts])


class TestTopKSentencesHelper:
    def test_strips_page_marker_lines_before_ranking(self, monkeypatch):
        # `_sentences` (embeddings.py) behaelt den Satzpunkt am Ende — die
        # Tabellen-Keys muessen ihn deshalb enthalten.
        table = {"Erster Satz.": 0.9, "Zweiter Satz.": 0.1, "QUERY": 1.0}
        monkeypatch.setattr(fg, "_model", lambda: _FakeEmbedModel(table))

        window = "[S. 4]\nErster Satz. Zweiter Satz."
        result = fg.top_k_sentences(window, "QUERY", k=2)

        assert result == ["Erster Satz.", "Zweiter Satz."]
        assert not any("[S." in s for s in result)

    def test_returns_at_most_k_sentences_ranked_by_similarity(self, monkeypatch):
        table = {"A.": 0.1, "B.": 0.9, "C.": 0.5, "QUERY": 1.0}
        monkeypatch.setattr(fg, "_model", lambda: _FakeEmbedModel(table))

        window = "A. B. C."
        result = fg.top_k_sentences(window, "QUERY", k=1)

        assert result == ["B."]


def test_window_embeddings_cached_per_run(monkeypatch):
    # Mistral-Review E5a: zwei Claims auf derselben Seite duerfen das Fenster
    # nur EINMAL encoden (per-Lauf-Cache), nicht pro Claim erneut.
    from generative.pipeline import faithfulness_gate as fg

    encode_calls = []

    class _FakeEmbModel:
        def encode(self, texts, **kw):
            encode_calls.append(list(texts))
            import numpy as np

            return [np.array([1.0, 0.0])] * len(texts)

    monkeypatch.setattr(fg, "_model", lambda: _FakeEmbModel())
    monkeypatch.setattr(fg, "_sentences", lambda t: [s for s in t.split(". ") if s])

    cache = {}
    fg.top_k_sentences("Satz eins. Satz zwei.", "query A", 2, _cache=cache)
    fg.top_k_sentences("Satz eins. Satz zwei.", "query B", 2, _cache=cache)

    window_encodes = [c for c in encode_calls if len(c) > 1]
    assert len(window_encodes) == 1, "Fenster-Saetze wurden mehrfach encodet trotz Cache"


def test_nli_hypothesis_strips_page_anchors():
    # Kalibrierungs-Befund E5b: "(S. 2)" im Hypothesen-Text kippt mDeBERTa auf
    # neutral (e=0.998 ohne vs. 0.002 mit Anker am identischen Claim).
    from generative.pipeline.faithfulness_gate import _nli_hypothesis

    assert _nli_hypothesis("Es wurden 12 Interviews gefuehrt (S. 2).") == "Es wurden 12 Interviews gefuehrt."
    assert (
        _nli_hypothesis("X betont Y (zit. n. Hrastinski, S. 2). Und Z (S. 5-8) bleibt.") == "X betont Y. Und Z bleibt."
    )
    assert _nli_hypothesis("Ohne Anker bleibt alles.") == "Ohne Anker bleibt alles."


def test_wegen_triggers_causal_risk():
    from generative.pipeline.claims import causal_risk

    assert causal_risk("Treffen sind wegen zeitzonenbedingter Verpflichtungen nicht planbar.")


# ---- Fix 1 (E5b): Prefix-Konkatenationen als Premises --------------------------


class TestPrefixConcatPremises:
    def test_top1_top2_concat_rescues_claim_that_full_concat_dilutes(self, monkeypatch):
        # Kalibrierungs-Befund E5b (Hrastinski FP-A): top1+top2 entailen mit
        # e=0.998, die Voll-Konkatenation aller 5 verwaessert auf e=0.149.
        # Nur die Prefix-Konkat "S1 S2" traegt hier — weder Einzelsaetze noch
        # die Voll-Konkat mit dem Rausch-Satz S3.
        body = "Die Analyse zeigt, dass 90 Prozent der Saetze inhaltsbezogen waren (S. 3)."
        page_index = {3: "S1. S2. S3."}

        monkeypatch.setattr(fg, "top_k_sentences", lambda w, q, k, **_kw: ["S1", "S2", "S3"])

        def _score_by_premise(pairs, batch_size=32):
            return [
                NliScores(0.95, 0.03, 0.02) if premise == "S1 S2" else NliScores(0.1, 0.88, 0.02)
                for premise, _hyp in pairs
            ]

        monkeypatch.setattr(fg, "score_pairs", _score_by_premise)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "supported"
        assert v.evidence == "S1 S2"
        assert v.entailment == pytest.approx(0.95)


# ---- Fix 2 (E5b): abstain_unverifiable_numbers ---------------------------------


class TestAbstainUnverifiableNumbers:
    def _low_nli(self, monkeypatch):
        monkeypatch.setattr(fg, "score_pairs", lambda pairs, batch_size=32: [NliScores(0.05, 0.9, 0.05)] * len(pairs))

    def test_numbers_present_in_table_fragments_abstain_instead_of_fail(self, monkeypatch, stub_top_k_sentences):
        # Hrastinski FP-B: exakte Prozentwerte stehen nur in pdftotext-
        # Tabellen-Fragmenten — Satz-NLI kann sie nicht verifizieren.
        body = "Synchrone Beitraege entfallen zu 58 % auf Inhalte und 34 % auf Planung (S. 4)."
        page_index = {4: "Almost 60 percent related to content.\n876  (58%)\n507  (34%)"}
        self._low_nli(monkeypatch)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.status == "abstain_unverifiable_numbers"
        assert result.failed is False
        assert result.n_abstained == 1
        assert result.n_failed == 0

    def test_fabricated_number_missing_from_window_still_fails(self, monkeypatch, stub_top_k_sentences):
        body = "Synchrone Beitraege entfallen zu 77 % auf Inhalte (S. 4)."
        page_index = {4: "Almost 60 percent related to content.\n876  (58%)\n507  (34%)"}
        self._low_nli(monkeypatch)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert result.verdicts[0].status == "failed_entailment"
        assert result.failed is True

    def test_one_of_two_numbers_missing_still_fails(self, monkeypatch, stub_top_k_sentences):
        # ALLE Claim-Zahlen muessen belegt sein — eine erfundene reicht zum Fail.
        body = "Die Verteilung lag bei 58 % zu 77 % (S. 4)."
        page_index = {4: "876  (58%)"}
        self._low_nli(monkeypatch)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert result.verdicts[0].status == "failed_entailment"

    def test_causal_claim_without_numbers_still_fails(self, monkeypatch, stub_top_k_sentences):
        # Zeitzonen-Pflichtfall-Schutz: causal-Claims ohne number-Risk duerfen
        # nie in den Zahlen-Abstain rutschen.
        body = "Treffen sind wegen zeitzonenbedingter Verpflichtungen nicht planbar (S. 4)."
        page_index = {4: "When synchronous meetings cannot be scheduled. 876  (58%)"}
        self._low_nli(monkeypatch)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        assert result.verdicts[0].status == "failed_entailment"
        assert result.failed is True

    def test_page_refs_years_footnotes_masked_before_number_check(self, monkeypatch, stub_top_k_sentences):
        # Seitenanker "S. 12", Jahr "(2008)" und Footnote-Marker duerfen die
        # Zahlen-Menge nicht befuellen — sonst wuerde jeder Claim abstainen,
        # dessen Seitenzahl im Fenster steht.
        # "bereits 2008" bewusst kleingeschrieben davor — ein grossgeschriebenes
        # Wort vor der Jahreszahl wuerde AUTHOR_YEAR_RE (Attribution) triggern.
        body = "Es wurden bereits 2008 insgesamt 42 Faelle gezaehlt (S. 12)."
        page_index = {12: "Im Jahr 2008 auf Seite 12 steht nichts von der Zahl."}
        self._low_nli(monkeypatch)

        result = fg.run_faithfulness_gate(body, page_index, _citation())

        # 42 fehlt im Fenster -> failed, obwohl 2008 und 12 dort stehen
        assert result.verdicts[0].status == "failed_entailment"


class TestClaimNumbersHelper:
    def test_separator_normalization_matches_thousands_variants(self):
        assert fg._numbers_present_in_window("Es wurden 1.507 Saetze analysiert.", "insgesamt 1,507 (100%)")

    def test_digit_glued_to_letter_does_not_count_as_presence(self):
        # Footnote-Superscript "hypothesis19" belegt keine 19 — konservativ
        # Richtung Fail, nicht Richtung Abstain.
        assert not fg._numbers_present_in_window("Es gab 19 Interviews.", "media naturalness hypothesis19 predicts")

    def test_claim_without_extractable_numbers_returns_false(self):
        assert not fg._numbers_present_in_window("Die Studie (2008) belegt dies (S. 4).", "876 (58%)")
