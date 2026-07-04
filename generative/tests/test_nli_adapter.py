"""Tests für den Shared NLI-Adapter (mDeBERTa/XNLI, Faithfulness-Gate E4a, #69).

Mechanik-Tests mit gemocktem Modell (kein echter Download): Batching (ein
Modell-Call für N Paare), Label-Mapping über `model.config.id2label` (robust
gegen vertauschte Label-Reihenfolge — der Kommentar in `eval_quality.py`
"Standard bei Laurer-Modellen" ist ausdrücklich KEINE Garantie), Abstain-Pfad
(Loader schlägt fehl → None, kein Crash, kein Log-Spam), Truncation-Parameter.

Ein optionaler `@pytest.mark.slow`-Test mit echtem Modell-Download existiert
bewusst NICHT — pytest.ini deselektiert `slow` in der Kanon-Suite ohnehin,
und ein Download in CI wäre nicht deterministisch.
"""

from __future__ import annotations

import sys
import types

from generative.pipeline import nli


class _FakeProbs(list):
    """Stand-in für `torch.softmax(...).tolist()` — Ergebnis ist hier bereits
    eine Liste von Wahrscheinlichkeits-Zeilen, `.tolist()` ist ein No-Op."""

    def tolist(self):
        return list(self)


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _FakeNoGrad()

    @staticmethod
    def softmax(logits, dim=-1):
        return _FakeProbs(logits)


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeModel:
    """Liefert pro Aufruf feste `logits` (bereits softmax-Wahrscheinlichkeiten,
    da `_FakeTorch.softmax` ein No-Op ist) und zählt die Calls (Batching-Nachweis)."""

    def __init__(self, id2label, rows):
        self.config = _FakeConfig(id2label)
        self.rows = rows
        self.calls: list[dict] = []

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return _FakeOutputs(self.rows)


class _FakeTokenizer:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.calls.append({"premises": premises, "hypotheses": hypotheses, **kwargs})
        return {"input_ids": premises, "attention_mask": hypotheses}


def _reset_cache(monkeypatch):
    monkeypatch.setattr(nli, "_MODEL_CACHE", {})


def _install_fake_loader(monkeypatch, id2label, rows):
    tokenizer = _FakeTokenizer()
    model = _FakeModel(id2label, rows)
    monkeypatch.setattr(nli, "_import_and_load", lambda: (tokenizer, model, _FakeTorch))
    return tokenizer, model


# ---- score_pairs: Batching ---------------------------------------------------


def test_score_pairs_makes_a_single_model_call_for_n_pairs(monkeypatch):
    _reset_cache(monkeypatch)
    rows = [[0.9, 0.05, 0.05], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]
    tokenizer, model = _install_fake_loader(monkeypatch, {0: "entailment", 1: "neutral", 2: "contradiction"}, rows)

    pairs = [("Premise A", "Hyp A"), ("Premise B", "Hyp B"), ("Premise C", "Hyp C")]
    scores = nli.score_pairs(pairs)

    assert len(model.calls) == 1
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["premises"] == ["Premise A", "Premise B", "Premise C"]
    assert tokenizer.calls[0]["hypotheses"] == ["Hyp A", "Hyp B", "Hyp C"]
    assert scores == [
        nli.NliScores(entailment=0.9, neutral=0.05, contradiction=0.05),
        nli.NliScores(entailment=0.1, neutral=0.2, contradiction=0.7),
        nli.NliScores(entailment=0.3, neutral=0.4, contradiction=0.3),
    ]


def test_score_pairs_empty_list_returns_empty_without_loading(monkeypatch):
    _reset_cache(monkeypatch)

    def _boom():
        raise AssertionError("sollte bei leerer Paar-Liste nicht laden")

    monkeypatch.setattr(nli, "_import_and_load", _boom)
    assert nli.score_pairs([]) == []


# ---- Label-Mapping via id2label ----------------------------------------------


def test_label_mapping_handles_swapped_label_order(monkeypatch):
    # Vertauschte Reihenfolge: contradiction=0, neutral=1, entailment=2 — das
    # Gegenteil der "Standard bei Laurer-Modellen"-Annahme aus eval_quality.py.
    # Genau die Falle, die das id2label-basierte Mapping abfangen muss.
    _reset_cache(monkeypatch)
    rows = [[0.7, 0.2, 0.1]]  # contradiction=0.7, neutral=0.2, entailment=0.1
    _install_fake_loader(monkeypatch, {0: "contradiction", 1: "neutral", 2: "entailment"}, rows)

    scores = nli.score_pairs([("P", "H")])

    assert scores == [nli.NliScores(entailment=0.1, neutral=0.2, contradiction=0.7)]


def test_label_mapping_is_case_insensitive(monkeypatch):
    _reset_cache(monkeypatch)
    rows = [[0.4, 0.3, 0.3]]
    _install_fake_loader(monkeypatch, {0: "Entailment", 1: "Neutral", 2: "Contradiction"}, rows)

    scores = nli.score_pairs([("P", "H")])

    assert scores == [nli.NliScores(entailment=0.4, neutral=0.3, contradiction=0.3)]


# ---- Truncation-Parameter -----------------------------------------------------


def test_truncation_parameters_passed_through(monkeypatch):
    _reset_cache(monkeypatch)
    tokenizer, _ = _install_fake_loader(
        monkeypatch, {0: "entailment", 1: "neutral", 2: "contradiction"}, [[1.0, 0.0, 0.0]]
    )

    nli.score_pairs([("P", "H")])

    call = tokenizer.calls[0]
    assert call["truncation"] is True
    assert call["max_length"] == 512


# ---- Abstain-Pfad --------------------------------------------------------------


def test_score_pairs_returns_none_when_loader_fails(monkeypatch, capsys):
    _reset_cache(monkeypatch)

    def _raise_import_error():
        raise ImportError("transformers nicht installiert")

    monkeypatch.setattr(nli, "_import_and_load", _raise_import_error)

    assert nli.score_pairs([("P", "H")]) is None
    assert "nicht ladbar" in capsys.readouterr().err


def test_score_pairs_abstain_path_logs_only_once(monkeypatch, capsys):
    _reset_cache(monkeypatch)

    def _raise_import_error():
        raise ImportError("offline")

    monkeypatch.setattr(nli, "_import_and_load", _raise_import_error)

    assert nli.score_pairs([("P", "H")]) is None
    assert nli.score_pairs([("P2", "H2")]) is None

    err = capsys.readouterr().err
    assert err.count("nicht ladbar") == 1


def test_score_pairs_does_not_crash_on_missing_dependencies(monkeypatch):
    _reset_cache(monkeypatch)

    def _raise():
        raise ModuleNotFoundError("no torch")

    monkeypatch.setattr(nli, "_import_and_load", _raise)
    assert nli.score_pairs([("P", "H")]) is None


# ---- nli_available() ------------------------------------------------------------


def test_nli_available_true_when_deps_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    assert nli.nli_available() is True


def test_nli_available_false_when_transformers_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert nli.nli_available() is False


def test_nli_available_does_not_attempt_model_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    def _boom():
        raise AssertionError("nli_available darf kein Modell laden")

    monkeypatch.setattr(nli, "_import_and_load", _boom)
    nli.nli_available()
