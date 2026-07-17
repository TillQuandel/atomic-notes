"""#332: die figure_alt-Diagnosemeldung ("N Kandidaten, 0 Anker-Matches") nannte
unbedingt einen `--load-drafts`-Kontext als Erklaerung -- auch wenn der Lauf
tatsaechlich mit `--fresh-run` lief (Coverage-Serie 2, Lauf 5/Kok, wortwoertlich
im Log belegt). `bind_figures_to_drafts` kennt den Aufrufkontext (--load-drafts
vs. --fresh-run) gar nicht -- die Meldung unterstellte also eine Ursache, die im
aktuellen Aufrufkontext unmoeglich war. Gleiche Fehlklasse wie das bereits
gefixte #288 ("keine Metadaten im Dateinamen erkannt").

Fix: neutrale Formulierung ohne unbelegte Kontext-Annahme -- reine Textaenderung,
keine Logikaenderung (Fix-Richtung des Issues).

RED vor dem Fix: Meldung nennt "--load-drafts" bedingungslos.
"""

from __future__ import annotations

from generative.pipeline.figure_alt import TaggedFigure, bind_figures_to_drafts
from generative.tests.test_figure_alt import _draft


def test_zero_match_diagnostic_does_not_claim_load_drafts_context(capsys):
    fig = TaggedFigure(anchor_page=5, alt_text="x", label=None)
    draft = _draft("Suche", ["S. 3"])

    bind_figures_to_drafts([fig], [draft])

    err = capsys.readouterr().err
    assert "figure_alt:" in err
    assert "0 Anker-Matches" in err
    # Die Meldung darf keinen --load-drafts-Kontext unterstellen -- der Aufrufer
    # kann ebenso gut --fresh-run gelaufen sein (Kok-Beleg, Coverage-Serie 2).
    assert "--load-drafts" not in err
    assert "übersprungen" in err
