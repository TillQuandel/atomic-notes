"""Trust-Cross-Check fuer den eingebetteten `/Title` (#234).

Befund: Die Pipeline quarantaenisiert den eingebetteten Info-Dict-`/Author`
bereits (pdf_chunker), vertraut dem `/Title` aber ungeprueft (Trust-Asymmetrie).
Beim Schlebbe-PDF hat ein frueherer enrich(rename=True)-Lauf ALLE eingebetteten
Metadaten von *Afzal 2017* zurueckgeschrieben (Selbst-Vergiftung) -> die Zitation
fuehrte den Fremd-Titel 'Conceptualisation and Measurement of Information Needs'
statt Schlebbe & Greifeneder 2022.

Fix (#234): Widerspricht der eingebettete Info-Dict-Autor dem Dateiname-Autor,
ist der GESAMTE Embedded-Block unglaubwuerdig -> `/Title` verwerfen, sodass die
vorhandene Fallback-Kaskade (enrich-Merge / q_title / vault_writer) den
dateinamen-abgeleiteten Titel nutzt. Kein Overreach: nur bei parsbarem Dateiname
UND echtem Autor-Widerspruch.
"""

from __future__ import annotations

from pathlib import Path

from generative import orchestrator
from generative.schemas.atomic_note import QualityReport
from generative.schemas.citation import build_citation_meta
from generative.pipeline.vault_writer import build_quellen_block

# Realer Schlebbe-Fall (Scout-verifiziert per pdfinfo).
SCHLEBBE_NAME = "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf"
FILENAME_TITLE = "Information Need, Informationsbedarf und -bedürfnis"
POISON_TITLE = "Conceptualisation and Measurement of Information Needs: A Literature Review"


def _qr(**kw) -> QualityReport:
    defaults = dict(peer_reviewed=None, citation_count=None, retracted=False, flags=[])
    defaults.update(kw)
    return QualityReport(**defaults)


def _poison_meta() -> dict:
    # So wie pdf_chunker.pdf_metadata das vergiftete PDF liefert: /Title bleibt im
    # `keep`-Set, /Author landet nur diagnostisch als InfoDictAuthor.
    return {"Title": POISON_TITLE, "Subject": "2017", "Pages": "9", "InfoDictAuthor": "Afzal"}


# ---- Bug-Dokumentation (ohne Fix): Fehlattribution reproduzieren ---------------


def test_without_quarantine_citation_carries_poison_title():
    """Ohne Cross-Check liest build_citation_meta pdf_meta['Title'] direkt
    (citation.py) -> der Gift-Titel landet ungefiltert in der Zitation."""
    cite = build_citation_meta(_poison_meta(), _qr(), None, SCHLEBBE_NAME)
    assert cite.title == POISON_TITLE  # die gemeldete Fehlattribution
    assert POISON_TITLE in build_quellen_block("Kernaussage (S. 3).", SCHLEBBE_NAME, cite)


# ---- Fix: Autor-Widerspruch quarantaenisiert den Titel -------------------------


def test_quarantine_discards_poison_title_on_author_mismatch():
    meta = _poison_meta()
    quarantined = orchestrator._quarantine_poisoned_embedded_title(meta, Path(SCHLEBBE_NAME))
    assert quarantined is True
    assert "Title" not in meta  # Gift-Titel verworfen
    # InfoDictAuthor bleibt diagnostisch erhalten, Seitenzahl unberuehrt.
    assert meta["InfoDictAuthor"] == "Afzal"
    assert meta["Pages"] == "9"


def test_quarantine_makes_citation_use_filename_title():
    meta = _poison_meta()
    orchestrator._quarantine_poisoned_embedded_title(meta, Path(SCHLEBBE_NAME))
    cite = build_citation_meta(meta, _qr(), None, SCHLEBBE_NAME)
    assert cite.title is None  # pdf_meta['Title'] verworfen -> keine Fehlattribution
    block = build_quellen_block("Kernaussage (S. 3).", SCHLEBBE_NAME, cite)
    assert POISON_TITLE not in block  # Gift-Titel nirgends im gerenderten Block
    assert f": {FILENAME_TITLE}" in block  # Dateiname-Titel im Titel-Slot der Zitation


# ---- Positiv-Kontrollen: sauberer Bestand darf nicht kaputtgehen ---------------


def test_matching_embedded_author_keeps_title():
    meta = {"Title": "Genuine Title", "InfoDictAuthor": "Bates", "Pages": "5"}
    quarantined = orchestrator._quarantine_poisoned_embedded_title(meta, Path("Bates - 2017 - Genuine Title.pdf"))
    assert quarantined is False
    assert meta["Title"] == "Genuine Title"


def test_no_embedded_author_keeps_title():
    # Viele PDFs tragen keinen Info-Dict-Autor -> kein widersprechendes Signal ->
    # der Titel bleibt zitierfaehig (Bestandsschutz, kein Overreach).
    meta = {"Title": "Genuine Title", "Pages": "5"}
    quarantined = orchestrator._quarantine_poisoned_embedded_title(meta, Path("Bates - 2017 - Genuine Title.pdf"))
    assert quarantined is False
    assert meta["Title"] == "Genuine Title"


def test_unparseable_filename_keeps_title():
    # Scan mit Info-Dict-Autor, aber nicht-parsbarem Dateiname: der Autor kann
    # NICHT widerlegt werden -> kein Widerspruch -> Titel bleibt (kein Overreach).
    meta = {"Title": "Genuine Title", "InfoDictAuthor": "Landry", "Pages": "5"}
    quarantined = orchestrator._quarantine_poisoned_embedded_title(meta, Path("scan001.pdf"))
    assert quarantined is False
    assert meta["Title"] == "Genuine Title"
