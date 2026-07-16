"""Punkt 8 (#294-Nebenfund, Reviews 15.07.): Chart 3/Legende 3 zeigt rohe
kleingeschriebene PDF-Keys.

Befund: `_chart_scaling()` (feeds ch3 "Scaling"/leg3 in
internal/dashboard/eval_dashboard.html) kopiert `r["label"]` unveraendert aus
den `all_log_runs`-Zeilen. `_read_all_log_runs()` setzt dort
`_PDF_LABELS.get(key, key)` -- fuer PDFs AUSSERHALB der 3 registrierten
_PDF_LABELS-Eintraege (Regelfall, s. #204 P8b/#294) faellt das auf den
rohen, kleingeschriebenen Log-Key zurueck (z. B. "cobaltite-paper" statt
"Cobaltite Paper"). Dieselbe Bugklasse traf bereits `_chart_longitudinal`
(Trade-off-Chart-2, U4-Fix, s. Kommentar dort) -- dort lokal per
`_PDF_LABELS.get(key) or re.sub(r"[-_]+", " ", key).strip().title()`
gefixt; ch3/leg3 bekam denselben Fix bisher nicht.

Fix hier: identische Formel direkt in `_chart_scaling` (gleiche Quelle wie
_chart_longitudinal, nach #311 kanonisierte Keys + _PDF_LABELS) -- bewusst
NICHT an der Quelle (_read_all_log_runs) geaendert, um den Blast-Radius auf
andere Konsumenten von all_log_runs[...]["label"] (_chart_acceptance u. a.,
nicht Teil dieses Fundes) nicht auszuweiten."""

from __future__ import annotations

from generative.eval_dashboard import _chart_scaling


def _run(key, label, ver="v1", words=5000, n_total=8, n_vault=6, pages=10):
    return {
        "key": key,
        "label": label,
        "ver": ver,
        "words": words,
        "n_total": n_total,
        "n_vault": n_vault,
        "pages": pages,
        "accept_pct": round(n_vault / n_total * 100, 1),
    }


def test_scaling_prettifies_raw_key_when_not_in_pdf_labels():
    # "cobaltite-paper" ist NICHT in _PDF_LABELS (nur bates/kuhlthau/schlebbe)
    # -- _read_all_log_runs faellt fuer solche Quellen auf den rohen Key
    # zurueck (label == key), genau das reproduziert diese Fixture.
    runs = [_run("cobaltite-paper", "cobaltite-paper")]
    chart = _chart_scaling(runs)
    assert chart["points"][0]["label"] == "Cobaltite Paper"


def test_scaling_uses_registered_pdf_labels_dict_when_available():
    runs = [_run("bates", "bates")]
    chart = _chart_scaling(runs)
    assert chart["points"][0]["label"] == "Bates 2017"


def test_scaling_prettify_collapses_multiple_separators():
    runs = [_run("multi__word--source", "multi__word--source")]
    chart = _chart_scaling(runs)
    assert chart["points"][0]["label"] == "Multi Word Source"


def test_scaling_preserves_already_proper_label_from_db_fallback_path():
    """DB-Fallback-Pfad (eval_dashboard_server.py) setzt label oft schon
    ordentlich (aus pdf_label/pdf_source abgeleitet, NICHT der rohe Key) --
    dieser Fall darf nicht kaputtgehen: label != key bleibt unangetastet."""
    runs = [_run("cobaltite-paper", "Cobaltite Paper (DB)")]
    chart = _chart_scaling(runs)
    assert chart["points"][0]["label"] == "Cobaltite Paper (DB)"
