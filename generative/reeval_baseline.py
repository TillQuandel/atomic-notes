"""reeval_baseline.py — Re-Evaluiert alle Baseline-Notes mit eval_quality_v4.

Läuft eval_quality_v4 über alle vault__*.md in .cache/eval/baseline/PDF-Name/
und schreibt Ergebnisse in quality_history.jsonl + atomic_analytics.db.

Resume-fähig: überspringt Notes die bereits mit der aktuellen EVAL_VERSION
(aus eval_quality_v4, s. `_eq.EVAL_VERSION`) in DB stehen. Additiv über
Versionsbumps hinweg — Notes mit nur älteren eval_version-Zeilen werden erneut
evaluiert, die alten Zeilen bleiben unverändert erhalten (#232-Re-Eval-Sweep).

--repeat: Wiederholungs-Sweep fuer Eval-Rausch-Messungen (Trendfaehigkeits-
Studie 2026-07-16: bisherige "Wiederholungen" trafen den content-adressierten
Judge-Cache und lieferten byte-identische Ergebnisse statt einer echten
zweiten Messung). Wirkt additiv:
  (a) Der Resume-Skip (`_already_done`) wird ignoriert — eine Note mit
      bereits vorhandener aktueller-EVAL_VERSION-Zeile wird TROTZDEM erneut
      evaluiert. Nichts wird ueberschrieben.
  (b) `eval_note(..., no_cache=True)` erzwingt eine echte Judge-Neuberechnung.
      Das ist die einzige Wiederverwendungs-Ebene, die dieses Skript
      durchlaeuft: der Re-Eval-Hash-Guard (`_eq.find_cached_eval`) wird
      ausschliesslich von orchestrator.py Stage 8 konsultiert, nie von hier
      (reeval_baseline.py uebergibt eval_note() nie einen content_hash).
  (c) Neue Zeilen werden mit `repeat_sweep: true` markiert (quality_history.
      jsonl) und der Run in pipeline_runs mit pipeline_version='reeval-repeat'
      (statt 'reeval') registriert, damit die Rauschanalyse Wiederholungs-
      gruppen sicher findet.
Ohne --repeat ist das Verhalten unveraendert.

Verwendung:
  python reeval_baseline.py [--dry-run] [--baseline-dir PFAD] [--repeat]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generative import eval_quality_v4 as _eq
from generative import db as _db
from generative.config import LITERATURE_DIR

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / ".cache" / "eval" / "baseline"

# Mapping: EXAKTER Ordnername (unter .cache/eval/baseline/) -> PDF-Pfad.
#
# Exakter Abgleich, kein Praefix-Match mehr (Alt-Code: `folder.startswith(prefix)`).
# Praefix-Matching kollidierte real: "Porst-2014-Auszug-S1-40-bak-preclean-20260530"
# beginnt mit "Porst-2014-Auszug-S1-40" und haette denselben Eintrag geerbt, obwohl
# der Backup-Ordner explizit ausgeschlossen sein soll. Seit #241 liegt der
# run_id-Namespace ohnehin als Unterordner (<stem>/<run_id>/), nicht als Suffix am
# Ordnernamen — Praefix-Matching hatte dafuer also auch keinen Zweck mehr.
#
# Herkunft je Eintrag: historische Note->PDF-Zuordnung aus atomic_analytics.db
# (note_evals.pdf je vault__*.md-Datei des Ordners, eval_version=4.1) + Existenz-
# Check gegen LITERATURE_DIR (#232-Re-Eval-Sweep-Recherche 2026-07-15, volle
# Beleg-Tabelle je Eintrag im PR-Body). `None` = bewusst ausgeschlossen, Grund
# im Kommentar — kein stilles Weglassen (Prinzip: dokumentieren, nicht raten).
PDF_MAP: dict[str, Path | None] = {
    # leerer Alt-Ordner: 0 Notes, 0 DB-Historie — vermutlich Vorgaenger von
    # "Bates - Information Behavior" vor einer Ordner-Umbenennung.
    "Bates - 2017 - Information Behavior": None,
    # DB-Mehrheit 4/5 Note-Zeilen zeigen auf dieses PDF (die 5. abweichende
    # Zeile "bates-2017.pdf" ist vermutlich eine Rename-Altlast); Ordnername
    # matcht das PDF exakt.
    "Bates - Information Behavior": LITERATURE_DIR / "Bates - Information Behavior.pdf",
    # historisches PDF "bates-2017.pdf" existiert nicht mehr in LITERATURE_DIR
    # — eigenstaendige DB-Zuordnung, NICHT ungeprueft durch das Bates-PDF von
    # oben ersetzt (keine Belegkette dafuer).
    "bates-2017": None,
    "Ebner und Gegenfurtner - 2019 - Learning and Satisfaction in Webinar, Online, and Face-to-Face Instruction A Meta-Analysis": (
        LITERATURE_DIR
        / "Ebner und Gegenfurtner - 2019 - Learning and Satisfaction in Webinar, Online, and Face-to-Face Instruction A Meta-Analysis.pdf"
    ),
    # Test-/Struktur-Ordner (Pytest-Fixture-Notes mit source-file "flags.pdf"),
    # keine echte Baseline.
    "flags": None,
    # Test-/Struktur-Ordner (Pytest-Fixture), keine echte Baseline.
    "fresh": None,
    "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning": (
        LITERATURE_DIR / "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf"
    ),
    # historisches PDF ist ein Auszug (S1-40) und existiert nicht mehr — NICHT
    # durch das Voll-PDF ersetzt (Score-Verwaesserung durch mehr Chunks; offener
    # Punkt, siehe PR-Body).
    "Porst-2014-Auszug-S1-40": None,
    # Backup-Ordner (Suffix "-bak-preclean", 0 Notes vorhanden) — Backups werden
    # nicht re-evaluiert.
    "Porst-2014-Auszug-S1-40-bak-preclean-20260530": None,
    "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis": (
        LITERATURE_DIR / "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf"
    ),
    "Spreitzer - Die Evaluation von Elementen des Wissenstransfers im Zuge des Onboarding-Prozesses am Beispiel eines": (
        LITERATURE_DIR
        / "Spreitzer - Die Evaluation von Elementen des Wissenstransfers im Zuge des Onboarding-Prozesses am Beispiel eines.pdf"
    ),
    # Test-/Struktur-Ordner (Pytest-Fixture, source-file "src.pdf"), keine echte
    # Baseline.
    "src": None,
    "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen …": (
        LITERATURE_DIR / "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen ….pdf"
    ),
    # historisches PDF "zettelkasten-primer.pdf" nicht in LITERATURE_DIR
    # vorhanden.
    "zettelkasten-primer": None,
}


def _find_pdf(folder_name: str) -> Path | None:
    """Liefert den PDF-Pfad fuer einen Baseline-Ordner (exakter Namensabgleich,
    s. PDF_MAP-Kommentar)."""
    return PDF_MAP.get(folder_name)


def _latest_notes_dir(pdf_dir: Path) -> Path:
    """Liefert das run_id-Verzeichnis mit den aktuell gültigen Baseline-Notes für
    einen PDF-Stamm-Ordner.

    #241: seit dem run_id-Namespace liegen Notes unter `<stem>/<run_id>/` statt
    direkt unter `<stem>/`. Bei mehreren run_ids (mehrere Läufe desselben PDFs,
    z.B. eine A/B-Messreihe) gilt die NEUESTE (lexikographisch größte, da das
    run_id-Format `YYYYMMDD-HHMMSS` chronologisch sortiert) als die aktuell
    gültige Baseline — reeval_baseline re-evaluiert den zuletzt geschriebenen
    Stand, nicht veraltete Zwischenläufe. Bei NULL run_id-Unterordnern gilt der
    Stamm-Ordner selbst als Notes-Verzeichnis (reine Legacy-Ablage vor #241).

    Achtung: im GEMISCHTEN Fall (Legacy-flat-Notes direkt unter `<stem>/` UND
    run_id-Unterordner) liefert diese Funktion NUR den run_id-Ordner; die
    zusätzliche Einbeziehung der Legacy-flat-Notes übernimmt `_baseline_note_files`
    (#261) — diese Funktion allein nicht direkt als Note-Quelle verwenden.
    """
    run_dirs = sorted((d for d in pdf_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
    return run_dirs[-1] if run_dirs else pdf_dir


def _baseline_note_files(pdf_dir: Path) -> list[Path]:
    """Liefert alle aktuell gültigen `vault__*.md`-Baseline-Notes eines PDF-Stamm-
    Ordners.

    #261: `_latest_notes_dir` verwirft im gemischten Ordner (Legacy-flat +
    run_id-Unterordner) die pre-#241-Legacy-flat-Notes still, sobald ein
    run_id-Ordner existiert. Da der Produktions-Baseline-Cache aktuell komplett
    aus Legacy-flat-Notes besteht, würde damit der gesamte historische Bestand
    eines PDFs beim ersten Neu-Lauf aus reeval/Kalibrierung fallen. Deshalb
    werden Legacy-flat-Notes zusätzlich einbezogen und per Dateiname dedupliziert
    — die gleichnamige Note des neuesten run_id-Ordners hat Vorrang (neueste
    Pipeline-Ausgabe ist maßgeblich).
    """
    latest = _latest_notes_dir(pdf_dir)
    by_name: dict[str, Path] = {}
    if latest != pdf_dir:  # gemischt: Legacy-flat direkt unter <stem>/ zuerst (niedrigster Vorrang)
        for f in pdf_dir.glob("vault__*.md"):
            by_name[f.name] = f
    for f in latest.glob("vault__*.md"):  # neuester run_id (bzw. reine Legacy-Ablage) gewinnt
        by_name[f.name] = f
    return sorted(by_name.values())


def _already_done(note_name: str, conn) -> bool:
    """Prüft ob Note schon mit der aktuellen EVAL_VERSION (`_eq.EVAL_VERSION`)
    in DB steht. Additiv über Versionsbumps: Notes mit nur älteren
    eval_version-Zeilen gelten NICHT als erledigt und werden erneut evaluiert."""
    row = conn.execute(
        "SELECT 1 FROM note_evals WHERE note_path=? AND eval_version=? LIMIT 1",
        (note_name, _eq.EVAL_VERSION),
    ).fetchone()
    return row is not None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Ueberschreibt BASELINE_DIR (Default: ROOT-relativ, .cache/eval/baseline neben diesem Skript). "
        "Fuer Funktionsnachweise aus einem isolierten Worktree ohne eigenes .cache.",
    )
    ap.add_argument(
        "--repeat",
        action="store_true",
        help="Wiederholungs-Sweep fuer Eval-Rausch-Messungen: ignoriert den Resume-Skip (additiv, "
        "nichts wird ueberschrieben) UND erzwingt echte Judge-Neuberechnung (no_cache=True statt "
        "Cache-Treffer). Neue Zeilen bekommen repeat_sweep=true; der Run wird in pipeline_runs als "
        "'reeval-repeat' statt 'reeval' gefuehrt.",
    )
    args = ap.parse_args(argv)

    baseline_dir = args.baseline_dir if args.baseline_dir is not None else BASELINE_DIR

    if not baseline_dir.exists():
        print(f"Baseline-Dir nicht gefunden: {baseline_dir}")
        sys.exit(1)

    notes = []
    for pdf_dir in sorted(baseline_dir.iterdir()):
        if not pdf_dir.is_dir():
            continue
        pdf_path = _find_pdf(pdf_dir.name)
        if pdf_path is None:
            print(f"  [skip] kein PDF für: {pdf_dir.name[:50]}")
            continue
        if not pdf_path.exists():
            print(f"  [skip] PDF nicht gefunden: {pdf_path.name}")
            continue
        for note_file in _baseline_note_files(pdf_dir):
            notes.append((note_file, pdf_path, pdf_dir.name))

    print(f"Zu evaluieren: {len(notes)} Notes\n")

    done = skip = errors = 0

    # Reeval-Run in pipeline_runs eintragen damit FK-Constraints erfüllt sind
    from generative.agents.base import _RUN_ID as _reeval_run_id

    # --repeat: eigenes Run-Label statt 'reeval', damit die Rauschanalyse
    # Wiederholungsgruppen ueber pipeline_runs.pipeline_version sicher findet
    # (join note_evals.run_id -> pipeline_runs.run_id). Additiv, keine neue Spalte.
    reeval_pipeline_version = "reeval-repeat" if args.repeat else "reeval"

    if not args.dry_run:
        with _db.get_db() as _conn_init:
            _conn_init.execute(
                """
                INSERT OR IGNORE INTO pipeline_runs
                (run_id, timestamp, pipeline_version, pdf_source, pdf_key, pdf_label,
                 n_generated, n_vault, n_inbox, fully_cached)
                VALUES (?, datetime('now'), ?, 'baseline-reeval', 'reeval', 'Re-Eval Baseline',
                        ?, 0, 0, 0)
            """,
                (_reeval_run_id, reeval_pipeline_version, len(notes)),
            )
        run_label = " (Repeat-Sweep)" if args.repeat else ""
        print(f"  Reeval-Run ID: {_reeval_run_id}{run_label}\n")

    with _db.get_db() as conn:
        for i, (note_path, pdf_path, folder) in enumerate(notes, 1):
            if not args.repeat and _already_done(note_path.name, conn):
                print(f"  [{i:2}/{len(notes)}] skip (bereits v{_eq.EVAL_VERSION}): {note_path.name[:55]}")
                skip += 1
                continue

            print(f"  [{i:2}/{len(notes)}] {note_path.name[:55]}")

            if args.dry_run:
                done += 1
                continue

            try:
                result = _eq.eval_note(note_path, pdf_path, no_cache=args.repeat)
                if "error" not in result:
                    if args.repeat:
                        result["repeat_sweep"] = True
                    _eq.save_result(result)
                    done += 1
                    hall = result.get("hallucination_rate", -1)
                    cov = result.get("coverage_factual", -1)
                    print(f"       → hall={hall:.1%}  cov={cov:.1%}")
                else:
                    print(f"       → FEHLER: {result['error']}")
                    errors += 1
            except Exception as e:
                print(f"       → EXCEPTION: {e}")
                errors += 1

    mode = ("[dry-run] " if args.dry_run else "") + ("[repeat] " if args.repeat else "")
    print(f"\n{mode}Fertig: {done} neu evaluiert, {skip} übersprungen, {errors} Fehler")

    if not args.dry_run:
        evals = _db.query_note_evals(eval_version=_eq.EVAL_VERSION)
        print(f"note_evals mit eval_version={_eq.EVAL_VERSION}: {len(evals)}")


if __name__ == "__main__":
    main()
