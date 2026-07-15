"""Tests fuer #232-Re-Eval-Sweep-Vorbereitung: reeval_baseline.py evaluierte
auf master 0 Notes (`--dry-run` -> "Zu evaluieren: 0 Notes"), obwohl alle 14
Ordner in .cache/eval/baseline/ echte Baseline-Notes enthalten.

Befund (zwei unabhaengige Bugs):

1. `_already_done()` prueft hartkodiert `eval_version='4.1'` (Zeile 92 auf
   master) -- nach dem EVAL_VERSION-Bump auf 4.3 (#298) werden dadurch ALLE
   historisch mit 4.1 evaluierten Notes als "schon erledigt" markiert, obwohl
   genau deren Re-Evaluierung unter der aktuellen Version der Zweck des
   Sweeps ist. RED auf master: TestAlreadyDoneUsesCurrentEvalVersion.

2. `PDF_MAP` (Praefix->PDF) ist veraltet und nutzt Praefix-Matching
   (`folder.startswith(prefix)`), das reale Ordner-Kollisionen erzeugt --
   z.B. wuerde ein Ordner "X-bak-preclean" denselben Praefix-Treffer wie "X"
   erben, obwohl der Backup-Ordner explizit ausgeschlossen sein soll. Fix:
   exakter Ordnername als Schluessel (PDF_MAP.get), aus der historischen
   Note->PDF-Zuordnung in atomic_analytics.db rekonstruiert (siehe PR-Body
   fuer die volle Beleg-Tabelle je Eintrag). RED auf master:
   TestFindPdfExactMatchNoPrefixCollision (synthetisch) und
   TestPdfMapRealBaselineFolders (echte Ordnernamen -- existieren auf master
   nicht im PDF_MAP, `_find_pdf` liefert also fuer alle `None`, nicht die
   erwarteten Pfade).

Zusaetzlich: PDF_MAP hardcodete absolute OneDrive-Pfade statt der
existierenden Config-Konstante `generative.config.LITERATURE_DIR` zu nutzen
(CLAUDE.md-Pfadregel: keine hartcodierten `C:/Users/...`-Literale im Code).
RED auf master: TestNoHardcodedLiteraturePath.

Und: fuer den Re-Eval-Sweep-Funktionsnachweis aus einem isolierten Worktree
(ohne eigenes .cache) muss BASELINE_DIR ueberschreibbar sein -- neue
`--baseline-dir`-Option (Default bleibt ROOT-relativ). RED auf master:
TestBaselineDirOption (Option existiert nicht -> argparse bricht ab).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from generative import db as _db
from generative import reeval_baseline


# ---------------------------------------------------------------------------
# 1. Guard: _already_done() muss die AKTUELLE EVAL_VERSION scopen, nicht 4.1
# ---------------------------------------------------------------------------


class TestAlreadyDoneUsesCurrentEvalVersion:
    def _db_with_eval(self, tmp_path: Path, note_path: str, eval_version: str) -> Path:
        db_path = tmp_path / "test.db"
        _db.init_db(db_path)
        with _db.get_db(db_path) as conn:
            _db.insert_run(conn, {"run_id": "r1", "pipeline_version": "v1"})
            _db.insert_eval(
                conn,
                {
                    "eval_id": f"r1__{note_path}",
                    "run_id": "r1",
                    "note_path": note_path,
                    "eval_version": eval_version,
                },
            )
        return db_path

    def test_note_with_current_eval_version_counts_as_done(self, tmp_path):
        """Eine Note, die bereits mit der AKTUELLEN EVAL_VERSION evaluiert wurde,
        muss uebersprungen werden (Resume-Faehigkeit bleibt erhalten)."""
        db_path = self._db_with_eval(tmp_path, "vault__Foo.md", reeval_baseline._eq.EVAL_VERSION)
        with _db.get_db(db_path) as conn:
            assert reeval_baseline._already_done("vault__Foo.md", conn) is True

    def test_note_with_only_old_4_1_version_does_not_count_as_done(self, tmp_path):
        """Kernbug: eine Note mit NUR einer historischen 4.1-Zeile darf NICHT als
        erledigt gelten, sobald EVAL_VERSION auf 4.3 gebumpt ist -- sonst wird sie
        (wie auf master beobachtet) faelschlich uebersprungen und der ganze Sweep
        evaluiert 0 Notes."""
        assert reeval_baseline._eq.EVAL_VERSION != "4.1", "Testvoraussetzung verletzt: EVAL_VERSION ist wieder 4.1?"
        db_path = self._db_with_eval(tmp_path, "vault__Foo.md", "4.1")
        with _db.get_db(db_path) as conn:
            assert reeval_baseline._already_done("vault__Foo.md", conn) is False

    def test_note_without_any_eval_row_does_not_count_as_done(self, tmp_path):
        db_path = tmp_path / "test.db"
        _db.init_db(db_path)
        with _db.get_db(db_path) as conn:
            assert reeval_baseline._already_done("vault__Unbekannt.md", conn) is False


# ---------------------------------------------------------------------------
# 2a. Map-/Skip-Logik: exakter Abgleich statt Praefix-Matching (synthetisch)
# ---------------------------------------------------------------------------


class TestFindPdfExactMatchNoPrefixCollision:
    """Reale Kollision im Produktions-Cache: "Porst-2014-Auszug-S1-40-bak-
    preclean-20260530" beginnt mit "Porst-2014-Auszug-S1-40" und wuerde unter
    Praefix-Matching denselben PDF-Eintrag erben wie der Haupt-Ordner --
    obwohl der Backup-Ordner einen eigenen (bewusst ausschliessenden) Eintrag
    hat. Synthetisch nachgestellt mit Foo/Foo-bak, damit der Test unabhaengig
    von den echten PDF_MAP-Werten bleibt."""

    def test_exact_match_returns_mapped_path(self, monkeypatch):
        fake_pdf = Path("literatur/Foo.pdf")
        monkeypatch.setattr(reeval_baseline, "PDF_MAP", {"Foo": fake_pdf})
        assert reeval_baseline._find_pdf("Foo") == fake_pdf

    def test_unknown_folder_returns_none(self, monkeypatch):
        monkeypatch.setattr(reeval_baseline, "PDF_MAP", {"Foo": Path("literatur/Foo.pdf")})
        assert reeval_baseline._find_pdf("Voellig-Anderer-Ordner") is None

    def test_own_none_entry_not_shadowed_by_shorter_prefix_key(self, monkeypatch):
        """RED gegen Praefix-Matching: "Foo" ist im Dict VOR "Foo-bak" eingetragen
        (Ordner-Erstellungsreihenfolge). Praefix-Code wuerde bei "Foo-bak" den
        ERSTEN passenden Praefix ("Foo") zurueckgeben und den eigenen
        None-Eintrag von "Foo-bak" nie erreichen."""
        fake_pdf = Path("literatur/Foo.pdf")
        monkeypatch.setattr(reeval_baseline, "PDF_MAP", {"Foo": fake_pdf, "Foo-bak": None})
        assert reeval_baseline._find_pdf("Foo") == fake_pdf
        assert reeval_baseline._find_pdf("Foo-bak") is None

    def test_longer_folder_name_does_not_inherit_shorter_entrys_pdf(self, monkeypatch):
        """RED gegen Praefix-Matching: "Foo-Variante" hat GAR KEINEN eigenen
        Eintrag, beginnt aber mit "Foo" -- Praefix-Code wuerde faelschlich
        fake_pdf zurueckgeben statt None."""
        fake_pdf = Path("literatur/Foo.pdf")
        monkeypatch.setattr(reeval_baseline, "PDF_MAP", {"Foo": fake_pdf})
        assert reeval_baseline._find_pdf("Foo-Variante") is None


# ---------------------------------------------------------------------------
# 2b. Map-/Skip-Logik: echte Baseline-Ordner (Regression-Lock der Recherche)
# ---------------------------------------------------------------------------


class TestPdfMapRealBaselineFolders:
    """Rekonstruiert aus atomic_analytics.db (note_evals.pdf je vault__*.md-
    Datei des jeweiligen Ordners, eval_version=4.1) + Existenz-Check gegen
    LITERATURE_DIR (#232-Re-Eval-Sweep-Recherche 2026-07-15). Volle
    Beleg-Tabelle je Eintrag im PR-Body."""

    def test_mapped_folders_resolve_under_literature_dir(self):
        from generative.config import LITERATURE_DIR

        expected_pdf_names = {
            "Bates - Information Behavior": "Bates - Information Behavior.pdf",
            (
                "Ebner und Gegenfurtner - 2019 - Learning and Satisfaction in Webinar, "
                "Online, and Face-to-Face Instruction A Meta-Analysis"
            ): (
                "Ebner und Gegenfurtner - 2019 - Learning and Satisfaction in Webinar, "
                "Online, and Face-to-Face Instruction A Meta-Analysis.pdf"
            ),
            "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning": (
                "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf"
            ),
            "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis": (
                "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf"
            ),
            (
                "Spreitzer - Die Evaluation von Elementen des Wissenstransfers im Zuge "
                "des Onboarding-Prozesses am Beispiel eines"
            ): (
                "Spreitzer - Die Evaluation von Elementen des Wissenstransfers im Zuge "
                "des Onboarding-Prozesses am Beispiel eines.pdf"
            ),
            ("Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen …"): (
                "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen ….pdf"
            ),
        }
        for folder, pdf_name in expected_pdf_names.items():
            assert reeval_baseline._find_pdf(folder) == LITERATURE_DIR / pdf_name, folder

    def test_excluded_folders_resolve_to_none(self):
        """Jeder Ausschluss ist im PDF_MAP-Kommentar begruendet (Backup, leerer
        Alt-Ordner, Test-/Struktur-Ordner, fehlendes historisches PDF)."""
        excluded_folders = [
            "Bates - 2017 - Information Behavior",
            "bates-2017",
            "Porst-2014-Auszug-S1-40",
            "Porst-2014-Auszug-S1-40-bak-preclean-20260530",
            "zettelkasten-primer",
            "flags",
            "fresh",
            "src",
        ]
        for folder in excluded_folders:
            assert reeval_baseline._find_pdf(folder) is None, folder


# ---------------------------------------------------------------------------
# 3. Regression-Guard: keine hartkodierten Literatur-Pfade (CLAUDE.md-Regel)
# ---------------------------------------------------------------------------


def test_no_hardcoded_literature_path_in_source():
    source = Path(reeval_baseline.__file__).read_text(encoding="utf-8")
    forbidden = ("One" + "Drive", "Dokumente" + "/" + "Literatur", "Dokumente" + "\\" + "Literatur")
    assert not any(item in source for item in forbidden)


# ---------------------------------------------------------------------------
# 4. --baseline-dir-Option: BASELINE_DIR fuer isolierte Worktree-Laeufe
#    ueberschreibbar machen (Default bleibt ROOT-relativ)
# ---------------------------------------------------------------------------


class TestBaselineDirOption:
    def test_baseline_dir_option_overrides_default_for_dry_run(self, tmp_path, monkeypatch, capsys):
        other_root = tmp_path / "alt-baseline"
        (other_root / "Foo").mkdir(parents=True)
        (other_root / "Foo" / "vault__X.md").write_text("x", encoding="utf-8")
        fake_pdf = tmp_path / "Foo.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(reeval_baseline, "PDF_MAP", {"Foo": fake_pdf})

        test_db_path = tmp_path / "test.db"
        _db.init_db(test_db_path)
        original_get_db = _db.get_db  # vor dem Patch sichern, sonst rekursiver Selbstaufruf
        monkeypatch.setattr(reeval_baseline._db, "get_db", lambda path=None: original_get_db(test_db_path))

        reeval_baseline.main(["--dry-run", "--baseline-dir", str(other_root)])

        out = capsys.readouterr().out
        assert "Zu evaluieren: 1 Notes" in out

    def test_default_baseline_dir_used_when_option_omitted(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(reeval_baseline, "BASELINE_DIR", tmp_path / "does-not-exist")

        with pytest.raises(SystemExit):
            reeval_baseline.main(["--dry-run"])
