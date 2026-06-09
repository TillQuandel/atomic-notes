# M1 — „Ein Fremder kann es installieren"

Kontext: Baseline-Bewertung 2026-06-10 (`2026-06-10-projekt-bewertung.md`) — konvergentes
Urteil aller Rater: Robustheit/Produktreife 3/5, Gap zu OSS = Installation, Backend-Wahl,
Doku. M1 schließt genau diesen Gap. Bezug: Issues #9, #10, #11 (release-prep).

**M1-Akzeptanzkriterium (Außenkriterium):** Eine fremde Person (oder frische VM) schafft
ohne Hilfe: Install → `doctor` grün → 1 PDF durch die Pipeline → Notes im Zielordner.

## Subscription-Portabilität (geklärt 2026-06-10)

Das Subscription-Backend ist **bereits portabel konzipiert** — jeder mit Claude Pro/Max
kann die Pipeline ohne API-Key nutzen:

- `ATOMIC_AGENT_BACKEND=subscription` ist Default; `_subscription_backend.py` ruft die
  Claude-Code-CLI (`claude -p --model <shorthand>`) per Subprocess, Auth = einmaliger
  OAuth-Login der CLI. Headless `-p` ist ein offiziell dokumentierter CLI-Modus.
- Voraussetzungen beim Nutzer: Claude Pro/Max-Abo, installierte Claude-Code-CLI,
  `claude`-Login. Kein Key, keine Zusatzkosten über das Abo hinaus.
- Bekannte Grenzen (dokumentieren, nicht verstecken): 5-h-Rate-Fenster ≈ ~8 volle
  Pipeline-Läufe, dann HTTP 429 bis Reset; Modell-Shorthand-Mapping
  (`_CLI_ALIASES`) muss bei neuen Modellen gepflegt werden.
- Für M1 ist hier **nichts Neues zu bauen** — nur: Fehlerpfade härten (CLI fehlt /
  nicht eingeloggt → klare Meldung statt Subprocess-Traceback) und dokumentieren.
  litellm-Backend bleibt die API-Alternative (Anthropic/OpenAI/Ollama via Keys).

## Arbeitspakete

### S1 — Packaging + CI (eine fokussierte Session, KEINE Parallel-Agenten)

Begründung: ändert Imports global (Monorepo: `shared/` auf Root wird von `generative/`
via sys.path-Hack importiert; `extractive/` parallel; Tests inserten Pfade selbst).
Parallel-Agenten würden auf derselben Fläche kollidieren.

- `pyproject.toml` (PEP 621), installierbares Layout für `generative` + `extractive` +
  `shared`; einfachste tragfähige Variante wählen, kein Over-Engineering.
- Console-Entry-Points: `atomic-notes run …` (orchestrator), Platzhalter `doctor`.
- Alle `sys.path.insert`-Hacks aus Quell- und Testfiles entfernen → Paket-Imports.
- `requirements.txt` → `pyproject` dependencies; Extras erwägen (z. B. `[eval]` für
  schwere Eval-Deps), nur wenn es die Installation real vereinfacht.
- GitHub Actions: ubuntu-latest, Python 3.12, `apt-get install poppler-utils`,
  `pip install -e .`, `pytest` (Suite ist LLM-frei, ~30 s). Badge ins README.
- CI deckt Linux-Portabilität zwangsläufig auf (Pfade, `claude` vs. `claude.cmd`) —
  portabel fixen, Tests nicht blind skippen.

**Akzeptanz S1:** frische venv → `pip install -e .` → volle Suite grün ohne
sys.path-Manipulation; Actions-Lauf grün auf ubuntu.

### S2 — Preflight (`doctor`) + Backend-Fehlerpfade

- `atomic-notes doctor`: prüft pdftotext/pdfinfo im PATH, Backend erreichbar
  (subscription: CLI vorhanden + eingeloggt; litellm: Key gesetzt), Vault-/Output-Pfad
  beschreibbar, optionale Deps — mit konkretem Installationshinweis pro Fehlschlag.
- Subscription-Fehlerpfad: fehlende/nicht eingeloggte CLI → eine verständliche Meldung.
- Rate-Limit-Verhalten dokumentieren (429/Reset).

**Akzeptanz S2:** Auf einem System ohne pdftotext bzw. ohne `claude`-Login liefert
`doctor` die korrekte Diagnose; Pipeline-Start bricht früh und verständlich ab.

### S3 — Quickstart + Beispiel + Projekt-Basics

- README-Quickstart: Install → doctor → Beispiel-Lauf → wo liegen die Notes.
- Lizenz-sicheres Beispiel-PDF (NICHT die Eval-Lehrbuch-PDFs — Urheberrecht; eigenes
  oder gemeinfreies Dokument) + 1–2 erzeugte Notes als Showcase.
- Root-`LICENSE` (aus `generative/LICENSE` heben, Konsistenz prüfen), kleines
  `CONTRIBUTING.md`.

**Akzeptanz S3:** README-Quickstart auf frischer Maschine nachvollziehbar; Beispiel-Lauf
reproduziert die Showcase-Notes.

Reihenfolge: S1 → S2 → S3 (S2/S3 sind nach S1 unabhängig; nacheinander in 1–2 Sessions
reicht — Parallel-Worktrees lohnen den Overhead hier nicht).

---

## Session-Prompt S1 (copy-paste für nächste Session)

```text
Repo: C:/Users/tillq/source/repos/atomic-notes. Aufgabe: M1-S1 aus
internal/docs/m1-installierbarkeit-plan.md — Packaging + CI. Lies zuerst diesen Plan
und internal/docs/2026-06-10-projekt-bewertung.md (Kontext).

Vorbedingung: Prüfe ob Branch fix/24-agreement-4-moc-suggest schon in master ist;
falls nein, kläre die Merge-Entscheidung mit mir BEVOR du anfängst (Packaging ändert
dieselben Files).

Scope (hart): pyproject.toml + installierbares Paket-Layout für generative/extractive/
shared, Console-Entry-Point `atomic-notes run`, ALLE sys.path.insert-Hacks raus,
GitHub-Actions-Workflow (ubuntu, poppler-utils, pip install -e ., pytest), Badge.
KEIN Verhalten ändern, KEINE Features, KEIN Refactoring jenseits der Imports.
Subscription-Backend bleibt Default. Windows bleibt unterstützt.

Arbeitsweise: TDD wo neue Logik entsteht; sonst Suite als Netz — vor UND nach jedem
Umbau-Schritt volle Suite (`python -m pytest generative -q`, Stand: 413 passed).
Mechanisch in kleinen Commits arbeiten (Layout → Imports modulweise → Entry-Points →
CI). Cross-Model-Review (codex exec) vor dem finalen Commit.

Akzeptanz: frische venv → pip install -e . → volle Suite grün ohne jede
sys.path-Manipulation in Quell-/Testfiles; Actions-Lauf grün; `atomic-notes run
--source <pdf> --dry-run` startet (bis Backend-Check) auf einem System ohne Vault.

Stolperfallen (bekannt): shared/db_schema.py liegt auf Repo-Root und wird von
generative/db.py via sys.path-Hack geladen; generative/tests/ inserten ROOT selbst;
calibration/-Module importieren sich gegenseitig über Pfad-Inserts (seit 2026-06-10
teils auf calibration.kappa-Paketimporte umgestellt); pytest.ini existiert auf Root
UND in generative/.
```

## Session-Prompt S2 (nach S1)

```text
Repo: atomic-notes. Aufgabe: M1-S2 aus internal/docs/m1-installierbarkeit-plan.md —
`atomic-notes doctor` (Preflight) + Backend-Fehlerpfade härten. Scope hart auf den
Plan begrenzt; TDD (Checks sind pure Funktionen + dünne CLI); Subscription-Pfad:
fehlende/nicht eingeloggte claude-CLI muss eine verständliche Meldung mit nächstem
Schritt liefern, kein Traceback. Akzeptanz siehe Plan-S2. Suite grün halten.
```

## Session-Prompt S3 (nach S2)

```text
Repo: atomic-notes. Aufgabe: M1-S3 aus internal/docs/m1-installierbarkeit-plan.md —
README-Quickstart, lizenz-sicheres Beispiel-PDF + Showcase-Notes, Root-LICENSE,
CONTRIBUTING.md. WICHTIG: kein urheberrechtlich geschütztes Eval-PDF bundlen —
eigenes/gemeinfreies Dokument erzeugen. Akzeptanz: Quickstart auf frischer Maschine
nachvollziehbar (beschreibe den Verifikationsweg explizit).
```
