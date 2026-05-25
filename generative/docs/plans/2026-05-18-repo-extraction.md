# Repo Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Code aus dem privaten Obsidian-Vault-Repo in ein separates oeffentliches GitHub-Repo DerTill123/atomic-agent extrahieren, sodass der Vault privat bleibt und der Code Open-Source verfuegbar ist.

**Architecture:** git filter-repo extrahiert den 98-system/scripts/atomic-agent/-Pfad mit voller Git-Historie in ein neues Repo. Persoenliche Pfade werden via Umgebungsvariablen parametrisiert. Der Vault bindet den Code danach als Git-Submodule ein.

**Tech Stack:** git, git-filter-repo (Python-Tool), Python os.environ / python-dotenv, GitHub CLI (gh)

---

## Voraussetzungen pruefen

- [ ] git filter-repo installiert: pip show git-filter-repo -- falls nicht: pip install git-filter-repo
- [ ] gh CLI installiert und eingeloggt: gh auth status
- [ ] Vault-Repo ist clean: git status im Vault zeigt keine uncommitted changes

---

## Task 1: Neues Repo per filter-repo extrahieren

**Files:**
- Erstellt: C:/Users/yourname/atomic-agent/ (neues Repo ausserhalb des Vaults)

- [ ] **Schritt 1: Vault-Repo klonen**

```bash
cd C:/Users/yourname
git clone C:/Users/yourname/Obsidian_Vault atomic-agent-extract
cd atomic-agent-extract
```

- [ ] **Schritt 2: filter-repo -- nur atomic-agent-Pfad behalten**

```bash
git filter-repo --path 98-system/scripts/atomic-agent/ --path-rename 98-system/scripts/atomic-agent/:
```

Erwartetes Ergebnis: Repo enthaelt nur noch Inhalt von 98-system/scripts/atomic-agent/ direkt im Root. Pruefen: git log --oneline | head -5

- [ ] **Schritt 3: Verzeichnis umbenennen**

```powershell
Rename-Item C:/Users/yourname/atomic-agent-extract C:/Users/yourname/atomic-agent
```

- [ ] **Schritt 4: Verifikation**

```bash
cd C:/Users/yourname/atomic-agent && ls
```

Erwartetes Ergebnis: orchestrator.py, agents/, pipeline/, tests/ direkt im Root -- kein 98-system/-Prefix.

---

## Task 2: .gitignore + .env.example anlegen

- [ ] **Schritt 1: .gitignore anlegen** in C:/Users/yourname/atomic-agent/.gitignore

```
.cache/
*.pyc
__pycache__/
.env
.vscode/
.idea/
.pytest_cache/
.coverage
```

- [ ] **Schritt 2: .env.example anlegen** in C:/Users/yourname/atomic-agent/.env.example

```
ATOMIC_AGENT_VAULT_PATH=C:/Users/yourname/YourVault
ATOMIC_AGENT_PDF_BASE=C:/Users/yourname/Documents/Literatur
ATOMIC_AGENT_MODEL=claude-sonnet-4-5
# ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Schritt 3: Commit**

```bash
git add .gitignore .env.example
git commit -m "chore: add .gitignore and .env.example"
```

---

## Task 3: Persoenliche Pfade in config.py parametrisieren

**Files:**
- Modifiziert: C:/Users/yourname/atomic-agent/config.py

- [ ] **Schritt 1: Hardcoded Pfade identifizieren**

```bash
grep -n "yourname\|OneDrive\|Obsidian_Vault\|Literatur" config.py
```

Notiere alle Zeilen mit absoluten Pfaden.

- [ ] **Schritt 2: python-dotenv installieren**

```bash
pip install python-dotenv
```

- [ ] **Schritt 3: config.py Anfang ersetzen**

Fuege am Anfang von config.py ein (vor allen anderen Importen):

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = Path(os.environ.get(
    "ATOMIC_AGENT_VAULT_PATH",
    str(Path.home() / "Obsidian_Vault")
))

PDF_BASE = Path(os.environ.get(
    "ATOMIC_AGENT_PDF_BASE",
    str(Path.home() / "Documents" / "Literatur")
))

DEFAULT_MODEL = os.environ.get("ATOMIC_AGENT_MODEL", "claude-sonnet-4-5")
```

Alle bisherigen Path("C:/Users/yourname/...") ersetzen durch VAULT_PATH / "..." bzw. PDF_BASE / "...".

- [ ] **Schritt 4: Unit-Tests pruefen**

```bash
python -m pytest tests/ -x -q --ignore=tests/test_e2e_baseline.py
```

Erwartetes Ergebnis: alle Unit-Tests gruen.

- [ ] **Schritt 5: Commit**

```bash
git add config.py
git commit -m "refactor: parametrize personal paths via env vars"
```

---

## Task 4: requirements.txt + LICENSE anlegen

- [ ] **Schritt 1: requirements.txt anlegen**

```
anthropic>=0.25.0
python-dotenv>=1.0.0
sentence-transformers>=2.7.0
rank-bm25>=0.2.2
requests>=2.31.0
```

Versionen gegen pip show <package> abgleichen.

- [ ] **Schritt 2: LICENSE anlegen (Apache 2.0)**

Volltext von https://www.apache.org/licenses/LICENSE-2.0.txt kopieren, Copyright-Zeile ersetzen durch:

    Copyright 2026 Till Quandel

- [ ] **Schritt 3: Commit**

```bash
git add requirements.txt LICENSE
git commit -m "chore: add requirements.txt and Apache 2.0 license"
```

---

## Task 5: README skeleton

- [ ] **Schritt 1: README.md anlegen** in C:/Users/yourname/atomic-agent/README.md mit Inhalt:

    # Atomic Agent
    
    > Notebook LM gives you a conversation. Atomic Agent gives you a card catalog.
    
    Turns your PDF library into a verified Obsidian card catalog --
    atomic notes with anchor verification, citation fidelity, and tier classification.
    Open-source, your API key, your data.
    
    ## Status
    
    Early development -- CLI tool, Obsidian plugin coming.
    
    ## What it does
    
    - Extracts atomic concept notes from PDFs
    - Verifies every citation anchor against the original text
    - Classifies sources (CrossRef, OpenAlex, Retraction Watch)
    - Writes directly into your Obsidian vault with wikilinks and PDF jump links
    - Model-agnostic: Claude, GPT-4o, local models via Ollama
    
    ## Quick start
    
        pip install -r requirements.txt
        cp .env.example .env
        python orchestrator.py --source "path/to/paper.pdf" --dry-run
    
    ## License
    
    Apache 2.0

- [ ] **Schritt 2: Commit**

```bash
git add README.md
git commit -m "docs: add README skeleton"
```

---

## Task 6: GitHub-Repo anlegen und pushen

- [ ] **Schritt 1: Repo anlegen**

```bash
cd C:/Users/yourname/atomic-agent
gh repo create DerTill123/atomic-agent --public --description "Turns PDFs into a verified Obsidian card catalog"
```

- [ ] **Schritt 2: Remote + Push**

```bash
git remote add origin https://github.com/DerTill123/atomic-agent.git
git push -u origin main
```

- [ ] **Schritt 3: Keine persoenlichen Pfade pruefen**

```bash
grep -r "yourname" . --include="*.py"
grep -r "yourname" . --include="*.md"
```

Erwartetes Ergebnis: keine Treffer.

- [ ] **Schritt 4: .cache/ nicht im Repo pruefen**

```bash
git ls-files .cache
```

Erwartetes Ergebnis: keine Ausgabe.

---

## Task 7: Vault -- Submodule einrichten

Dieser Task laeuft im Vault-Repo (C:/Users/yourname/Obsidian_Vault).

- [ ] **Schritt 1: Vault clean pruefen**

```bash
cd C:/Users/yourname/Obsidian_Vault
git status
```

Falls Aenderungen: zuerst committen.

- [ ] **Schritt 2: Alten Pfad aus Vault-Tracking entfernen**

```bash
git rm -r --cached 98-system/scripts/atomic-agent
git commit -m "chore: remove atomic-agent from vault tracking (moving to submodule)"
```

- [ ] **Schritt 3: Physischen Ordner loeschen und Submodule hinzufuegen**

```powershell
Remove-Item -Recurse -Force 98-system/scripts/atomic-agent
```

```bash
git submodule add https://github.com/DerTill123/atomic-agent.git 98-system/scripts/atomic-agent
git commit -m "chore: add atomic-agent as git submodule"
```

- [ ] **Schritt 4: Verifikation**

```bash
git submodule status
```

Erwartetes Ergebnis: <hash> 98-system/scripts/atomic-agent (heads/main)

- [ ] **Schritt 5: .env fuer lokale Runs**

```bash
cd 98-system/scripts/atomic-agent
cp .env.example .env
```

.env befuellen:

```
ATOMIC_AGENT_VAULT_PATH=C:/Users/yourname/Obsidian_Vault
ATOMIC_AGENT_PDF_BASE=C:/Users/yourname/OneDrive/Dokumente/Literatur
ATOMIC_AGENT_MODEL=claude-sonnet-4-5
```

- [ ] **Schritt 6: Smoke-Test**

```bash
python -m pytest tests/ -x -q --ignore=tests/test_e2e_baseline.py
```

Erwartetes Ergebnis: alle Unit-Tests gruen.

---

## Abschluss-Checkliste

- [ ] https://github.com/DerTill123/atomic-agent public erreichbar
- [ ] Kein yourname-Pfad im Public Repo: grep -r "yourname" . gibt keine Treffer
- [ ] .cache/ nicht im Repo: git ls-files .cache gibt keine Ausgabe
- [ ] Submodule korrekt: git submodule status zeigt Hash
- [ ] Unit-Tests gruen
- [ ] README + LICENSE sichtbar

## Naechster Plan

2026-05-18-subscription-refactor.md -- Umstieg von claude -p Subprocess auf nutzerseitigen API-Key (Anthropic SDK + OpenAI-kompatibles Interface).