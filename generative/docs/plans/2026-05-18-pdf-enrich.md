# PDF Enrichment Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein eigenständiges Script (`tools/pdf_enrich.py`) das PDFs ohne Metadaten anreichert — DOI aus Text extrahieren, CrossRef/OpenAlex abfragen, Datei umbenennen, PDF-Metadaten schreiben — damit atomic-agent korrekte Quellenangaben produziert.

**Architecture:** DOI-Regex → CrossRef (kein LLM nötig für 80%+ akademischer Paper). Fallback-Kaskade: CrossRef via DOI → OpenAlex via Titel → optionaler LLM-Fallback (Flag). OCR nur wenn `ocrmypdf`-Binary vorhanden und PDF leer. Standalone nutzbar ohne atomic-agent; als Stage-0 in orchestrator.py integriert.

**Tech Stack:** Python 3.10+, `pypdf` (1 pip install, pure Python), `urllib` (stdlib), `argparse` (stdlib), `re` (stdlib). Kein weiteres Pflicht-Dependency.

---

## File Map

| File | Aktion | Verantwortlichkeit |
|---|---|---|
| `tools/pdf_enrich.py` | Create | Standalone-Script: Text-Extraktion, DOI-Regex, CrossRef/OpenAlex, Rename, Metadaten-Write |
| `tests/test_pdf_enrich.py` | Create | Unit-Tests für alle Kernfunktionen |
| `orchestrator.py` | Modify | Stage-0: `pdf_enrich` vor Pipeline aufrufen wenn Metadaten fehlen |

---

### Task 1: PDF-Text-Extraktion + Leer-Erkennung

**Files:**
- Create: `tools/__init__.py` (leer)
- Create: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Test schreiben**

```python
# tests/test_pdf_enrich.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_extract_text_returns_string(tmp_path):
    """extract_text() gibt String zurück (auch bei leerer PDF)."""
    from tools.pdf_enrich import extract_text
    # Minimale valide PDF erstellen
    pdf_bytes = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(pdf_bytes)
    text = extract_text(pdf_path, max_pages=3)
    assert isinstance(text, str)


def test_is_scanned_detects_empty():
    """is_scanned() gibt True zurück bei leerem Text."""
    from tools.pdf_enrich import is_scanned
    assert is_scanned("") is True
    assert is_scanned("   \n  ") is True
    assert is_scanned("Some text here") is False
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
cd c:\Users\tillq\Obsidian_Vault\98-system\scripts\atomic-agent
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: `tools/__init__.py` und Anfang von `tools/pdf_enrich.py` erstellen**

```python
# tools/pdf_enrich.py
"""PDF Metadata Enrichment Tool.

Reichert PDFs ohne Metadaten an: DOI-Extraktion → CrossRef/OpenAlex → Rename + Metadaten-Write.

Usage:
  python tools/pdf_enrich.py paper.pdf
  python tools/pdf_enrich.py paper.pdf --dry-run
  python tools/pdf_enrich.py paper.pdf --llm-fallback

Requires: pip install pypdf
Optional: ocrmypdf (für gescannte PDFs)
"""
from __future__ import annotations
import re
import shutil
import sys
from pathlib import Path


def extract_text(pdf_path: Path, max_pages: int = 3) -> str:
    """Extrahiert Text aus den ersten max_pages Seiten via pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Fehler: pypdf nicht installiert. Bitte: pip install pypdf", file=sys.stderr)
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        pages = reader.pages[:max_pages]
        return "\n".join(p.extract_text() or "" for p in pages)
    except Exception as e:
        print(f"[pdf_enrich] Textextraktion fehlgeschlagen: {e}", file=sys.stderr)
        return ""


def is_scanned(text: str) -> bool:
    """Gibt True zurück wenn der extrahierte Text leer ist (gescannte PDF)."""
    return not text.strip()
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 2 passed

- [ ] **Step 5: Commit**

```
git add tools/__init__.py tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): text extraction + scan detection (Task 1)"
```

---

### Task 2: DOI-Extraktion per Regex

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Test schreiben** (ergänzen)

```python
def test_extract_doi_finds_doi_in_text():
    from tools.pdf_enrich import extract_doi
    text = "See also doi:10.1016/j.ipm.2019.05.003 for details."
    assert extract_doi(text) == "10.1016/j.ipm.2019.05.003"


def test_extract_doi_finds_https_doi():
    from tools.pdf_enrich import extract_doi
    text = "Available at https://doi.org/10.3389/fpsyg.2019.02730"
    assert extract_doi(text) == "10.3389/fpsyg.2019.02730"


def test_extract_doi_returns_none_when_missing():
    from tools.pdf_enrich import extract_doi
    assert extract_doi("No DOI here, just text.") is None
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
python -m pytest tests/test_pdf_enrich.py::test_extract_doi_finds_doi_in_text -v
```
Erwartet: `ImportError: cannot import name 'extract_doi'`

- [ ] **Step 3: `extract_doi()` implementieren**

```python
# In tools/pdf_enrich.py ergänzen:

_DOI_RE = re.compile(
    r"(?:doi[:./\s]|https?://doi\.org/|DOI[:.\s])"
    r"(10\.\d{4,9}/[^\s\"'<>]+)",
    re.IGNORECASE,
)

def extract_doi(text: str) -> str | None:
    """Extrahiert ersten DOI per Regex. Gibt None zurück wenn keiner gefunden."""
    m = _DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(1).rstrip(".,;)")
    return doi
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 5 passed

- [ ] **Step 5: Commit**

```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): DOI regex extraction (Task 2)"
```

---

### Task 3: CrossRef-Lookup via DOI

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Test schreiben**

```python
def test_crossref_lookup_returns_dict_on_valid_doi(monkeypatch):
    from tools.pdf_enrich import crossref_lookup
    import urllib.request

    fake_response = b'{"status":"ok","message":{"title":["Information Behavior"],"author":[{"family":"Bates","given":"Marcia J."}],"published":{"date-parts":[[2017]]},"DOI":"10.1002/asi.23681","type":"journal-article"}}'

    class FakeResp:
        def read(self): return fake_response
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    result = crossref_lookup("10.1002/asi.23681")
    assert result is not None
    assert result["title"] == "Information Behavior"
    assert result["author"] == "Bates"
    assert result["year"] == 2017


def test_crossref_lookup_returns_none_on_error(monkeypatch):
    from tools.pdf_enrich import crossref_lookup
    import urllib.request, urllib.error
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(urllib.error.HTTPError(None, 404, "Not Found", {}, None)))
    assert crossref_lookup("10.9999/fake") is None
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
python -m pytest tests/test_pdf_enrich.py::test_crossref_lookup_returns_dict_on_valid_doi -v
```

- [ ] **Step 3: `crossref_lookup()` implementieren**

```python
# In tools/pdf_enrich.py ergänzen:
import json
import urllib.error
import urllib.parse
import urllib.request

_CROSSREF_BASE = "https://api.crossref.org/works"
_USER_AGENT = "pdf-enrich/1.0 (mailto:atomic-agent-user)"


def crossref_lookup(doi: str) -> dict | None:
    """Fragt CrossRef mit DOI ab. Gibt normalisiertes Metadaten-Dict oder None zurück."""
    url = f"{_CROSSREF_BASE}/{urllib.parse.quote(doi, safe='/')}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())["message"]
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError):
        return None
    return _normalize_crossref(data)


def _normalize_crossref(data: dict) -> dict:
    """Extrahiert title, author, year, doi, type aus CrossRef-Response."""
    titles = data.get("title") or []
    title = titles[0] if titles else ""
    authors = data.get("author") or []
    first_author = authors[0].get("family", "") if authors else ""
    published = data.get("published", {}).get("date-parts", [[None]])
    year = published[0][0] if published and published[0] else None
    return {
        "title": title,
        "author": first_author,
        "year": int(year) if year else None,
        "doi": data.get("DOI", ""),
        "type": data.get("type", ""),
    }
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 7 passed

- [ ] **Step 5: Commit**

```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): CrossRef DOI lookup (Task 3)"
```

---

### Task 4: OpenAlex Titel-Fallback

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Test schreiben**

```python
def test_openalex_title_search_returns_result(monkeypatch):
    from tools.pdf_enrich import openalex_title_search
    import urllib.request

    fake = b'{"results":[{"title":"Information Behavior","authorships":[{"author":{"display_name":"Marcia J. Bates"}}],"publication_year":2017,"doi":"https://doi.org/10.1002/asi.23681","type":"journal-article"}]}'

    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    result = openalex_title_search("Information Behavior")
    assert result is not None
    assert result["title"] == "Information Behavior"
    assert result["year"] == 2017


def test_openalex_returns_none_on_empty_results(monkeypatch):
    from tools.pdf_enrich import openalex_title_search
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{"results":[]}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert openalex_title_search("zzz_nonexistent_zzz") is None
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
python -m pytest tests/test_pdf_enrich.py::test_openalex_title_search_returns_result -v
```

- [ ] **Step 3: `openalex_title_search()` implementieren**

```python
# In tools/pdf_enrich.py ergänzen:

_OPENALEX_BASE = "https://api.openalex.org/works"


def openalex_title_search(title: str) -> dict | None:
    """Sucht in OpenAlex nach Titel. Gibt normalisiertes Dict oder None zurück."""
    params = urllib.parse.urlencode({"search": title, "per-page": "1"})
    url = f"{_OPENALEX_BASE}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            results = json.loads(resp.read()).get("results", [])
    except (urllib.error.URLError, json.JSONDecodeError):
        return None
    if not results:
        return None
    item = results[0]
    authors = item.get("authorships", [])
    first_author = ""
    if authors:
        first_author = authors[0].get("author", {}).get("display_name", "").split()[-1]
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    return {
        "title": item.get("title", ""),
        "author": first_author,
        "year": item.get("publication_year"),
        "doi": doi,
        "type": item.get("type", ""),
    }
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 9 passed

- [ ] **Step 5: Commit**

```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): OpenAlex title fallback (Task 4)"
```

---

### Task 5: Datei umbenennen + PDF-Metadaten schreiben

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben**

```python
def test_build_filename_standard():
    from tools.pdf_enrich import build_filename
    meta = {"author": "Bates", "year": 2017, "title": "Information Behavior"}
    assert build_filename(meta) == "Bates - 2017 - Information Behavior.pdf"


def test_build_filename_truncates_long_title():
    from tools.pdf_enrich import build_filename
    long = "A" * 150
    result = build_filename({"author": "X", "year": 2020, "title": long})
    assert len(result) <= 120
    assert result.endswith(".pdf")


def test_build_filename_sanitizes_special_chars():
    from tools.pdf_enrich import build_filename
    meta = {"author": "Müller", "year": 2021, "title": "Test/Paper: Results"}
    result = build_filename(meta)
    assert "/" not in result
    assert ":" not in result


def test_rename_pdf_renames_file(tmp_path):
    from tools.pdf_enrich import rename_pdf
    src = tmp_path / "paper123.pdf"
    src.write_bytes(b"%PDF-1.4")
    meta = {"author": "Bates", "year": 2017, "title": "Information Behavior"}
    new_path = rename_pdf(src, meta, dry_run=False)
    assert new_path.name == "Bates - 2017 - Information Behavior.pdf"
    assert new_path.exists()
    assert not src.exists()


def test_rename_pdf_dry_run_does_not_rename(tmp_path):
    from tools.pdf_enrich import rename_pdf
    src = tmp_path / "paper123.pdf"
    src.write_bytes(b"%PDF-1.4")
    meta = {"author": "Bates", "year": 2017, "title": "Information Behavior"}
    new_path = rename_pdf(src, meta, dry_run=True)
    assert src.exists()  # Original bleibt
    assert new_path.name == "Bates - 2017 - Information Behavior.pdf"
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
python -m pytest tests/test_pdf_enrich.py::test_build_filename_standard -v
```

- [ ] **Step 3: `build_filename()` + `rename_pdf()` implementieren**

```python
# In tools/pdf_enrich.py ergänzen:

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def build_filename(meta: dict, max_title_len: int = 80, max_author_len: int = 40) -> str:
    """Baut Zotero-kompatiblen Dateinamen: 'Author - Year - Title.pdf'."""
    author = (_UNSAFE_CHARS.sub("", meta.get("author", "Unknown")).strip() or "Unknown")[:max_author_len]
    year = str(meta.get("year") or "n.d.")
    title = _UNSAFE_CHARS.sub("", meta.get("title", "Untitled")).strip() or "Untitled"
    if len(title) > max_title_len:
        title = title[:max_title_len].rsplit(" ", 1)[0]
    return f"{author} - {year} - {title}.pdf"


def rename_pdf(pdf_path: Path, meta: dict, dry_run: bool = False) -> Path:
    """Benennt PDF um. Bei dry_run=True: nur neuen Pfad zurückgeben ohne Umbenennen."""
    new_name = build_filename(meta)
    new_path = pdf_path.parent / new_name
    if not dry_run:
        pdf_path.rename(new_path)
    return new_path
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 13 passed

- [ ] **Step 5: Commit**

```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): filename builder + rename (Task 5)"
```

---

### Task 6: OCR-Erkennung + optionaler ocrmypdf-Fallback

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben**

```python
def test_ocr_available_returns_bool():
    from tools.pdf_enrich import ocr_available
    result = ocr_available()
    assert isinstance(result, bool)


def test_run_ocr_returns_path_when_available(tmp_path, monkeypatch):
    from tools.pdf_enrich import run_ocr
    import shutil
    # ocrmypdf simulieren
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ocrmypdf" if cmd == "ocrmypdf" else None)

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
        # Erstelle Output-Datei
        out = tmp_path / "ocr_out.pdf"
        out.write_bytes(b"%PDF-1.4 OCR")
        return R()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.4")
    result = run_ocr(src)
    assert result is not None
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
python -m pytest tests/test_pdf_enrich.py::test_ocr_available_returns_bool -v
```

- [ ] **Step 3: `ocr_available()` + `run_ocr()` implementieren**

```python
# In tools/pdf_enrich.py ergänzen:
import subprocess
import tempfile


def ocr_available() -> bool:
    """Gibt True zurück wenn ocrmypdf-Binary verfügbar ist."""
    return shutil.which("ocrmypdf") is not None


def run_ocr(pdf_path: Path) -> Path | None:
    """Führt ocrmypdf auf PDF aus. Gibt Pfad zur OCR-PDF zurück oder None bei Fehler."""
    if not ocr_available():
        return None
    out_path = pdf_path.parent / f"_ocr_{pdf_path.name}"
    try:
        result = subprocess.run(
            ["ocrmypdf", "--quiet", "--skip-text", str(pdf_path), str(out_path)],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and out_path.exists():
            return out_path
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 15 passed

- [ ] **Step 5: Commit**

```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): optional OCR via ocrmypdf (Task 6)"
```

---

### Task 7: Haupt-Pipeline + CLI

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Test schreiben**

```python
def test_enrich_returns_meta_for_pdf_with_doi(tmp_path, monkeypatch):
    """enrich() gibt Metadaten zurück wenn DOI im Text gefunden und CrossRef antwortet."""
    from tools.pdf_enrich import enrich
    import urllib.request

    # pypdf mock
    monkeypatch.setattr("tools.pdf_enrich.extract_text",
                        lambda path, max_pages=3: "See doi:10.1002/asi.23681 for details")

    fake_cr = b'{"status":"ok","message":{"title":["Information Behavior"],"author":[{"family":"Bates"}],"published":{"date-parts":[[2017]]},"DOI":"10.1002/asi.23681","type":"journal-article"}}'
    class FakeResp:
        def read(self): return fake_cr
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    meta = enrich(pdf, dry_run=True)
    assert meta is not None
    assert meta["author"] == "Bates"
    assert meta["year"] == 2017
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
python -m pytest tests/test_pdf_enrich.py::test_enrich_returns_meta_for_pdf_with_doi -v
```

- [ ] **Step 3: `enrich()` + `main()` implementieren**

```python
# In tools/pdf_enrich.py ergänzen:
import argparse


def enrich(pdf_path: Path, dry_run: bool = False, llm_fallback: bool = False) -> dict | None:
    """Haupt-Pipeline: Text → DOI → CrossRef → OpenAlex → (LLM) → Rename.

    Gibt Metadaten-Dict zurück oder None wenn nichts gefunden.
    Bei dry_run=True: kein Umbenennen, nur Metadaten zurückgeben.
    """
    print(f"[pdf-enrich] {pdf_path.name}")

    text = extract_text(pdf_path)

    if is_scanned(text):
        print("  → Gescannte PDF erkannt")
        if ocr_available():
            print("  → OCR via ocrmypdf...")
            ocr_path = run_ocr(pdf_path)
            if ocr_path:
                text = extract_text(ocr_path)
                if not dry_run:
                    pdf_path = ocr_path
        if is_scanned(text):
            print("  → Kein Text extrahierbar — übersprungen", file=sys.stderr)
            return None

    # Stage 1: DOI per Regex
    doi = extract_doi(text)
    meta = None
    if doi:
        print(f"  → DOI gefunden: {doi}")
        meta = crossref_lookup(doi)
        if meta:
            print(f"  → CrossRef: {meta['author']} ({meta['year']}) — {meta['title'][:60]}")

    # Stage 2: OpenAlex Titel-Fallback
    if not meta:
        # Erste nicht-leere Zeile als Titel-Kandidat
        title_guess = next((l.strip() for l in text.splitlines() if len(l.strip()) > 20), "")
        if title_guess:
            print(f"  → Kein DOI, suche Titel: '{title_guess[:60]}'")
            meta = openalex_title_search(title_guess)
            if meta:
                print(f"  → OpenAlex: {meta['author']} ({meta['year']}) — {meta['title'][:60]}")

    # Stage 3: LLM-Fallback (optional)
    if not meta and llm_fallback:
        print("  → LLM-Fallback (Haiku)...")
        meta = _llm_extract(text[:3000])

    if not meta:
        print("  → Keine Metadaten gefunden — übersprungen", file=sys.stderr)
        return None

    new_path = rename_pdf(pdf_path, meta, dry_run=dry_run)
    if dry_run:
        print(f"  → [dry-run] würde umbenennen zu: {new_path.name}")
    else:
        print(f"  → Umbenannt: {new_path.name}")
        _write_pdf_metadata(new_path, meta)

    return meta


def _llm_extract(text: str) -> dict | None:
    """Haiku-Fallback: extrahiert Titel/Autor/Jahr aus Rohtext."""
    try:
        import subprocess as _sp
        prompt = (
            "Extrahiere aus diesem Anfang eines akademischen Papers: Titel, Erstautor (Nachname), Jahr.\n"
            "Antworte NUR in diesem Format (eine Zeile je Feld):\n"
            "title: <Titel>\nauthor: <Nachname>\nyear: <Jahr>\n\n"
            f"Text:\n{text}"
        )
        r = _sp.run(["claude", "-p", "--model", "haiku"],
                    input=prompt, capture_output=True, text=True, timeout=30)
        lines = {l.split(":")[0].strip(): l.split(":", 1)[1].strip()
                 for l in r.stdout.splitlines() if ":" in l}
        return {
            "title": lines.get("title", ""),
            "author": lines.get("author", ""),
            "year": int(lines["year"]) if lines.get("year", "").isdigit() else None,
            "doi": "",
            "type": "",
        } if lines.get("title") else None
    except Exception:
        return None


def _write_pdf_metadata(pdf_path: Path, meta: dict) -> None:
    """Schreibt Metadaten (Titel, Autor, Jahr) in PDF-Datei via Temp-File (atomar)."""
    try:
        import tempfile
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.add_metadata({
            "/Title": meta.get("title", ""),
            "/Author": meta.get("author", ""),
            "/Subject": str(meta.get("year", "")),
            "/Keywords": meta.get("doi", ""),
        })
        # Temp-File: Original bleibt intakt bis Write erfolgreich
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=pdf_path.parent) as tmp:
            writer.write(tmp)
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(pdf_path))
    except Exception as e:
        print(f"  [warn] Metadaten-Schreiben fehlgeschlagen: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF Metadata Enrichment — DOI → CrossRef → Rename"
    )
    parser.add_argument("pdf", nargs="+", help="PDF-Datei(en) oder Glob-Pattern")
    parser.add_argument("--dry-run", action="store_true", help="Nicht umbenennen, nur ausgeben")
    parser.add_argument("--llm-fallback", action="store_true", help="Haiku nutzen wenn CrossRef nichts findet")
    args = parser.parse_args()

    import glob
    paths = [Path(p) for pattern in args.pdf for p in glob.glob(pattern)]
    if not paths:
        print("Keine PDFs gefunden.", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        enrich(path, dry_run=args.dry_run, llm_fallback=args.llm_fallback)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: 16 passed

- [ ] **Step 5: Smoke-Test**

```
python tools/pdf_enrich.py "C:\Users\tillq\OneDrive\Dokumente\Literatur\Bates - 2017 - Information Behavior.pdf" --dry-run
```
Erwartet:
```
[pdf-enrich] Bates - 2017 - Information Behavior.pdf
  → DOI gefunden: 10.1002/asi.23681
  → CrossRef: Bates (2017) — Information Behavior
  → [dry-run] würde umbenennen zu: Bates - 2017 - Information Behavior.pdf
```

- [ ] **Step 6: Commit**

```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): main pipeline + CLI (Task 7)"
```

---

### Task 8: atomic-agent Integration als Stage-0

**Files:**
- Modify: `orchestrator.py`

- [ ] **Step 1: Integration in `orchestrator.py` einbauen**

In `main()`, direkt nach dem Einlesen von `args.source` und VOR dem ersten Pipeline-Schritt:

```python
# Stage 0: PDF-Enrichment wenn Metadaten fehlen
from pipeline.pdf_chunker import pdf_metadata as _pdf_meta
_meta_check = _pdf_meta(source_path)
_has_author = bool(_meta_check.get("Author") or _meta_check.get("author"))
_has_year   = bool(_meta_check.get("Year")   or _meta_check.get("year"))
if not (_has_author and _has_year):
    print(f"[0/7] PDF-Enrichment — keine Metadaten im Dateinamen erkannt…")
    try:
        from tools.pdf_enrich import enrich as _enrich
        _enrich(source_path, dry_run=False, llm_fallback=args.llm_fallback if hasattr(args, "llm_fallback") else False)
    except Exception as _e:
        print(f"  [warn] PDF-Enrichment fehlgeschlagen: {_e}", file=sys.stderr)
```

- [ ] **Step 2: `--llm-fallback`-Flag zu orchestrator.py hinzufügen**

In `argparse`-Setup von `main()`:
```python
parser.add_argument("--llm-fallback", action="store_true",
                    help="LLM (Haiku) für PDF-Enrichment nutzen wenn CrossRef nichts findet")
```

- [ ] **Step 3: Smoke-Test mit namenlosen PDF**

```
# PDF umbenennen zum Testen:
copy "C:\Users\tillq\OneDrive\Dokumente\Literatur\Bates - 2017 - Information Behavior.pdf" "C:\tmp\paper_ohne_name.pdf"

python orchestrator.py --source "C:\tmp\paper_ohne_name.pdf" --dry-run
```
Erwartet: `[0/7] PDF-Enrichment — keine Metadaten im Dateinamen erkannt…` gefolgt von normalem Pipeline-Output.

- [ ] **Step 4: Commit**

```
git add orchestrator.py
git commit -m "feat(orchestrator): Stage-0 PDF-Enrichment bei fehlenden Metadaten"
```

---

## Self-Review

**Spec-Coverage:**
- [x] Standalone-Script (`tools/pdf_enrich.py`) mit CLI
- [x] DOI-Regex → CrossRef (kein LLM nötig)
- [x] OpenAlex-Fallback via Titel
- [x] OCR optional via ocrmypdf
- [x] LLM-Fallback via `--llm-fallback` Flag
- [x] Datei umbenennen (Zotero-Format)
- [x] PDF-Metadaten schreiben
- [x] atomic-agent Integration als Stage-0
- [x] Nur `pypdf` als Pflicht-Dependency

**Bewusst offen:**
- Zotero Local API Integration (localhost:23119) — nicht im Scope v1, da Zotero laufen muss
- Batch-Modus für ganze Ordner — funktioniert bereits via Glob: `python tools/pdf_enrich.py *.pdf`

**Placeholder-Check:** Keine TBDs. Alle Code-Blöcke vollständig.

**Type-Konsistenz:** `enrich()` gibt `dict | None` zurück, `rename_pdf()` gibt `Path` zurück, `crossref_lookup()` gibt `dict | None` — konsistent durch alle Tasks.
