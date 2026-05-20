# PDF Enrichment — ISBN + Embedded Metadata + arXiv + PubMed Extension Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tools/pdf_enrich.py` um (1) Lesen eingebetteter PDF-Metadaten, (2) ISBN-Lookup via Open Library/Google Books, (3) arXiv-ID-Lookup und (4) PubMed-PMID-Lookup erweitern, damit Buecher, Preprints und medizinische Paper neben akademischen Journals angereichert werden koennen.

**Architecture:** Neue Kaskade in `enrich()`: Eingebettete Metadaten (Stage 0) → DOI (Stage 1) → ISBN (Stage 2) → arXiv-ID (Stage 3) → PMID (Stage 4) → Titel-Suche OpenAlex (Stage 5) → LLM (Stage 6). Normalisiertes Dict-Format identisch durch alle Stages (title, author, year, doi, type). Alle APIs kostenlos, kein Auth.

**Tech Stack:** Python 3.10+, `re` + `urllib` + `pypdf` (bereits importiert/vorhanden). Keine neue Pflicht-Dependency.

---

## File Map

| File | Aktion | Verantwortlichkeit |
|---|---|---|
| `tools/pdf_enrich.py` | Modify | 6 neue Funktionen + enrich()-Kaskade |
| `tests/test_pdf_enrich.py` | Modify | Tests fuer alle neuen Funktionen |

---

### Task 1: Eingebettete PDF-Metadaten lesen (Stage 0)

Wenn ein PDF bereits Author + Title in seinem Info-Dictionary hat, kein API-Call noetig.

**Files:**
- Modify: `tools/pdf_enrich.py` (nach `extract_text` + `is_scanned`, vor `_DOI_RE`)
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben** (ans Ende von `tests/test_pdf_enrich.py` APPENDEN)

```python
def test_read_pdf_metadata_returns_dict(tmp_path, monkeypatch):
    from tools.pdf_enrich import read_pdf_metadata
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {"/Title": "Information Behavior", "/Author": "Bates", "/Subject": "2017"}
    monkeypatch.setattr("tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    result = read_pdf_metadata(tmp_path / "paper.pdf")
    assert result is not None
    assert result["title"] == "Information Behavior"
    assert result["author"] == "Bates"
    assert result["year"] == 2017


def test_read_pdf_metadata_returns_none_when_no_author(tmp_path, monkeypatch):
    from tools.pdf_enrich import read_pdf_metadata
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {"/Title": "Some Title"}
    monkeypatch.setattr("tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    assert read_pdf_metadata(tmp_path / "paper.pdf") is None


def test_read_pdf_metadata_returns_none_when_empty(tmp_path, monkeypatch):
    from tools.pdf_enrich import read_pdf_metadata
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {}
    monkeypatch.setattr("tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    assert read_pdf_metadata(tmp_path / "paper.pdf") is None
```

- [ ] **Step 2: Tests ausfuehren, muessen FAIL**
```
cd c:\Users\tillq\Obsidian_Vault\98-system\scripts\atomic-agent
python -m pytest tests/test_pdf_enrich.py::test_read_pdf_metadata_returns_dict -v
```
Erwartet: `ImportError: cannot import name 'read_pdf_metadata'`

- [ ] **Step 3: `read_pdf_metadata()` implementieren** (nach `is_scanned`, vor `_DOI_RE`)

```python
def read_pdf_metadata(pdf_path: Path) -> dict | None:
    """Liest eingebettete PDF-Metadaten (Info-Dictionary). Gibt normalisiertes Dict zurueck
    wenn Author + Title vorhanden, sonst None."""
    try:
        from pypdf import PdfReader
        info = PdfReader(str(pdf_path)).metadata or {}
    except Exception:
        return None
    title = (info.get("/Title") or "").strip()
    author = (info.get("/Author") or "").strip()
    if not (title and author):
        return None
    subject = (info.get("/Subject") or "")
    year_match = re.search(r"\d{4}", subject)
    year = int(year_match.group()) if year_match else None
    doi = (info.get("/Keywords") or "")
    doi = doi if doi.startswith("10.") else ""
    return {
        "title": title,
        "author": author.split()[-1] if author else "",
        "year": year,
        "doi": doi,
        "type": info.get("/Creator", ""),
    }
```

- [ ] **Step 4: Tests ausfuehren, muessen PASS**
```
python -m pytest tests/test_pdf_enrich.py -v
```
Erwartet: alle bisherigen + 3 neue Tests gruen (gesamt 20 passed)

- [ ] **Step 5: Commit**
```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): read embedded PDF metadata (task-1)"
```

---

### Task 2: ISBN-Regex-Extraktion

**Files:**
- Modify: `tools/pdf_enrich.py` (nach `extract_doi`)
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben** (APPENDEN)

```python
def test_extract_isbn_finds_isbn13():
    from tools.pdf_enrich import extract_isbn
    text = "ISBN: 978-3-16-148410-0"
    assert extract_isbn(text) == "9783161484100"

def test_extract_isbn_finds_isbn10():
    from tools.pdf_enrich import extract_isbn
    text = "ISBN 0-596-51774-2 is the identifier."
    assert extract_isbn(text) == "0596517742"

def test_extract_isbn_returns_none_when_missing():
    from tools.pdf_enrich import extract_isbn
    assert extract_isbn("No ISBN here.") is None

def test_extract_isbn_strips_hyphens_and_spaces():
    from tools.pdf_enrich import extract_isbn
    text = "ISBN 978 0 596 51774 8"
    assert extract_isbn(text) == "9780596517748"
```

- [ ] **Step 2: Tests ausfuehren, muessen FAIL**
```
python -m pytest tests/test_pdf_enrich.py::test_extract_isbn_finds_isbn13 -v
```

- [ ] **Step 3: `_ISBN_RE` + `extract_isbn()` implementieren** (nach `extract_doi`)

```python
_ISBN_RE = re.compile(
    r"ISBN(?:-1[03])?[:\s]*"
    r"((?:97[89][- ]?)?(?:\d[- ]?){9}[\dX])",
    re.IGNORECASE,
)

def extract_isbn(text: str) -> str | None:
    """Extrahiert erste ISBN per Regex. Gibt normalisierte Digits-only-String oder None zurueck."""
    m = _ISBN_RE.search(text)
    if not m:
        return None
    return re.sub(r"[^0-9X]", "", m.group(1))
```

- [ ] **Step 4: Tests ausfuehren (24 passed)**
```
python -m pytest tests/test_pdf_enrich.py -v
```

- [ ] **Step 5: Commit**
```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): ISBN regex extraction (task-2)"
```

---

### Task 3: Open Library + Google Books Lookup

**Files:**
- Modify: `tools/pdf_enrich.py` (nach `_normalize_crossref`)
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben** (APPENDEN)

```python
def test_open_library_lookup_returns_dict(monkeypatch):
    from tools.pdf_enrich import open_library_lookup
    import urllib.request
    fake = b'{"ISBN:9780596517748":{"title":"JavaScript: The Good Parts","authors":[{"name":"Douglas Crockford"}],"publish_date":"May 2008"}}'
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    result = open_library_lookup("9780596517748")
    assert result is not None
    assert result["title"] == "JavaScript: The Good Parts"
    assert result["author"] == "Crockford"
    assert result["year"] == 2008
    assert result["type"] == "book"

def test_open_library_lookup_returns_none_on_empty(monkeypatch):
    from tools.pdf_enrich import open_library_lookup
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert open_library_lookup("9999999999999") is None

def test_google_books_lookup_returns_dict(monkeypatch):
    from tools.pdf_enrich import google_books_lookup
    import urllib.request
    fake = b'{"items":[{"volumeInfo":{"title":"JavaScript: The Good Parts","authors":["Douglas Crockford"],"publishedDate":"2008-05-01"}}]}'
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    result = google_books_lookup("9780596517748")
    assert result is not None
    assert result["title"] == "JavaScript: The Good Parts"
    assert result["author"] == "Crockford"
    assert result["year"] == 2008
    assert result["type"] == "book"

def test_google_books_lookup_returns_none_on_empty(monkeypatch):
    from tools.pdf_enrich import google_books_lookup
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{"items":[]}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert google_books_lookup("9999999999999") is None
```

- [ ] **Step 2: Tests ausfuehren, muessen FAIL**
```
python -m pytest tests/test_pdf_enrich.py::test_open_library_lookup_returns_dict -v
```

- [ ] **Step 3: Implementieren** (nach `_normalize_crossref` einfuegen)

```python
_OPEN_LIBRARY_BASE = "https://openlibrary.org/api/books"

def open_library_lookup(isbn: str) -> dict | None:
    """Fragt Open Library mit ISBN ab. Gibt normalisiertes Dict oder None zurueck."""
    params = urllib.parse.urlencode({"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"})
    url = f"{_OPEN_LIBRARY_BASE}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError):
        return None
    key = f"ISBN:{isbn}"
    if key not in data:
        return None
    return _normalize_open_library(data[key])

def _normalize_open_library(item: dict) -> dict:
    """Extrahiert title, author, year, doi, type aus Open-Library-Response."""
    title = item.get("title", "")
    authors = item.get("authors", [])
    first_author = ""
    if authors:
        name_parts = (authors[0].get("name") or "").split()
        first_author = name_parts[-1] if name_parts else ""
    year_match = re.search(r"\d{4}", item.get("publish_date", ""))
    year = int(year_match.group()) if year_match else None
    return {"title": title, "author": first_author, "year": year, "doi": "", "type": "book"}

_GOOGLE_BOOKS_BASE = "https://www.googleapis.com/books/v1/volumes"

def google_books_lookup(isbn: str) -> dict | None:
    """Fragt Google Books mit ISBN ab. Gibt normalisiertes Dict oder None zurueck."""
    params = urllib.parse.urlencode({"q": f"isbn:{isbn}"})
    url = f"{_GOOGLE_BOOKS_BASE}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError):
        return None
    items = data.get("items") or []
    if not items:
        return None
    info = items[0].get("volumeInfo", {})
    authors = info.get("authors") or []
    first_author = ""
    if authors:
        name_parts = authors[0].split()
        first_author = name_parts[-1] if name_parts else ""
    published = info.get("publishedDate", "")
    year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
    return {"title": info.get("title", ""), "author": first_author, "year": year, "doi": "", "type": "book"}
```

- [ ] **Step 4: Tests ausfuehren (28 passed)**
```
python -m pytest tests/test_pdf_enrich.py -v
```

- [ ] **Step 5: Commit**
```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): Open Library + Google Books ISBN lookup (task-3)"
```

---

### Task 4: arXiv-ID-Extraktion + arXiv API

Preprints haben arXiv-IDs im Text: `arXiv:2301.12345` oder `arXiv:cs.AI/0612072`.

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben** (APPENDEN)

```python
def test_extract_arxiv_id_finds_new_format():
    from tools.pdf_enrich import extract_arxiv_id
    text = "See arXiv:2301.12345 for the preprint."
    assert extract_arxiv_id(text) == "2301.12345"

def test_extract_arxiv_id_finds_old_format():
    from tools.pdf_enrich import extract_arxiv_id
    text = "Available at arXiv:cs.AI/0612072"
    assert extract_arxiv_id(text) == "cs.AI/0612072"

def test_extract_arxiv_id_returns_none_when_missing():
    from tools.pdf_enrich import extract_arxiv_id
    assert extract_arxiv_id("No arXiv ID here.") is None

def test_arxiv_lookup_returns_dict(monkeypatch):
    from tools.pdf_enrich import arxiv_lookup
    import urllib.request
    fake = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <author><name>Vaswani, Ashish</name></author>
    <published>2017-06-12T00:00:00Z</published>
    <id>http://arxiv.org/abs/1706.03762v5</id>
  </entry>
</feed>'''
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    result = arxiv_lookup("1706.03762")
    assert result is not None
    assert result["title"] == "Attention Is All You Need"
    assert result["author"] == "Vaswani"
    assert result["year"] == 2017
    assert result["type"] == "preprint"

def test_arxiv_lookup_returns_none_on_empty(monkeypatch):
    from tools.pdf_enrich import arxiv_lookup
    import urllib.request
    fake = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    assert arxiv_lookup("9999.99999") is None
```

- [ ] **Step 2: Tests ausfuehren, muessen FAIL**
```
python -m pytest tests/test_pdf_enrich.py::test_extract_arxiv_id_finds_new_format -v
```

- [ ] **Step 3: `_ARXIV_RE` + `extract_arxiv_id()` + `arxiv_lookup()` implementieren** (nach `extract_isbn`)

```python
_ARXIV_RE = re.compile(
    r"arXiv[:\s]+(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)

def extract_arxiv_id(text: str) -> str | None:
    """Extrahiert erste arXiv-ID per Regex. Gibt ID-String oder None zurueck."""
    m = _ARXIV_RE.search(text)
    return m.group(1) if m else None

_ARXIV_API = "https://export.arxiv.org/api/query"

def arxiv_lookup(arxiv_id: str) -> dict | None:
    """Fragt arXiv Atom-API mit ID ab. Gibt normalisiertes Dict oder None zurueck."""
    params = urllib.parse.urlencode({"id_list": arxiv_id})
    url = f"{_ARXIV_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml = resp.read().decode("utf-8")
    except (urllib.error.URLError, UnicodeDecodeError):
        return None
    entry_m = re.search(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    if not entry_m:
        return None
    entry = entry_m.group(1)
    title_m = re.search(r"<title>(.+?)</title>", entry, re.DOTALL)
    author_m = re.search(r"<author>\s*<name>([^<]+)</name>", entry)
    date_m = re.search(r"<published>(\d{4})", entry)
    if not title_m:
        return None
    title = title_m.group(1).strip()
    author = author_m.group(1).split(",")[0].strip() if author_m else ""
    year = int(date_m.group(1)) if date_m else None
    doi_m = re.search(r'<arxiv:doi[^>]*>([^<]+)</arxiv:doi>', entry)
    doi = doi_m.group(1).strip() if doi_m else ""
    return {"title": title, "author": author, "year": year, "doi": doi, "type": "preprint"}
```

- [ ] **Step 4: Tests ausfuehren (33 passed)**
```
python -m pytest tests/test_pdf_enrich.py -v
```

- [ ] **Step 5: Commit**
```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): arXiv ID extraction + API lookup (task-4)"
```

---

### Task 5: PMID-Extraktion + PubMed API

Medizinische/Life-Sciences-Paper haben PMIDs: `PMID: 12345678`.

**Files:**
- Modify: `tools/pdf_enrich.py`
- Test: `tests/test_pdf_enrich.py`

- [ ] **Step 1: Failing Tests schreiben** (APPENDEN)

```python
def test_extract_pmid_finds_pmid():
    from tools.pdf_enrich import extract_pmid
    text = "PMID: 30049270 — see also the supplementary."
    assert extract_pmid(text) == "30049270"

def test_extract_pmid_returns_none_when_missing():
    from tools.pdf_enrich import extract_pmid
    assert extract_pmid("No PMID here.") is None

def test_pubmed_lookup_returns_dict(monkeypatch):
    from tools.pdf_enrich import pubmed_lookup
    import urllib.request
    fake = b'{"result":{"30049270":{"title":"Information Behavior","authors":[{"name":"Bates MJ"}],"pubdate":"2017 Nov","fulljournalname":"Annual Review","elocationid":"doi: 10.1002/asi.23681"}}}'
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    result = pubmed_lookup("30049270")
    assert result is not None
    assert result["title"] == "Information Behavior"
    assert result["author"] == "Bates"
    assert result["year"] == 2017
    assert result["type"] == "journal-article"

def test_pubmed_lookup_returns_none_on_missing_result(monkeypatch):
    from tools.pdf_enrich import pubmed_lookup
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{"result":{}}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert pubmed_lookup("99999999") is None
```

- [ ] **Step 2: Tests ausfuehren, muessen FAIL**
```
python -m pytest tests/test_pdf_enrich.py::test_extract_pmid_finds_pmid -v
```

- [ ] **Step 3: `_PMID_RE` + `extract_pmid()` + `pubmed_lookup()` implementieren** (nach `arxiv_lookup`)

```python
_PMID_RE = re.compile(r"PMID[:\s]+(\d{6,9})", re.IGNORECASE)

def extract_pmid(text: str) -> str | None:
    """Extrahiert erste PubMed-ID per Regex. Gibt ID-String oder None zurueck."""
    m = _PMID_RE.search(text)
    return m.group(1) if m else None

_PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def pubmed_lookup(pmid: str) -> dict | None:
    """Fragt PubMed E-Utilities mit PMID ab. Gibt normalisiertes Dict oder None zurueck."""
    params = urllib.parse.urlencode({"db": "pubmed", "id": pmid, "retmode": "json"})
    url = f"{_PUBMED_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError):
        return None
    result = (data.get("result") or {}).get(pmid)
    if not result:
        return None
    title = result.get("title", "")
    authors = result.get("authors") or []
    first_author = ""
    if authors:
        name = authors[0].get("name", "")
        first_author = name.split()[0] if name else ""
    pubdate = result.get("pubdate", "")
    year_m = re.search(r"\d{4}", pubdate)
    year = int(year_m.group()) if year_m else None
    eloc = result.get("elocationid", "")
    doi_m = re.search(r"10\.\d{4,9}/\S+", eloc)
    doi = doi_m.group().rstrip(".,;)") if doi_m else ""
    return {"title": title, "author": first_author, "year": year, "doi": doi, "type": "journal-article"}
```

- [ ] **Step 4: Tests ausfuehren (37 passed)**
```
python -m pytest tests/test_pdf_enrich.py -v
```

- [ ] **Step 5: Commit**
```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): PMID extraction + PubMed lookup (task-5)"
```

---

### Task 6: Vollstaendige Kaskade in enrich() einbauen

**Files:**
- Modify: `tools/pdf_enrich.py` — `enrich()` Docstring + Kaskade (Z. ~171)

- [ ] **Step 1: Failing Test schreiben** (APPENDEN)

```python
def test_enrich_uses_embedded_metadata_first(tmp_path, monkeypatch):
    """enrich() nutzt eingebettete PDF-Metadaten wenn Author + Title vorhanden."""
    from tools.pdf_enrich import enrich
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {"/Title": "Information Behavior", "/Author": "Marcia Bates", "/Subject": "2017"}
    monkeypatch.setattr("tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    meta = enrich(pdf, dry_run=True)
    assert meta is not None
    assert meta["title"] == "Information Behavior"
    assert meta["author"] == "Bates"

def test_enrich_uses_isbn_when_no_doi(tmp_path, monkeypatch):
    """enrich() findet ISBN wenn kein DOI und keine eingebetteten Metadaten."""
    from tools.pdf_enrich import enrich
    import urllib.request
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {}
    monkeypatch.setattr("tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    monkeypatch.setattr("tools.pdf_enrich.extract_text",
                        lambda path, max_pages=3: "ISBN 978-0-596-51774-8\nBook content.")
    fake_ol = b'{"ISBN:9780596517748":{"title":"JavaScript: The Good Parts","authors":[{"name":"Douglas Crockford"}],"publish_date":"2008"}}'
    class FakeResp:
        def read(self): return fake_ol
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF")
    meta = enrich(pdf, dry_run=True)
    assert meta is not None
    assert meta["author"] == "Crockford"
    assert meta["type"] == "book"
```

- [ ] **Step 2: Tests ausfuehren, muessen FAIL**
```
python -m pytest tests/test_pdf_enrich.py::test_enrich_uses_embedded_metadata_first -v
```

- [ ] **Step 3: `enrich()` vollstaendig ersetzen**

Ersetze die gesamte `enrich()`-Funktion (von `def enrich(` bis zum abschliessenden `return meta`):

```python
def enrich(pdf_path: Path, dry_run: bool = False, llm_fallback: bool = False) -> dict | None:
    """Haupt-Pipeline: EmbeddedMeta -> DOI -> ISBN -> arXiv -> PMID -> Titel(OpenAlex) -> LLM -> Rename.

    Gibt Metadaten-Dict zurueck oder None wenn nichts gefunden.
    Bei dry_run=True: kein Umbenennen, nur Metadaten zurueckgeben.
    """
    print(f"[pdf-enrich] {pdf_path.name}")

    # Stage 0: Eingebettete PDF-Metadaten
    meta = read_pdf_metadata(pdf_path)
    if meta:
        print(f"  -> Eingebettete Metadaten: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Text-Extraktion (fuer Stages 1-4 benoetigt)
    text = None
    if not meta:
        text = extract_text(pdf_path)
        if is_scanned(text):
            print("  -> Gescannte PDF erkannt")
            if ocr_available():
                print("  -> OCR via ocrmypdf...")
                ocr_path = run_ocr(pdf_path)
                if ocr_path:
                    text = extract_text(ocr_path)
                    if not dry_run:
                        pdf_path = ocr_path
            if is_scanned(text):
                print("  -> Kein Text extrahierbar -- uebersprungen", file=sys.stderr)
                return None

    # Stage 1: DOI -> CrossRef
    if not meta:
        doi = extract_doi(text)
        if doi:
            print(f"  -> DOI gefunden: {doi}")
            meta = crossref_lookup(doi)
            if meta:
                print(f"  -> CrossRef: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 2: ISBN -> Open Library -> Google Books
    if not meta:
        isbn = extract_isbn(text)
        if isbn:
            print(f"  -> ISBN gefunden: {isbn}")
            meta = open_library_lookup(isbn)
            if meta:
                print(f"  -> Open Library: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")
            if not meta:
                meta = google_books_lookup(isbn)
                if meta:
                    print(f"  -> Google Books: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 3: arXiv-ID -> arXiv API
    if not meta:
        arxiv_id = extract_arxiv_id(text)
        if arxiv_id:
            print(f"  -> arXiv-ID gefunden: {arxiv_id}")
            meta = arxiv_lookup(arxiv_id)
            if meta:
                print(f"  -> arXiv: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 4: PMID -> PubMed
    if not meta:
        pmid = extract_pmid(text)
        if pmid:
            print(f"  -> PMID gefunden: {pmid}")
            meta = pubmed_lookup(pmid)
            if meta:
                print(f"  -> PubMed: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 5: Titel-Suche -> OpenAlex
    if not meta:
        title_guess = next((l.strip() for l in text.splitlines() if len(l.strip()) > 20), "")
        if title_guess:
            print(f"  -> Kein ID, suche Titel: '{title_guess[:60]}'")
            meta = openalex_title_search(title_guess)
            if meta:
                print(f"  -> OpenAlex: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 6: LLM-Fallback
    if not meta and llm_fallback:
        print("  -> LLM-Fallback (Haiku)...")
        meta = _llm_extract(text[:3000])

    if not meta:
        print("  -> Keine Metadaten gefunden -- uebersprungen", file=sys.stderr)
        return None

    new_path = rename_pdf(pdf_path, meta, dry_run=dry_run)
    if dry_run:
        print(f"  -> [dry-run] wuerde umbenennen zu: {new_path.name}")
    else:
        print(f"  -> Umbenannt: {new_path.name}")
        _write_pdf_metadata(new_path, meta)

    return meta
```

- [ ] **Step 4: Tests ausfuehren (39 passed)**
```
python -m pytest tests/test_pdf_enrich.py -v
```

- [ ] **Step 5: Smoke-Test**
```
python tools/pdf_enrich.py --dry-run "C:\Users\tillq\OneDrive\Dokumente\Literatur\Bates - 2017 - Information Behavior.pdf"
```

- [ ] **Step 6: Commit**
```
git add tools/pdf_enrich.py tests/test_pdf_enrich.py
git commit -m "feat(pdf-enrich): full enrichment cascade Stage 0-6 (task-6)"
```

---

## Self-Review

**Spec-Coverage:**
- [x] Eingebettete PDF-Metadaten lesen (Stage 0, pypdf reader.metadata)
- [x] ISBN-Regex + Open Library + Google Books (Stage 2)
- [x] arXiv-ID-Regex + arXiv Atom-API (Stage 3)
- [x] PMID-Regex + PubMed E-Utilities (Stage 4)
- [x] Kaskade: EmbeddedMeta -> DOI -> ISBN -> arXiv -> PMID -> Titel -> LLM
- [x] Text-Extraktion nur wenn eingebettete Metadaten fehlen (Performance)
- [x] Keine neue Pflicht-Dependency
- [x] Alle APIs kostenlos, kein Auth

**Bewusst offen:**
- ISSN fuer Zeitschriften-Issues: kein freies API mit Einzelartikel-Lookup
- Zeitungsartikel: keine freie Universal-API
- ISO/DIN Normen: Regex-Erkennung waere moeglich, API fragil

**Placeholder-Check:** Keine TBDs.

**Type-Konsistenz:**
- `read_pdf_metadata()`, `open_library_lookup()`, `google_books_lookup()`, `arxiv_lookup()`, `pubmed_lookup()` -> alle `dict | None`
- `extract_isbn()`, `extract_arxiv_id()`, `extract_pmid()` -> alle `str | None`
- Normalisiertes Dict: immer keys `title`, `author`, `year`, `doi`, `type`