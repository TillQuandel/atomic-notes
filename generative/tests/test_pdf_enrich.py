import sys
from pathlib import Path

def test_extract_text_returns_string(tmp_path):
    """extract_text() gibt String zurück (auch bei leerer PDF)."""
    from generative.tools.pdf_enrich import extract_text
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
    from generative.tools.pdf_enrich import is_scanned
    assert is_scanned("") is True
    assert is_scanned("   \n  ") is True
    assert is_scanned("Some text here") is False


def test_extract_doi_finds_doi_in_text():
    from generative.tools.pdf_enrich import extract_doi
    text = "See also doi:10.1016/j.ipm.2019.05.003 for details."
    assert extract_doi(text) == "10.1016/j.ipm.2019.05.003"


def test_extract_doi_finds_https_doi():
    from generative.tools.pdf_enrich import extract_doi
    text = "Available at https://doi.org/10.3389/fpsyg.2019.02730"
    assert extract_doi(text) == "10.3389/fpsyg.2019.02730"


def test_extract_doi_returns_none_when_missing():
    from generative.tools.pdf_enrich import extract_doi
    assert extract_doi("No DOI here, just text.") is None


def test_extract_isbn_finds_isbn13():
    from generative.tools.pdf_enrich import extract_isbn
    text = "ISBN: 978-3-16-148410-0"
    assert extract_isbn(text) == "9783161484100"


def test_extract_isbn_finds_isbn10():
    from generative.tools.pdf_enrich import extract_isbn
    text = "ISBN 0-596-51774-2 is the identifier."
    assert extract_isbn(text) == "0596517742"


def test_extract_isbn_returns_none_when_missing():
    from generative.tools.pdf_enrich import extract_isbn
    assert extract_isbn("No ISBN here.") is None


def test_extract_isbn_strips_hyphens_and_spaces():
    from generative.tools.pdf_enrich import extract_isbn
    text = "ISBN 978 0 596 51774 8"
    assert extract_isbn(text) == "9780596517748"


def test_crossref_lookup_returns_dict_on_valid_doi(monkeypatch):
    from generative.tools.pdf_enrich import crossref_lookup
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
    from generative.tools.pdf_enrich import crossref_lookup
    import urllib.request, urllib.error
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(urllib.error.HTTPError(None, 404, "Not Found", {}, None)))
    assert crossref_lookup("10.9999/fake") is None


def test_openalex_title_search_returns_result(monkeypatch):
    from generative.tools.pdf_enrich import openalex_title_search
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
    from generative.tools.pdf_enrich import openalex_title_search
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{"results":[]}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert openalex_title_search("zzz_nonexistent_zzz") is None


def test_build_filename_standard():
    from generative.tools.pdf_enrich import build_filename
    meta = {"author": "Bates", "year": 2017, "title": "Information Behavior"}
    assert build_filename(meta) == "Bates - 2017 - Information Behavior.pdf"


def test_build_filename_truncates_long_title():
    from generative.tools.pdf_enrich import build_filename
    long = "A" * 150
    result = build_filename({"author": "X", "year": 2020, "title": long})
    assert len(result) <= 120
    assert result.endswith(".pdf")


def test_build_filename_sanitizes_special_chars():
    from generative.tools.pdf_enrich import build_filename
    meta = {"author": "Müller", "year": 2021, "title": "Test/Paper: Results"}
    result = build_filename(meta)
    assert "/" not in result
    assert ":" not in result


def test_rename_pdf_renames_file(tmp_path):
    from generative.tools.pdf_enrich import rename_pdf
    src = tmp_path / "paper123.pdf"
    src.write_bytes(b"%PDF-1.4")
    meta = {"author": "Bates", "year": 2017, "title": "Information Behavior"}
    new_path = rename_pdf(src, meta, dry_run=False)
    assert new_path.name == "Bates - 2017 - Information Behavior.pdf"
    assert new_path.exists()
    assert not src.exists()


def test_rename_pdf_dry_run_does_not_rename(tmp_path):
    from generative.tools.pdf_enrich import rename_pdf
    src = tmp_path / "paper123.pdf"
    src.write_bytes(b"%PDF-1.4")
    meta = {"author": "Bates", "year": 2017, "title": "Information Behavior"}
    new_path = rename_pdf(src, meta, dry_run=True)
    assert src.exists()  # Original bleibt
    assert new_path.name == "Bates - 2017 - Information Behavior.pdf"


def test_ocr_available_returns_bool():
    from generative.tools.pdf_enrich import ocr_available
    result = ocr_available()
    assert isinstance(result, bool)


def test_run_ocr_returns_path_when_available(tmp_path, monkeypatch):
    from generative.tools.pdf_enrich import run_ocr
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ocrmypdf" if cmd == "ocrmypdf" else None)

    def fake_run(cmd, capture_output=False, timeout=None):
        class R:
            returncode = 0
        # cmd ist eine Liste wie ["ocrmypdf", "--quiet", "--skip-text", "in.pdf", "out.pdf"]
        # Das letzte Element ist der Output-Pfad
        out_path = cmd[-1]
        import pathlib
        pathlib.Path(out_path).write_bytes(b"%PDF-1.4 OCR")
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.4")
    result = run_ocr(src)
    assert result is not None


def test_enrich_returns_meta_for_pdf_with_doi(tmp_path, monkeypatch):
    """enrich() gibt Metadaten zurück wenn DOI im Text gefunden und CrossRef antwortet."""
    from generative.tools.pdf_enrich import enrich
    import urllib.request

    monkeypatch.setattr("generative.tools.pdf_enrich.extract_text",
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


def test_read_pdf_metadata_returns_dict(tmp_path, monkeypatch):
    from generative.tools.pdf_enrich import read_pdf_metadata
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {"/Title": "Information Behavior", "/Author": "Bates", "/Subject": "2017"}
    monkeypatch.setattr("generative.tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    result = read_pdf_metadata(tmp_path / "paper.pdf")
    assert result is not None
    assert result["title"] == "Information Behavior"
    assert result["author"] == "Bates"
    assert result["year"] == 2017


def test_read_pdf_metadata_returns_none_when_no_author(tmp_path, monkeypatch):
    from generative.tools.pdf_enrich import read_pdf_metadata
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {"/Title": "Some Title"}
    monkeypatch.setattr("generative.tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    assert read_pdf_metadata(tmp_path / "paper.pdf") is None


def test_read_pdf_metadata_returns_none_when_empty(tmp_path, monkeypatch):
    from generative.tools.pdf_enrich import read_pdf_metadata
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {}
    monkeypatch.setattr("generative.tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    assert read_pdf_metadata(tmp_path / "paper.pdf") is None


def test_open_library_lookup_returns_dict(monkeypatch):
    from generative.tools.pdf_enrich import open_library_lookup
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
    from generative.tools.pdf_enrich import open_library_lookup
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert open_library_lookup("9999999999999") is None

def test_google_books_lookup_returns_dict(monkeypatch):
    from generative.tools.pdf_enrich import google_books_lookup
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
    from generative.tools.pdf_enrich import google_books_lookup
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{"items":[]}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert google_books_lookup("9999999999999") is None


def test_extract_arxiv_id_finds_new_format():
    from generative.tools.pdf_enrich import extract_arxiv_id
    text = "See arXiv:2301.12345 for the preprint."
    assert extract_arxiv_id(text) == "2301.12345"

def test_extract_arxiv_id_finds_old_format():
    from generative.tools.pdf_enrich import extract_arxiv_id
    text = "Available at arXiv:cs.AI/0612072"
    assert extract_arxiv_id(text) == "cs.AI/0612072"

def test_extract_arxiv_id_returns_none_when_missing():
    from generative.tools.pdf_enrich import extract_arxiv_id
    assert extract_arxiv_id("No arXiv ID here.") is None

def test_arxiv_lookup_returns_dict(monkeypatch):
    from generative.tools.pdf_enrich import arxiv_lookup
    import urllib.request
    fake = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
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
    from generative.tools.pdf_enrich import arxiv_lookup
    import urllib.request
    fake = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    assert arxiv_lookup("9999.99999") is None


def test_extract_pmid_finds_pmid():
    from generative.tools.pdf_enrich import extract_pmid
    text = "PMID: 30049270 — see also the supplementary."
    assert extract_pmid(text) == "30049270"

def test_extract_pmid_returns_none_when_missing():
    from generative.tools.pdf_enrich import extract_pmid
    assert extract_pmid("No PMID here.") is None

def test_pubmed_lookup_returns_dict(monkeypatch):
    from generative.tools.pdf_enrich import pubmed_lookup
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
    from generative.tools.pdf_enrich import pubmed_lookup
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: type("R", (), {"read": lambda s: b'{"result":{}}', "__enter__": lambda s: s, "__exit__": lambda s, *a: None})())
    assert pubmed_lookup("99999999") is None


def test_enrich_uses_embedded_metadata_first(tmp_path, monkeypatch):
    """enrich() nutzt eingebettete PDF-Metadaten wenn Author + Title vorhanden."""
    from generative.tools.pdf_enrich import enrich
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {"/Title": "Information Behavior", "/Author": "Marcia Bates", "/Subject": "2017"}
    monkeypatch.setattr("generative.tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    meta = enrich(pdf, dry_run=True)
    assert meta is not None
    assert meta["title"] == "Information Behavior"
    assert meta["author"] == "Bates"

def test_enrich_uses_isbn_when_no_doi(tmp_path, monkeypatch):
    """enrich() findet ISBN wenn kein DOI und keine eingebetteten Metadaten."""
    from generative.tools.pdf_enrich import enrich
    import urllib.request
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {}
    monkeypatch.setattr("generative.tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    monkeypatch.setattr("generative.tools.pdf_enrich.extract_text",
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
