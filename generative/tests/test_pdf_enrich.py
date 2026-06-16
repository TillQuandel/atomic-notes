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

def test_title_match_confident_rejects_weak_overlap():
    """Generischer Guess matcht fremden Titel mit nur 1 gemeinsamen Wort -> verwerfen.
    Realer Bug-Fall: 'zettelkasten primer' vs. spanischer Torres-Salinas-Titel."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident(
        "zettelkasten primer",
        "Curso de escritura académica. Tomar notas con el método Zettelkasten y Zotero",
    ) is False


def test_title_match_confident_accepts_exact():
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident("Information Behavior", "Information Behavior") is True


def test_title_match_confident_accepts_subtitle_extension():
    """Guess ist Haupttitel, Treffer hat Untertitel -> trotzdem vertrauenswürdig."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident(
        "Information Search Process",
        "The Information Search Process: A Cognitive Approach",
    ) is True


def test_title_match_confident_rejects_single_generic_token():
    """Ein gemeinsames bedeutungstragendes Wort reicht nicht."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident("information", "Information Behavior Research") is False


def test_title_match_confident_handles_none_result():
    """OpenAlex kann title=null liefern -> kein Crash, sondern False."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident("Information Behavior", None) is False


def test_title_match_confident_folds_accents():
    """Akzent-Varianten gelten als gleich (método == metodo)."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident("Método Único", "Metodo Unico") is True


def test_title_match_confident_rejects_generic_two_token_in_long_foreign_title():
    """#41-Restrisiko R1: 2 generische Query-Tokens sind Präfix eines LÄNGEREN
    fremden Haupttitels. Forward-Containment allein (beide Wörter drin = 1.0) reicht
    nicht — der Haupttitel des Treffers (vor dem Untertitel) darf nicht wesentlich
    mehr als die Query-Tokens enthalten. Fail-closed."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident(
        "Information Behavior",
        "Information Behavior in Everyday Contexts: A Study of Organizational "
        "Knowledge Workers and Their Information Practices",
    ) is False
    assert _title_match_confident(
        "Knowledge Management",
        "Knowledge Management Systems in Modern Enterprise Architecture Design",
    ) is False


def test_title_match_confident_accepts_short_main_title_long_subtitle():
    """#41-R1-Regression-Schutz: kurzer Haupttitel + langer Untertitel ist legitim.
    Reverse-Containment muss gegen den HAUPTTITEL (vor ':') prüfen, nicht den ganzen
    String — sonst werden kanonische Werke fälschlich verworfen.
    Belegte Titel: Wenger 1998 (Communities of Practice), Lave & Wenger 1991
    (Situated Learning)."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident(
        "Communities of Practice",
        "Communities of Practice: Learning, Meaning, and Identity",
    ) is True
    assert _title_match_confident(
        "Situated Learning",
        "Situated Learning: Legitimate Peripheral Participation",
    ) is True


def test_title_match_confident_accepts_colon_subtitle_without_space():
    """#41-R1: Untertitel-Trenner ':' wird auch ohne folgendes Leerzeichen erkannt
    (OpenAlex normalisiert ': ' nicht immer)."""
    from generative.tools.pdf_enrich import _title_match_confident
    assert _title_match_confident(
        "Communities of Practice",
        "Communities of Practice:Learning, Meaning, and Identity",
    ) is True


def test_enrich_discards_weak_openalex_match(tmp_path, monkeypatch):
    """enrich() verwirft OpenAlex-Treffer mit schwachem Titel-Match statt fehlzuattribuieren."""
    from generative.tools.pdf_enrich import enrich
    import urllib.request
    from unittest.mock import MagicMock
    mock_reader = MagicMock()
    mock_reader.metadata = {}
    monkeypatch.setattr("generative.tools.pdf_enrich.PdfReader", lambda *a, **kw: mock_reader)
    monkeypatch.setattr("generative.tools.pdf_enrich.extract_text",
                        lambda path, max_pages=3: "Zettelkasten Primer\nEine kurze Einfuehrung in atomare Notizen.")
    fake = b'{"results":[{"title":"Curso de escritura acad\xc3\xa9mica. Tomar notas con el m\xc3\xa9todo Zettelkasten y Zotero","authorships":[{"author":{"display_name":"Daniel Torres-Salinas"}}],"publication_year":2024,"doi":"https://doi.org/10.3145/infonomy.24.055","type":"article"}]}'
    class FakeResp:
        def read(self): return fake
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    pdf = tmp_path / "zettelkasten-primer.pdf"
    pdf.write_bytes(b"%PDF")
    meta = enrich(pdf, dry_run=True)
    assert meta is None


def test_enrich_rename_false_keeps_original_file(tmp_path, monkeypatch):
    """enrich(rename=False) mutiert die Eingabedatei nicht (Pipeline-Schutz),
    auch ohne dry_run."""
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
    meta = enrich(pdf, rename=False)
    assert meta is not None
    assert meta["author"] == "Bates"
    assert pdf.exists()  # Original unverändert
    assert not (tmp_path / "Bates - 2017 - Information Behavior.pdf").exists()


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


def test_cli_does_not_rename_by_default(tmp_path, monkeypatch):
    """#41-Restrisiko R3: Standalone-CLI mutiert getrackte PDFs nicht mehr.
    Umbenennen + In-PDF-Metadaten-Write ist jetzt Opt-in (rename=False per Default)."""
    from generative.tools import pdf_enrich
    captured = {}

    def fake_enrich(path, dry_run=False, llm_fallback=False, rename=True):
        captured["rename"] = rename
        return {"author": "X", "year": 2020, "title": "Y"}

    monkeypatch.setattr(pdf_enrich, "enrich", fake_enrich)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(sys, "argv", ["pdf_enrich", str(pdf)])
    pdf_enrich.main()
    assert captured["rename"] is False


def test_cli_renames_with_flag(tmp_path, monkeypatch):
    """--rename aktiviert das Umbenennen explizit (Opt-in)."""
    from generative.tools import pdf_enrich
    captured = {}

    def fake_enrich(path, dry_run=False, llm_fallback=False, rename=True):
        captured["rename"] = rename
        return {"author": "X", "year": 2020, "title": "Y"}

    monkeypatch.setattr(pdf_enrich, "enrich", fake_enrich)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(sys, "argv", ["pdf_enrich", "--rename", str(pdf)])
    pdf_enrich.main()
    assert captured["rename"] is True


# --- #41-Restrisiko R2 + Geschwister: ID-Stages nur aus dem Kopfbereich ---------
# Bug-Klasse: extract_doi/isbn/arxiv/pmid nehmen alle den ersten Regex-Treffer aus
# demselben 10-Seiten-Text. Eine ZITIERTE ID (Fußnote/Referenz auf einer späteren
# Seite) wird so zur Dokument-Quelle. Fix: IDs nur aus dem Kopfbereich (erste Seite)
# ziehen — für alle vier Extraktoren konsistent.

def _front_matter_aware_extract(header: str, full: str):
    """Mock für extract_text: die volle 10-Seiten-Extraktion sieht die zitierte ID,
    die Kopfbereich-Extraktion (weniger Seiten) nicht. Simuliert eine zitierte ID, die
    erst auf einer späteren Seite steht."""
    def fake(path, max_pages=3):
        return full if max_pages >= 10 else header
    return fake


def test_enrich_ignores_cited_doi_on_later_page(tmp_path, monkeypatch):
    """#41-R2: eine zitierte DOI (Literaturverzeichnis, spätere Seite) darf nicht zur
    Dokument-Quelle werden. Kopfbereich hat keine DOI -> keine Attribution."""
    from generative.tools import pdf_enrich
    monkeypatch.setattr(pdf_enrich, "extract_text", _front_matter_aware_extract(
        header="Eine kurze Notiz\nVon der Autorin\nEinleitende Bemerkungen ohne Kennung.",
        full="Eine kurze Notiz\nVon der Autorin\nEinleitende Bemerkungen ohne Kennung.\n"
             "Literatur\n[1] Bates 2017. doi:10.1002/asi.23681"))
    monkeypatch.setattr(pdf_enrich, "openalex_title_search", lambda *a, **k: None)
    called = {}

    def spy(doi):
        called["doi"] = doi
        return {"author": "Bates", "year": 2017, "title": "Information Behavior",
                "doi": doi, "type": "journal-article"}
    monkeypatch.setattr(pdf_enrich, "crossref_lookup", spy)
    pdf = tmp_path / "kurze-notiz.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is None
    assert "doi" not in called  # zitierte DOI nie nachgeschlagen


def test_enrich_ignores_cited_isbn_on_later_page(tmp_path, monkeypatch):
    """#41-R2-Geschwister: zitierte ISBN auf späterer Seite darf nicht attribuieren."""
    from generative.tools import pdf_enrich
    monkeypatch.setattr(pdf_enrich, "extract_text", _front_matter_aware_extract(
        header="Kapiteltitel\nFließtext ohne Kennung.",
        full="Kapiteltitel\nFließtext ohne Kennung.\nLiteratur\nCrockford, ISBN 978-0-596-51774-8."))
    monkeypatch.setattr(pdf_enrich, "openalex_title_search", lambda *a, **k: None)
    called = {}

    def spy(isbn):
        called["isbn"] = isbn
        return {"author": "Crockford", "year": 2008, "title": "JavaScript",
                "doi": "", "type": "book"}
    monkeypatch.setattr(pdf_enrich, "open_library_lookup", spy)
    monkeypatch.setattr(pdf_enrich, "google_books_lookup", spy)
    pdf = tmp_path / "kapitel-auszug.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is None
    assert "isbn" not in called


def test_enrich_ignores_cited_arxiv_on_later_page(tmp_path, monkeypatch):
    """#41-R2-Geschwister: zitierte arXiv-ID auf späterer Seite darf nicht attribuieren."""
    from generative.tools import pdf_enrich
    monkeypatch.setattr(pdf_enrich, "extract_text", _front_matter_aware_extract(
        header="Seminararbeit\nÜberblick ohne Kennung.",
        full="Seminararbeit\nÜberblick ohne Kennung.\nReferenzen\n[3] Vaswani et al., arXiv:1706.03762."))
    monkeypatch.setattr(pdf_enrich, "openalex_title_search", lambda *a, **k: None)
    called = {}

    def spy(arxiv_id):
        called["arxiv"] = arxiv_id
        return {"author": "Vaswani", "year": 2017, "title": "Attention",
                "doi": "", "type": "preprint"}
    monkeypatch.setattr(pdf_enrich, "arxiv_lookup", spy)
    pdf = tmp_path / "seminararbeit-text.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is None
    assert "arxiv" not in called


def test_enrich_ignores_cited_pmid_on_later_page(tmp_path, monkeypatch):
    """#41-R2-Geschwister: zitierte PMID auf späterer Seite darf nicht attribuieren."""
    from generative.tools import pdf_enrich
    monkeypatch.setattr(pdf_enrich, "extract_text", _front_matter_aware_extract(
        header="Studienprotokoll\nMethodischer Überblick ohne Kennung.",
        full="Studienprotokoll\nMethodischer Überblick ohne Kennung.\nReferences\n[7] Bates MJ. PMID: 30049270."))
    monkeypatch.setattr(pdf_enrich, "openalex_title_search", lambda *a, **k: None)
    called = {}

    def spy(pmid):
        called["pmid"] = pmid
        return {"author": "Bates", "year": 2017, "title": "Information Behavior",
                "doi": "", "type": "journal-article"}
    monkeypatch.setattr(pdf_enrich, "pubmed_lookup", spy)
    pdf = tmp_path / "studienprotokoll-text.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is None
    assert "pmid" not in called


def test_enrich_uses_doi_from_front_matter(tmp_path, monkeypatch):
    """Gegenprobe zu R2: eine DOI im Kopfbereich (erste Seite) wird weiterhin genutzt,
    und zwar die Kopf-DOI — nicht eine zitierte DOI weiter unten."""
    from generative.tools import pdf_enrich
    monkeypatch.setattr(pdf_enrich, "extract_text", _front_matter_aware_extract(
        header="Titel\nAutor\nhttps://doi.org/10.1002/asi.23681",
        full="Titel\nAutor\nhttps://doi.org/10.1002/asi.23681\nLiteratur\ndoi:10.9999/cited"))

    def spy(doi):
        return {"author": "Bates", "year": 2017, "title": "Information Behavior",
                "doi": doi, "type": "journal-article"}
    monkeypatch.setattr(pdf_enrich, "crossref_lookup", spy)
    pdf = tmp_path / "titel-doc.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is not None
    assert meta["doi"] == "10.1002/asi.23681"  # Kopf-DOI, nicht 10.9999/cited


def test_enrich_book_chapter_doi_ignores_cited_isbn(tmp_path, monkeypatch):
    """#41-R2 (Buchkapitel-Branch von Stage 1): bei einem Buchkapitel-DOI im Kopf darf
    eine ZITIERTE ISBN (spätere Seite) nicht den ISBN-Lookup-Umweg triggern und die
    Quelle auf das zitierte Buch verfälschen. Header hat keine ISBN -> Kapitel-Meta bleibt."""
    from generative.tools import pdf_enrich
    monkeypatch.setattr(pdf_enrich, "extract_text", _front_matter_aware_extract(
        header="Buchkapitel-Titel\nAutor\ndoi:10.1007/978-3-540-chapter",
        full="Buchkapitel-Titel\nAutor\ndoi:10.1007/978-3-540-chapter\nLiteratur\nISBN 978-0-596-51774-8."))
    monkeypatch.setattr(pdf_enrich, "crossref_lookup",
        lambda doi: {"author": "Kapitelautor", "year": 2010, "title": "Das Kapitel",
                     "doi": doi, "type": "book-chapter"})

    def isbn_spy(isbn):
        return {"author": "FremderAutor", "year": 2008, "title": "Zitiertes Buch",
                "doi": "", "type": "book"}
    monkeypatch.setattr(pdf_enrich, "open_library_lookup", isbn_spy)
    monkeypatch.setattr(pdf_enrich, "google_books_lookup", isbn_spy)
    pdf = tmp_path / "buchkapitel.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is not None
    assert meta["author"] == "Kapitelautor"  # Kapitel-Meta, nicht das zitierte Buch


def test_enrich_header_from_ocr_source(tmp_path, monkeypatch):
    """#41-R2 (OCR-Pfad): bei einem gescannten PDF wird der Kopfbereich aus der OCR-Quelle
    gezogen, nicht aus dem (leeren) Original. Schützt das source_path-Tracking — unter
    dry_run=True bleibt pdf_path das Original, also muss header_text aus source_path (ocr)
    kommen, sonst geht die Kopf-DOI verloren."""
    from generative.tools import pdf_enrich
    ocr_pdf = tmp_path / "scan.ocr.pdf"

    def fake_extract(path, max_pages=3):
        if str(path).endswith("scan.ocr.pdf"):
            if max_pages >= 10:
                return ("Titel\nAutor\nhttps://doi.org/10.1002/asi.23681\n"
                        "Literatur\ndoi:10.9999/cited")
            return "Titel\nAutor\nhttps://doi.org/10.1002/asi.23681"
        return ""  # Original ist gescannt -> leer
    monkeypatch.setattr(pdf_enrich, "extract_text", fake_extract)
    monkeypatch.setattr(pdf_enrich, "ocr_available", lambda: True)
    monkeypatch.setattr(pdf_enrich, "run_ocr", lambda p: ocr_pdf)
    called = {}

    def spy(doi):
        called["doi"] = doi
        return {"author": "Bates", "year": 2017, "title": "Information Behavior",
                "doi": doi, "type": "journal-article"}
    monkeypatch.setattr(pdf_enrich, "crossref_lookup", spy)
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    meta = pdf_enrich.enrich(pdf, dry_run=True)
    assert meta is not None
    assert called["doi"] == "10.1002/asi.23681"  # Kopf-DOI aus OCR-Quelle, nicht 10.9999/cited
