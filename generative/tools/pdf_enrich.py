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
import argparse
import datetime
import html
import json
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from shared.author_norm import _AUTHOR_SEP_RE, drop_institutional_coauthors

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore


def extract_text(pdf_path: Path, max_pages: int = 3) -> str:
    """Extrahiert Text aus den ersten max_pages Seiten via pypdf."""
    if PdfReader is None:
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


# #41-R2: Standard-Kennungen (DOI/ISBN/arXiv/PMID) werden NUR aus dem Kopfbereich
# gezogen, nicht aus den vollen 10 Seiten. Dokument-eigene IDs stehen auf der Titel-/
# Kopfseite; zitierte IDs (Fußnoten, Literaturverzeichnis) erscheinen erst im Body —
# ohne diese Beschränkung würde die erste *zitierte* ID zur falschen Dokument-Quelle.
# Trade-off: ISBNs auf einer separaten Impressum-/Copyright-Seite (S. 2-4 bei
# Voll-Büchern) werden bei =1 nicht erfasst; für den Korpus (Kapitel-Auszüge, Paper)
# vernachlässigbar, zitierte ISBNs sind selten. Bei Bedarf hochsetzen.
_FRONT_MATTER_PAGES = 1


_GARBAGE_AUTHORS = {
    "someauthor",
    "author",
    "unknown",
    "user",
    "administrator",
    "admin",
    "root",
    "owner",
    "creator",
    "microsoft",
    "adobe",
    "none",
    "n/a",
    "anonymous",
    "null",
}
_GARBAGE_TITLE_PREFIXES = re.compile(r"^(endnotes?[\s\d]|sometitle|untitled|document\b)", re.IGNORECASE)
_TITLE_FILE_EXTS = re.compile(r"\.(pdf|docx?|odt|pptx?)$", re.IGNORECASE)


def _plausible_author(s: str) -> bool:
    """Gibt False zurueck bei offensichtlichen Platzhaltern oder unbrauchbaren Werten."""
    s = s.strip()
    if len(s) < 2:
        return False
    if s.lower() in _GARBAGE_AUTHORS:
        return False
    if s.startswith("(") or s.endswith(")"):  # z.B. "(Hrsg.)"
        return False
    if re.fullmatch(r"[A-Z]\.?", s):  # Einzelbuchstabe wie "E." oder "E"
        return False
    return True


def _plausible_title(s: str) -> bool:
    """Gibt False zurueck bei offensichtlichen Platzhaltern oder Dateinamen."""
    s = s.strip()
    if len(s) < 3:
        return False
    if _GARBAGE_TITLE_PREFIXES.match(s):
        return False
    if _TITLE_FILE_EXTS.search(s):  # Dateiname als Titel z.B. "paper.pdf"
        return False
    return True


def read_pdf_metadata(pdf_path: Path) -> dict | None:
    """Liest eingebettete PDF-Metadaten (Info-Dictionary). Gibt normalisiertes Dict zurueck
    wenn Author + Title vorhanden und plausibel, sonst None."""
    if PdfReader is None:
        return None
    try:
        info = PdfReader(str(pdf_path)).metadata or {}
    except Exception:
        return None
    title = (info.get("/Title") or "").strip()
    author = (info.get("/Author") or "").strip()
    if not (title and author):
        return None
    if not (_plausible_author(author) and _plausible_title(title)):
        return None
    subject = info.get("/Subject") or ""
    year_match = re.search(r"\d{4}", subject)
    year = int(year_match.group()) if year_match else None
    doi = info.get("/Keywords") or ""
    doi = doi if doi.startswith("10.") else ""
    return {
        "title": title,
        "author": author.split()[-1] if author else "",
        "year": year,
        "doi": doi,
        "type": (info.get("/Creator") or ""),
    }


_DOI_RE = re.compile(
    r"(?:doi[:\s./]*|https?://doi\.org/|DOI[:\s.]*)"
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
    raw = author_m.group(1).strip() if author_m else ""
    author = raw.split(",")[0].strip() if "," in raw else raw.split()[-1] if raw else ""
    year = int(date_m.group(1)) if date_m else None
    doi_m = re.search(r"<arxiv:doi[^>]*>([^<]+)</arxiv:doi>", entry)
    doi = doi_m.group(1).strip() if doi_m else ""
    return {"title": title, "author": author, "year": year, "doi": doi, "type": "preprint"}


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
    first_author = (authors[0] or {}).get("family", "") if authors else ""
    published = data.get("published", {}).get("date-parts", [[None]])
    year = published[0][0] if published and published[0] else None
    return {
        "title": title,
        "author": first_author,
        "year": int(year) if year else None,
        "doi": data.get("DOI", ""),
        "type": data.get("type", ""),
    }


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


_OPENALEX_BASE = "https://api.openalex.org/works"

# Title-Match-Gate: Die OpenAlex-Titel-Suche ist der einzige Enrichment-Pfad ohne
# harte ID (DOI/ISBN/arXiv/PMID). Ein generischer Guess matcht sonst einen fremden
# Treffer und schreibt dessen Autor/Jahr/DOI in die Note (Fabrikation). Fail-closed.
_MIN_SIGNIFICANT_TOKEN_LEN = 4  # kürzere Wörter (de, the, y, …) tragen keine Titel-Identität
_MIN_TITLE_TOKEN_OVERLAP = 2  # mind. 2 bedeutungstragende Wörter müssen übereinstimmen
_MIN_TITLE_CONTAINMENT = 0.5  # Anteil der Query-Tokens, der im Treffer vorkommen muss
# Reverse-Containment: Anteil der HAUPTTITEL-Tokens (vor dem Untertitel), die aus der
# Query stammen. Schützt gegen 2 generische Query-Wörter, die Präfix eines längeren
# fremden Haupttitels sind ("Information Behavior" ⊂ "Information Behavior in Everyday
# Contexts" → forward=1.0, reverse niedrig). Gemessen wird gegen den Haupttitel, NICHT
# den ganzen String — sonst würden kanonische Werke mit langem Untertitel fälschlich
# verworfen (Wenger "Communities of Practice: Learning, Meaning, and Identity").
_MIN_TITLE_REVERSE_CONTAINMENT = 0.6
# Untertitel-Trenner: Doppelpunkt (auch ohne folgendes Leerzeichen, OpenAlex
# normalisiert nicht immer) ODER Gedankenstrich/Bindestrich nur mit Whitespace auf
# BEIDEN Seiten — damit Komposita wie "Note-Taking"/"Self-Determination" intakt bleiben.
_SUBTITLE_SEP_RE = re.compile(r":\s*|\s+[–—-]\s+")


def _strip_diacritics(text: str) -> str:
    """Faltet Diakritika via NFKD + Combining-Strip (método==metodo, Yanartaş==Yanartas).
    Kein Lowercasing — das entscheidet der Aufrufer."""
    folded = unicodedata.normalize("NFKD", text)
    return "".join(c for c in folded if not unicodedata.combining(c))


def _significant_tokens(title: str) -> set[str]:
    """Lowercase-Wörter ab _MIN_SIGNIFICANT_TOKEN_LEN Zeichen (Satzzeichen/Ziffern raus).

    Akzent-gefaltet (método==metodo), damit Sprach-/Encoding-Varianten matchen.
    Defensiv gegen None (OpenAlex kann title=null liefern). Ziffern bleiben bewusst
    draußen — Jahreszahlen o.ä. als Match-Token würden Fehltreffer erzeugen.

    HTML/MathML aus dem Titel entfernen (#41-MED): OpenAlex liefert Markup teils
    literal (<i>, <span>), teils HTML-entity-kodiert (&lt;span&gt;). Tag-Namen mit
    ≥4 Zeichen würden sonst als bedeutungstragende Tokens die Containment-/Subset-
    Checks verfälschen (z.B. 'span' unterläuft r_main⊊q). Erst entkodieren, dann Tags
    strippen.
    """
    raw = re.sub(r"<[^>]+>", " ", html.unescape(str(title or "")))
    folded = _strip_diacritics(raw.lower())
    words = re.findall(r"[^\W\d_]+", folded, flags=re.UNICODE)
    return {w for w in words if len(w) >= _MIN_SIGNIFICANT_TOKEN_LEN}


def _title_match_confident(query: str, result_title: str) -> bool:
    """True wenn der OpenAlex-Treffer dem Suchtitel hinreichend ähnelt.

    Kriterien (alle nötig):
    - mind. _MIN_TITLE_TOKEN_OVERLAP gemeinsame bedeutungstragende Wörter,
    - Forward-Containment (Anteil der Query-Tokens im vollen Treffer) >=
      _MIN_TITLE_CONTAINMENT — Query-Wörter dürfen auch im Untertitel liegen,
    - Reverse-Containment (Anteil der HAUPTTITEL-Tokens aus der Query) >=
      _MIN_TITLE_REVERSE_CONTAINMENT — der Haupttitel des Treffers darf nicht
      wesentlich mehr als die Query enthalten (sonst ist es ein längeres fremdes Werk).
    Sprach-/Reihenfolge-robust durch Token-Mengen-Vergleich.
    """
    q = _significant_tokens(query)
    r_full = _significant_tokens(result_title)
    r_main = _significant_tokens(_SUBTITLE_SEP_RE.split(str(result_title or ""), maxsplit=1)[0])
    if not q or not r_main:
        return False
    # #41 (inverser R1-Fall): Treffer-Haupttitel ist eine echte Teilmenge der
    # spezifischeren Query (r_main ⊊ q) UND die Query trägt Tokens, die im GANZEN
    # Treffer fehlen (q - r_full ≠ ∅) — z.B. Query "Situated Learning Theory" gegen
    # Treffer "Situated Learning" (Lave & Wenger): "theory" steht nirgends im Treffer.
    # Forward- UND Reverse-Containment passieren beide (r_main ⊆ q → reverse = 1.0),
    # aber der generische Kurztitel identifiziert das Werk nicht; Autor/Jahr fehlt im
    # Title-Pfad zur Disambiguierung. OpenAlex speichert Titel praktisch immer voll
    # (inkl. Untertitel; 21 Live-Abfragen 2026-06-16), ein echter gekürzter Treffer
    # ist kaum legitim — fail-closed verwerfen. Die q-r_full-Bedingung schützt den
    # legitimen Volltitel-mit-Untertitel-Fall (Query enthält den Untertitel, q ⊆ r_full).
    if r_main < q and (q - r_full):
        return False
    if len(q & r_main) < _MIN_TITLE_TOKEN_OVERLAP:
        return False
    if len(q & r_full) / len(q) < _MIN_TITLE_CONTAINMENT:
        return False
    return len(q & r_main) / len(r_main) >= _MIN_TITLE_REVERSE_CONTAINMENT


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
        name_parts = authors[0].get("author", {}).get("display_name", "").split()
        first_author = name_parts[-1] if name_parts else ""
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    return {
        "title": item.get("title", ""),
        "author": first_author,
        "year": item.get("publication_year"),
        "doi": doi,
        "type": item.get("type", ""),
    }


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


def ocr_available() -> bool:
    """Gibt True zurueck wenn ocrmypdf verfuegbar ist (Binary oder Python-Modul)."""
    if shutil.which("ocrmypdf"):
        return True
    try:
        import ocrmypdf as _ocr  # noqa: F401

        return True
    except ImportError:
        return False


def run_ocr(pdf_path: Path) -> Path | None:
    """Fuehrt ocrmypdf auf PDF aus. Nutzt Binary wenn verfuegbar, sonst Python-Modul."""
    if not ocr_available():
        return None
    out_path = pdf_path.parent / f"_ocr_{pdf_path.name}"
    cmd = ["ocrmypdf"] if shutil.which("ocrmypdf") else [sys.executable, "-m", "ocrmypdf"]
    try:
        result = subprocess.run(
            # S6 (#150): PDF-Pfade absolutieren -> ein relativer Name mit
            # fuehrendem "-" kann nicht als ocrmypdf-Option fehlinterpretiert werden.
            cmd + ["--quiet", "--skip-text", str(Path(pdf_path).resolve()), str(Path(out_path).resolve())],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and out_path.exists():
            return out_path
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


_BOOK_PART_TYPES = {"book-chapter", "book-section", "book-part", "reference-entry"}

_HEADER_RE = re.compile(
    r"^(university|department|faculty|institute|journal of|proceedings of|vol\.|volume\s+\d"
    r"|transactions of|annals of|lecture notes in|report|working paper|advances in"
    r"|international conference on|j\.\s|proc\.|springer|elsevier|wiley|taylor|copyright"
    r"|issn|isbn|doi:|http|endnotes?[\s:]|elis[\s–-]|published by|copyright \d"
    r"|the picture can'?t be displayed|master thesis|bachelor thesis|degree project"
    r"|examensarbete|diplomarbeit)",
    re.IGNORECASE,
)

# Dateinamen-Erkennung: DOI, arXiv, Jahr-basiert, kein-Jahr
_FILENAME_DOI_RE = re.compile(r"^10[._]\d{4,9}[._]\S{3,}")
_FILENAME_ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_YEAR_BOUNDARY_RE = re.compile(r"(?<![0-9])(1[89]\d{2}|20[012]\d)(?![0-9])")
_SEPARATORS = [" - ", "_", "-", " "]
_ZOTERO_NO_YEAR_RE = re.compile(r"^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s,\.]+?)\s+-\s+([A-Z].{10,})$")


def _extract_doi_from_filename(stem: str) -> str | None:
    """Erkennt DOI-Muster im Dateinamen. Gibt normalisierten DOI zurueck oder None."""
    m = _FILENAME_DOI_RE.match(stem)
    if not m:
        return None
    doi = re.sub(r"[_]", "/", m.group(), count=1)
    doi = doi.rstrip(".")
    return doi if doi.startswith("10.") else None


def _extract_arxiv_from_filename(stem: str) -> str | None:
    """Erkennt arXiv-ID-Muster im Dateinamen. Gibt ID-String zurueck oder None."""
    m = _FILENAME_ARXIV_RE.match(stem)
    return m.group(1) if m else None


def _parse_filename_dynamic(pdf_path: Path) -> dict | None:
    """Parst Metadaten aus Dateiname dynamisch.

    Reihenfolge:
      1. Zotero 'Autor - Jahr - Titel' (strikt, primaer)
      2. Jahr-Anker mit Separator-Autoerkennung (_ - oder -)
      3. Zotero 'Autor - Titel' ohne Jahr als Fallback
    """
    stem = pdf_path.stem

    # Primaer: Zotero-Format " - " mit Jahr
    m = re.match(r"^(.+?)\s+-\s+(\d{4})\s+-\s+(.+)$", stem)
    if m:
        author_raw, year, title = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
        author = re.sub(r"\s+et al\.?$", "", author_raw, flags=re.IGNORECASE).strip()
        author = drop_institutional_coauthors(author)
        return {"title": title, "author": author, "year": year, "doi": "", "type": ""}

    # Jahr-Anker mit automatischer Separator-Erkennung (Unterstrich, Bindestrich)
    year_m = _YEAR_BOUNDARY_RE.search(stem)
    if year_m:
        year = int(year_m.group())
        before = stem[: year_m.start()].strip(" -_")
        after = stem[year_m.end() :].strip(" -_")
        if not before and after:
            # Jahr steht am Anfang: "2020-Smith-Title" -> split after on first separator
            sep_m = re.search(r"[-_]", after)
            if sep_m:
                before = after[: sep_m.start()].strip()
                after = after[sep_m.end() :].strip(" -_")
        if before and after and len(before) >= 2 and len(after) >= 4:
            author_raw = re.sub(r"[_]", " ", before).strip()
            title = re.sub(r"[_]", " ", after).strip()
            author = re.sub(r"\s+et al\.?$", "", author_raw, flags=re.IGNORECASE).strip()
            author = drop_institutional_coauthors(author)
            return {"title": title, "author": author.split()[-1] if author else "", "year": year, "doi": "", "type": ""}

    # Fallback: kein Jahr (z.B. "Kuhlthau - INFORMATION SEARCH PROCESS")
    m2 = _ZOTERO_NO_YEAR_RE.match(stem)
    if m2:
        author_raw, title = m2.group(1).strip(), m2.group(2).strip()
        author = re.sub(r"\s+et al\.?$", "", author_raw, flags=re.IGNORECASE).strip()
        author = drop_institutional_coauthors(author)
        return {"title": title, "author": author, "year": None, "doi": "", "type": ""}

    return None


def _zotero_author_matches_embedded(pdf_path: Path, embedded_author: str) -> bool:
    """Prueft ob der eingebettete Info-Dict-Autor durch den Dateinamen POSITIV
    bestaetigt wird (Gate: darf der embedded Author als zitierfaehig dienen?).

    Der Info-Dict-`/Author` ist der Datei-Ersteller, nicht zwingend der Werk-Autor
    (abgetippte/gescannte PDFs). Er wird daher nur akzeptiert, wenn der Dateiname
    ihn bestaetigt. Ein nicht parsbarer Dateiname kann den embedded Author NICHT
    bestaetigen -> False (zuvor faelschlich True/"kein Widerspruch", wodurch der
    Datei-Ersteller als Zitierautor durchrutschte; Codex-Review 2026-06-25)."""
    parsed = _parse_filename_dynamic(pdf_path)
    if not parsed:
        return False  # Kein bestaetigendes Signal -> embedded Author nicht zitierfaehig
    fn_author = parsed["author"].split()[-1].lower()
    emb_tokens = embedded_author.strip().split()
    if not emb_tokens:
        return False
    emb_author = emb_tokens[-1].lower()
    # Nachname-Gleichheit, KEIN Substring: ein kurzer Dateiname-Nachname ("Li") darf
    # nicht zufaellig einen unverwandten embedded Autor ("Williams") bestaetigen
    # (Substring-Falle, Codex-Pass-2 2026-06-25).
    return fn_author == emb_author


# Deutsche Transliterationen VOR dem generischen Diakritik-Fold (#260): NFKD+
# Combining-Strip allein entkernt 'ü' nur zu 'u' (nicht zum gaengigen ASCII-
# Digraph 'ue'), wodurch Dateiname 'Bühner' (-> 'buhner') und embedded, bereits
# ASCII-transliteriertes 'Buehner' (-> 'buehner') sich NICHT traefen. Nur EINE
# Richtung (Umlaut/Eszett -> Digraph): ein echter 'Buehner' (kein Umlaut im
# Dateinamen) bleibt unveraendert und matcht weiterhin gegen embedded 'Bühner'
# — grosszuegig, weil die Folge einer Fehl-Faltung nur Titel-Degradation waere,
# nie eine Fehlattribution.
_GERMAN_TRANSLIT_MAP = str.maketrans({"ü": "ue", "ö": "oe", "ä": "ae", "ß": "ss"})


def _fold_surname_token(token: str) -> str:
    """Kanonische Form eines Nachname-Tokens fuer den Autor-Cross-Check:
    lowercase -> deutsche Transliteration (ue/oe/ae/ss, #260) -> generischer
    Diakritik-Fold. Reihenfolge ist Pflicht, siehe `_GERMAN_TRANSLIT_MAP`."""
    return _strip_diacritics(token.lower().translate(_GERMAN_TRANSLIT_MAP))


def _author_surnames(author: str, *, comma_splits_authors: bool = False) -> set[str]:
    """Menge der gefalteten Nachnamen aus einem Autor-String (Dateiname ODER
    Info-Dict). Trenner: ';', ' und ', ' and ', ' & ' (`_AUTHOR_SEP_RE`) —
    Komma standardmaessig NICHT (das trennt 'Nachname, Vorname').

    Pro Segment ist der Nachname das letzte Token; bei 'Nachname, Vorname'-
    Segmenten der Teil VOR dem Komma (Deci-Fall: 'Deci, Edward L.' -> 'deci',
    nicht 'l'). Gefaltet ueber `_fold_surname_token` (Diakritik + gaengige
    deutsche Transliteration ue/oe/ae/ss, #260), damit 'Yanartaş' == 'Yanartas'
    UND 'Bühner' == 'Buehner' matchen.

    comma_splits_authors (#259): nur fuer Dateiname-Autor-Strings dieser
    Pipeline sicher aktivierbar. Von hier auto-generierte/Zotero-Dateinamen
    tragen NIE Vornamen (Citekey-Konvention: nur Nachname[+Nachname2]) — ein
    Komma im Dateiname-Autor-Segment ist daher praktisch immer ein Autor-
    Trenner zwischen zwei Nachnamen ('Schlebbe, Greifeneder'), nie
    'Nachname, Vorname'. Auf der Embedded-Metadaten-Seite gilt das Gegenteil
    (bibliografische Norm 'Nachname, Vorname') — dort bleibt der Default False,
    sonst wuerde z.B. 'Deci, Edward L.' faelschlich in zwei Personen zerrissen.
    """
    surnames: set[str] = set()
    for segment in _AUTHOR_SEP_RE.split(author):
        seg = segment.strip()
        if not seg:
            continue
        if "," in seg:
            if comma_splits_authors:
                for sub in seg.split(","):
                    sub_tokens = sub.strip().split()
                    if sub_tokens:
                        surnames.add(_fold_surname_token(sub_tokens[-1]))
                continue
            seg = seg.split(",", 1)[0].strip()
        tokens = seg.split()
        if tokens:
            surnames.add(_fold_surname_token(tokens[-1]))
    return surnames


def _filename_contradicts_embedded_author(pdf_path: Path, embedded_author: str) -> bool:
    """Gegenstueck zu `_zotero_author_matches_embedded`: meldet einen POSITIVEN
    Widerspruch zwischen Dateiname-Autor(en) und eingebettetem Info-Dict-Autor.

    Anders als der Match-Guard (der bei nicht-parsbarem Dateiname ODER fehlendem
    Autor False = "nicht positiv bestaetigt" liefert) meldet diese Funktion einen
    Widerspruch NUR, wenn der Dateiname zu Autor(en) parst UND KEIN eingebetteter
    Nachname zu IRGENDEINEM Dateiname-Nachnamen passt. Fehlender Embedded-Autor
    oder nicht-parsbarer Dateiname -> False (kein Widerspruch, keine Quarantaene).

    Mehrautoren-/Diakritika-robust (Review PR #256, an 4 realen Literatur-PDFs
    verifiziert): Beide Seiten werden an ';'/'und'/'and'/'&' in Einzelautoren
    zerlegt und nachname-weise (diakritik- + transliterations-gefaltet, #260)
    verglichen. Ein legitimer Embedded-ERSTautor (= Autor1 einer Mehrautoren-
    Datei) erzeugt so KEINEN Widerspruch mehr; nur ein echt fremder Autor
    (Schlebbe/Afzal: 'Afzal' passt zu weder 'Schlebbe' noch 'Greifeneder') loest
    die /Title-Quarantaene aus. Auf der Dateiname-Seite zaehlt zusaetzlich ein
    blosses Komma OHNE Konjunktion als Autor-Trenner (#259: 'Schlebbe,
    Greifeneder' -> beide Nachnamen statt nur 'Schlebbe') — sicher nur dort,
    weil Dateinamen dieser Pipeline nie Vornamen tragen (siehe
    `_author_surnames`-Docstring).

    Genutzt (#234), um den eingebetteten `/Title` zu verwerfen, wenn der
    Metadatenblock nachweislich aus einer fremden Quelle stammt (z.B. via
    frueherem enrich(rename=True) zurueckgeschriebenes Fremd-Meta) — analog zur
    bereits bestehenden `/Author`-Quarantaene."""
    if not embedded_author or not embedded_author.strip():
        return False
    parsed = _parse_filename_dynamic(pdf_path)
    if not parsed or not parsed.get("author"):
        return False
    fn_surnames = _author_surnames(parsed["author"], comma_splits_authors=True)
    emb_surnames = _author_surnames(embedded_author)
    if not fn_surnames or not emb_surnames:
        return False
    # Widerspruch nur, wenn KEIN Embedded-Nachname zu IRGENDEINEM Dateiname-Nachnamen
    # passt (grosszuegig gegen Mehrautoren-Overreach).
    return emb_surnames.isdisjoint(fn_surnames)


def _meta_complete(meta: dict) -> bool:
    """Gibt True zurueck wenn Metadaten-Dict brauchbar ist: Author und Titel beide nicht leer."""
    return bool(meta.get("author", "").strip()) and bool(meta.get("title", "").strip())


def grobid_lookup(pdf_path: Path, grobid_url: str = "http://localhost:8070", timeout: int = 10) -> dict | None:
    """Grobid-Header-Extraktion via lokalen REST-Server (Apache 2.0).

    Server starten: docker run -t --rm -p 8070:8070 lfoppiano/grobid:latest
    Gibt None zurück wenn Server nicht erreichbar oder Parsing fehlschlägt.
    """
    import xml.etree.ElementTree as ET

    url = f"{grobid_url.rstrip('/')}/api/processHeaderDocument"
    boundary = "AtomicAgentGrobid"
    try:
        with open(pdf_path, "rb") as fh:
            pdf_data = fh.read()
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8")
        body = header + pdf_data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            tei = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None  # Server nicht erreichbar oder Fehler — stilles Fallthrough

    try:
        ns = {"t": "http://www.tei-c.org/ns/1.0"}
        root = ET.fromstring(tei)

        # Titel: analytic > titleStmt Fallback
        t_el = root.find(".//t:analytic/t:title[@type='main']", ns)
        if t_el is None:
            t_el = root.find(".//t:titleStmt/t:title", ns)
        title = (t_el.text or "").strip() if t_el is not None else ""

        # Erstautor Nachname
        a_el = root.find(".//t:analytic/t:author/t:persName/t:surname", ns)
        author = (a_el.text or "").strip() if a_el is not None else ""

        # Jahr aus date[@when] — Format YYYY oder YYYY-MM-DD
        d_el = root.find(".//t:monogr/t:imprint/t:date[@when]", ns)
        year = d_el.get("when", "")[:4] if d_el is not None else ""

        # DOI optional — ermöglicht CrossRef-Followup
        doi_el = root.find(".//t:analytic/t:idno[@type='DOI']", ns)
        doi = (doi_el.text or "").strip() if doi_el is not None else ""

        if not title or not author:
            return None
        meta: dict = {"title": title, "author": author, "year": year or "????"}
        if doi:
            meta["doi"] = doi
        return meta
    except ET.ParseError:
        return None


# --- #329: Kaskadierter Jahres-Fallback wenn der Dateiname kein Jahr traegt --
# Bug-Klasse: Stage 5 (`_parse_filename_dynamic`) faengt auch den jahrlosen
# Zotero-Dateinamen "Autor - Titel" ab (`_ZOTERO_NO_YEAR_RE`) und liefert dann
# `year=None`, ohne dass eine spaetere Stage nochmal ein Jahr sucht -- obwohl es
# oft woertlich im Dokument steht (Selbstzitat-/"vorgeschlagene Zitation"-Zeile)
# oder im PDF-Info-Dict `CreationDate` liegt. Kaskade strikt: Dateiname
# (bestehend, siehe `_resolve_year_fallback`-Aufrufstelle: laeuft nur wenn
# `meta["year"]` noch fehlt) -> Selbstzitat-Zeile -> CreationDate. Beide neuen
# Stufen sind fail-closed plausibilitaetsgeprueft (nie ein Jahr erfinden) und
# markieren ihre Herkunft ueber `meta["year_source"]` (analog zum #263-
# `doi_from_title`-Provenienz-Marker) -- ein aus dem Dateiname/CrossRef/etc.
# stammendes Jahr traegt weiterhin KEINEN `year_source`, nur die beiden neuen
# (schwaecheren) Fallback-Stufen setzen ihn.
_SELF_CITATION_PAGES = 2  # #329: "erste ~2 Seiten" -- Selbstzitat-Zeilen stehen
# teils auf S. 1 (Titelblatt), teils erst im Impressum auf S. 2.
_SELF_CITATION_WINDOW = 200  # Zeichen nach dem Label, in denen das Jahr gesucht wird
_SELF_CITATION_LABEL_RE = re.compile(
    r"(?:vorgeschlagene\s+zitation|zitationsvorschlag|suggested\s+citation|to\s+cite\s+this)",
    re.IGNORECASE,
)


def _plausible_year(year: int) -> bool:
    """Plausibilitaetsfenster fuer Jahres-Fallbacks (#329): 1900 <= Jahr <=
    aktuelles Jahr+1. Fail-closed -- ein Jahr ausserhalb wird verworfen statt
    durchgereicht (nie ein Jahr erfinden)."""
    return 1900 <= year <= datetime.date.today().year + 1


def _extract_self_citation_year(text: str) -> int | None:
    """Sucht eine Selbstzitat-/"vorgeschlagene Zitation"-Zeile im Kopfbereich
    (dt./engl. Muster) und extrahiert das darin genannte Jahr, z.B.
    "vorgeschlagene Zitation: Witt, S. (2020). Informationskompetenz...".

    Nimmt das erste plausible Jahr in einem Fenster NACH dem Label -- reicht
    fuer die gaengigen Formate "(2020)" / ", 2020" / "2020." direkt hinter
    Autor/Titel. Gibt None zurueck wenn kein Label oder kein plausibles Jahr
    im Fenster gefunden wird (fail-closed, siehe `_plausible_year`)."""
    m = _SELF_CITATION_LABEL_RE.search(text)
    if not m:
        return None
    window = text[m.end() : m.end() + _SELF_CITATION_WINDOW]
    year_m = _YEAR_BOUNDARY_RE.search(window)
    if not year_m:
        return None
    year = int(year_m.group())
    return year if _plausible_year(year) else None


def _extract_creation_date_year(pdf_path: Path) -> int | None:
    """Letzter Jahres-Fallback (#329): PDF-Info-Dict `/CreationDate` via pypdf.

    Bewusst NUR als letzte, klar provenienz-markierte Stufe nach Dateiname UND
    Selbstzitat-Zeile -- `CreationDate` ist der Speicher-/Abtipp-Zeitpunkt der
    PDF-Datei, nicht zwingend das Publikationsjahr (vgl. den bereits
    dokumentierten Praezedenzfall in `pdf_chunker._parse_pdfinfo_output`: ein in
    Word abgetipptes Knowles-Kapitel trug `CreationDate` 2019 statt des
    tatsaechlichen Erscheinungsjahrs). Der Aufrufer setzt deshalb immer
    `meta["year_source"] = "creation_date"`, damit diese schwaechere Quelle von
    Dateiname/Selbstzitat/CrossRef unterscheidbar bleibt. Nur ein plausibles
    Jahr (`_plausible_year`) wird akzeptiert -- sonst None."""
    if PdfReader is None:
        return None
    try:
        info = PdfReader(str(pdf_path)).metadata or {}
    except Exception:
        return None
    raw = str(info.get("/CreationDate") or "")
    m = re.search(r"(?:19|20)\d{2}", raw)
    if not m:
        return None
    year = int(m.group())
    return year if _plausible_year(year) else None


def _resolve_year_fallback(pdf_path: Path) -> tuple[int, str] | None:
    """Kaskade Selbstzitat -> CreationDate (#329), wenn weder Dateiname noch
    eine harte ID (DOI/ISBN/arXiv/PMID/CrossRef/...) ein Jahr geliefert hat.
    Gibt `(Jahr, Quelle)` zurueck oder None, wenn keine Stufe ein plausibles
    Jahr findet -- die Downstream-Degradierung auf "[o. J.]" bleibt in dem
    Fall das Sicherheitsnetz."""
    header_text = extract_text(pdf_path, max_pages=_SELF_CITATION_PAGES)
    self_cite_year = _extract_self_citation_year(header_text)
    if self_cite_year:
        return self_cite_year, "self_citation"

    creation_year = _extract_creation_date_year(pdf_path)
    if creation_year:
        return creation_year, "creation_date"

    return None


def enrich(pdf_path: Path, dry_run: bool = False, llm_fallback: bool = False, rename: bool = True) -> dict | None:
    """Haupt-Pipeline: EmbeddedMeta -> DOI -> ISBN -> arXiv -> PMID -> Titel(OpenAlex) -> LLM -> Rename.

    Gibt Metadaten-Dict zurueck oder None wenn nichts gefunden.
    Bei dry_run=True: kein Umbenennen, nur Metadaten zurueckgeben.
    Bei rename=False: Eingabedatei wird nie mutiert (Umbenennen + Metadaten-Write
    übersprungen) — für Aufrufer, die nur die Metadaten brauchen (Pipeline).
    """
    print(f"[pdf-enrich] {pdf_path.name}")

    # #282: roh-extrahierter, format-valider DOI aus dem Kopfbereich (Stage 1) merken
    # -- auch wenn CrossRef ihn nicht kennt. Betrifft strukturell DataCite-registrierte
    # DOIs deutscher Repositorien (peDOCS, edoc.hu-berlin.de u.ae.), die CrossRef gar
    # nicht fuehrt. Der DOI stand woertlich im Dokument -- das ist bereits eine harte
    # Quelle, keine Titel-Heuristik (#263). Registry-Bestaetigung bleibt eine
    # Anreicherung (Autor/Titel/Jahr, Peer-Review-Signal), ist aber kein Beleg-Kriterium
    # fuer den DOI-Wert selbst.
    _raw_doi: str | None = None

    # Stage 0: Eingebettete PDF-Metadaten (mit Dateiname-Cross-Check)
    meta = read_pdf_metadata(pdf_path)
    if meta:
        if not _zotero_author_matches_embedded(pdf_path, meta["author"]):
            print(f"  -> Eingebettete Metadaten verworfen: '{meta['author']}' passt nicht zum Dateinamen")
            meta = None
        else:
            print(f"  -> Eingebettete Metadaten: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 0.5: DOI oder arXiv-ID im Dateinamen
    if not meta:
        fn_doi = _extract_doi_from_filename(pdf_path.stem)
        if fn_doi:
            print(f"  -> DOI im Dateinamen: {fn_doi}")
            meta = crossref_lookup(fn_doi)
            if meta:
                print(f"  -> CrossRef: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")
        if not meta:
            fn_arxiv = _extract_arxiv_from_filename(pdf_path.stem)
            if fn_arxiv:
                print(f"  -> arXiv-ID im Dateinamen: {fn_arxiv}")
                meta = arxiv_lookup(fn_arxiv)
                if meta:
                    print(f"  -> arXiv: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Text-Extraktion (fuer Stages 1-6 benoetigt)
    text = None
    header_text = ""  # Kopfbereich-Text fuer die ID-Stages 1-4 (#41-R2)
    if not meta:
        source_path = pdf_path
        text = extract_text(pdf_path, max_pages=10)
        if is_scanned(text):
            print("  -> Gescannte PDF erkannt")
            if ocr_available():
                print("  -> OCR via ocrmypdf...")
                ocr_path = run_ocr(pdf_path)
                if ocr_path:
                    text = extract_text(ocr_path, max_pages=10)
                    source_path = ocr_path
                    if not dry_run:
                        pdf_path = ocr_path
            if is_scanned(text):
                print("  -> Kein Text extrahierbar -- uebersprungen", file=sys.stderr)
                return None
        # IDs nur aus dem Kopfbereich (gleiche Quelle wie text, OCR-konsistent)
        header_text = extract_text(source_path, max_pages=_FRONT_MATTER_PAGES)

    # Stage 1: DOI -> CrossRef
    if not meta:
        doi = extract_doi(header_text)
        if doi:
            _raw_doi = doi
            print(f"  -> DOI gefunden: {doi}")
            cr = crossref_lookup(doi)
            if cr and _meta_complete(cr):
                # Book-chapter DOI: ISBN-Lookup bevorzugen wenn ISBN im Kopfbereich
                if cr.get("type") in _BOOK_PART_TYPES and extract_isbn(header_text):
                    print("  -> CrossRef Buchkapitel — versuche ISBN-Lookup")
                else:
                    meta = cr
                    print(f"  -> CrossRef: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")
            elif cr:
                print("  -> CrossRef unvollstaendig (kein Autor/Titel) — weiter")

    # Stage 2: ISBN -> Open Library -> Google Books
    if not meta:
        isbn = extract_isbn(header_text)
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
        arxiv_id = extract_arxiv_id(header_text)
        if arxiv_id:
            print(f"  -> arXiv-ID gefunden: {arxiv_id}")
            meta = arxiv_lookup(arxiv_id)
            if meta:
                print(f"  -> arXiv: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 4: PMID -> PubMed
    if not meta:
        pmid = extract_pmid(header_text)
        if pmid:
            print(f"  -> PMID gefunden: {pmid}")
            meta = pubmed_lookup(pmid)
            if meta:
                print(f"  -> PubMed: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 5: Dateiname dynamisch parsen (Zotero, Unterstrich, Jahr-Anker, kein Jahr)
    if not meta:
        meta = _parse_filename_dynamic(pdf_path)
        if meta:
            print(f"  -> Zotero-Dateiname: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 6: Titel-Suche -> OpenAlex
    # Kandidaten-Reihenfolge: (1) Dateiname wenn Title-artig, (2) erste nicht-header Textzeile
    if not meta:
        fn_stem = re.sub(r"[_\-]", " ", pdf_path.stem).strip()
        _CODE_PREFIX = re.compile(r"^[A-Z0-9]{3,}\d", re.ASCII)
        fn_title_candidate = fn_stem if len(fn_stem) > 10 and not _CODE_PREFIX.match(fn_stem) else ""
        _AUTHOR_LINE_RE = re.compile(r"^[A-Z][a-zÀ-ÿ]+(\s+[A-Z][a-zÀ-ÿ]+)?\s+and\s+[A-Z]", re.UNICODE)

        def _title_block(text: str, max_chars: int = 150) -> str:
            """Sammelt aufeinanderfolgende nicht-header Zeilen zu einem Titelblock."""
            block: list[str] = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    if block:
                        break  # Leerzeile beendet Block
                    continue
                if _HEADER_RE.match(stripped) or _AUTHOR_LINE_RE.match(stripped) or len(stripped) < 15:
                    if block:
                        break  # Header nach erstem Inhalt beendet Block
                    continue
                block.append(stripped)
                if sum(len(x) for x in block) >= max_chars:
                    break
            return " ".join(block).strip()

        text_candidate = _title_block(text or "")
        title_guess = fn_title_candidate or text_candidate
        if title_guess:
            src = "Dateiname" if title_guess == fn_title_candidate else "Textzeile"
            print(f"  -> Kein ID, suche Titel ({src}): '{title_guess[:60]}'")
            meta = openalex_title_search(title_guess)
            if meta and not _title_match_confident(title_guess, meta.get("title", "")):
                print(f"  -> OpenAlex-Treffer verworfen (schwacher Titel-Match): '{meta.get('title', '')[:60]}'")
                meta = None
            elif meta:
                # #263: Provenienz-Marker. Dieser DOI stammt aus reiner Titel-
                # Heuristik (OpenAlex-Suche + _title_match_confident-Gate), NICHT
                # aus einer harten ID (DOI/arXiv/PMID im Text) oder CrossRef-
                # Bestätigung. Downstream (Edition-Gate) darf ihn deshalb nicht wie
                # eine hart verifizierte ID behandeln — er fällt auf die eigene
                # (soft flaggende) Title-Match-Suche des Quality-Agents zurück.
                if meta.get("doi"):
                    meta["doi_from_title"] = True
                print(f"  -> OpenAlex: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 6.5: Grobid (lokaler Server, optional)
    if not meta:
        _grobid_url = __import__("os").getenv("GROBID_URL", "http://localhost:8070")
        meta = grobid_lookup(pdf_path, grobid_url=_grobid_url)
        if meta:
            print(f"  -> Grobid: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")
            # Grobid-DOI → CrossRef für vollständigere Metadaten
            if meta.get("doi"):
                _cr = crossref_lookup(meta["doi"])
                if _cr and _meta_complete(_cr):
                    meta = _cr
                    print(f"  -> Grobid+CrossRef: {meta['author']} ({meta['year']}) -- {meta['title'][:60]}")

    # Stage 7: LLM-Fallback
    if not meta and llm_fallback:
        print("  -> LLM-Fallback (Haiku)...")
        meta = _llm_extract(text[:3000])

    if not meta:
        print("  -> Keine Metadaten gefunden -- uebersprungen", file=sys.stderr)
        return None

    # #282: keine spaetere Stage hat einen DOI geliefert (weder Registry-bestaetigt
    # noch titel-geraten) -- der roh-extrahierte Kopfbereich-DOI darf trotzdem
    # durchgereicht werden. Kein `doi_from_title`-Marker: das ist keine
    # Titel-Heuristik. Ueberschreibt NIE einen bereits gesetzten (auch titel-
    # geratenen) DOI -- die #263-Provenienz-Markierung bleibt unangetastet.
    if _raw_doi and not meta.get("doi"):
        meta["doi"] = _raw_doi
        print(f"  -> DOI ohne Registry-Bestaetigung uebernommen (Format-valide): {_raw_doi}")

    # #329: keine bisherige Stage hat ein Jahr geliefert (typischer Fall: Stage 5
    # traf nur das jahrlose Zotero-Muster "Autor - Titel"). Dateiname-Jahr hat
    # weiterhin Vorrang -- dieser Block laeuft nur wenn `year` noch fehlt.
    if not meta.get("year"):
        _year_fb = _resolve_year_fallback(pdf_path)
        if _year_fb:
            _year, _year_source = _year_fb
            meta["year"] = _year
            meta["year_source"] = _year_source
            print(f"  -> Jahr-Fallback ({_year_source}): {_year}")

    if not rename:
        return meta

    new_path = rename_pdf(pdf_path, meta, dry_run=dry_run)
    if dry_run:
        print(f"  -> [dry-run] wuerde umbenennen zu: {new_path.name}")
    else:
        print(f"  -> Umbenannt: {new_path.name}")
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
        r = _sp.run(["claude", "-p", "--model", "haiku"], input=prompt, capture_output=True, text=True, timeout=30)
        lines = {
            line.split(":")[0].strip(): line.split(":", 1)[1].strip() for line in r.stdout.splitlines() if ":" in line
        }
        return (
            {
                "title": lines.get("title", ""),
                "author": lines.get("author", ""),
                "year": int(lines["year"]) if lines.get("year", "").isdigit() else None,
                "doi": "",
                "type": "",
            }
            if lines.get("title")
            else None
        )
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
        writer.add_metadata(
            {
                "/Title": meta.get("title", ""),
                "/Author": meta.get("author", ""),
                "/Subject": str(meta.get("year", "")),
                "/Keywords": meta.get("doi", ""),
            }
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=pdf_path.parent) as tmp:
            writer.write(tmp)
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(pdf_path))
    except Exception as e:
        print(f"  [warn] Metadaten-Schreiben fehlgeschlagen: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Metadata Enrichment — DOI → CrossRef → Metadaten (Rename opt-in)")
    parser.add_argument("pdf", nargs="+", help="PDF-Datei(en) oder Glob-Pattern")
    parser.add_argument("--dry-run", action="store_true", help="Nicht umbenennen, nur ausgeben")
    parser.add_argument("--llm-fallback", action="store_true", help="Haiku nutzen wenn CrossRef nichts findet")
    parser.add_argument(
        "--rename",
        action="store_true",
        help="PDF umbenennen + Metadaten in die Datei schreiben "
        "(Default: aus — schützt git-getrackte PDFs vor Mutation)",
    )
    args = parser.parse_args()

    import glob

    paths = [Path(p) for pattern in args.pdf for p in glob.glob(pattern)]
    if not paths:
        print("Keine PDFs gefunden.", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        enrich(path, dry_run=args.dry_run, llm_fallback=args.llm_fallback, rename=args.rename)


if __name__ == "__main__":
    main()
