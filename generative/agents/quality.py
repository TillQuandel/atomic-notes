"""Quality-Agent: CrossRef + OpenAlex + Retraction-Check → QualityReport."""

from __future__ import annotations
import json
import urllib.parse
import urllib.request
from typing import Optional

from generative.config import USER_AGENT
from generative.schemas.atomic_note import QualityReport


def _http_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _crossref_meta(doi: str) -> Optional[dict]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    data = _http_json(url)
    if not data:
        return None
    return data.get("message")


def _openalex_work(doi: str) -> Optional[dict]:
    url = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi, safe='/')}"
    return _http_json(url)


def _crossref_doi_lookup(title: str, author: Optional[str] = None, year: Optional[str] = None) -> Optional[str]:
    """Sucht DOI per CrossRef-Title-Match. Gibt nur DOI zurück wenn Match überzeugend."""
    if not title or len(title) < 10:
        return None
    params = {"query.bibliographic": title, "rows": "3"}
    if author:
        params["query.author"] = author
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    data = _http_json(url)
    if not data:
        return None
    items = (data.get("message") or {}).get("items", [])
    if not items:
        return None
    top = items[0]
    # Conservative: nur akzeptieren wenn Score hoch und Year matcht (falls gegeben)
    score = top.get("score", 0)
    if score < 80:
        return None
    if year:
        published = top.get("published-print") or top.get("published-online") or top.get("issued") or {}
        date_parts = (published.get("date-parts") or [[None]])[0]
        if date_parts and date_parts[0] and str(date_parts[0]) != str(year):
            return None
    return top.get("DOI")


def check_quality(
    doi: Optional[str] = None, author: Optional[str] = None, year: Optional[str] = None, title: Optional[str] = None
) -> QualityReport:
    flags: list[str] = []
    peer_reviewed: Optional[bool] = None
    citation_count: Optional[int] = None
    retracted = False
    crossref_title: Optional[str] = None
    crossref_author: Optional[str] = None
    crossref_year: Optional[str] = None
    doi_from_title_match = False

    # DOI-Fallback: per Title+Author Suche wenn nicht explizit übergeben
    if not doi and title:
        doi = _crossref_doi_lookup(title, author=author, year=year)
        if doi:
            doi_from_title_match = True
            flags.append(f"ℹ️ DOI per Title-Match gefunden: {doi}")

    if doi:
        meta = _crossref_meta(doi)
        if meta:
            # Peer-Review-Signal: Journal-Artikel in CrossRef = i.d.R. peer-reviewed
            ctype = meta.get("type", "")
            peer_reviewed = ctype in ("journal-article", "proceedings-article", "dissertation")
            citation_count = meta.get("is-referenced-by-count")
            # CrossRef-Subtype "retracted-article" oder update-to-Einträge mit
            # abgestufter Klassifikation (Retraction ≠ Erratum ≠ Expression of Concern).
            # Quality-Differenzierung: nur retraction/withdrawal → retracted=True,
            # andere Notice-Typen werden als separate (weiche) Quality-Flags geführt.
            if ctype == "retracted-article":
                retracted = True
            update_types = {(u.get("type") or "").lower() for u in (meta.get("update-to") or []) if isinstance(u, dict)}
            if "retraction" in update_types or "withdrawal" in update_types:
                retracted = True
            if "expression-of-concern" in update_types:
                flags.append("⚠️ Expression of Concern — Aussagen kritisch prüfen")
            if update_types & {"correction", "erratum"}:
                flags.append("ℹ️ Erratum/Korrektur veröffentlicht — Datenstand vor Korrektur möglich")
            # F2: CrossRef-Metadata für Quellen-Block-Override extrahieren
            titles = meta.get("title") or []
            if titles:
                crossref_title = titles[0]
            authors = meta.get("author") or []
            if authors:
                names = []
                for a in authors[:3]:
                    fam = a.get("family", "").strip()
                    giv = a.get("given", "").strip()
                    if fam and giv:
                        names.append(f"{fam}, {giv[0]}.")
                    elif fam:
                        names.append(fam)
                if names:
                    crossref_author = " & ".join(names) if len(names) <= 2 else names[0] + " et al."
            published = meta.get("published-print") or meta.get("published-online") or meta.get("issued") or {}
            date_parts = (published.get("date-parts") or [[None]])[0]
            if date_parts and date_parts[0]:
                crossref_year = str(date_parts[0])

        # OpenAlex als zweite Zitations-Quelle + autoritative Retraction-Info
        oa = _openalex_work(doi)
        if oa:
            if citation_count is None:
                citation_count = oa.get("cited_by_count")
            # OpenAlex `is_retracted` ist die zuverlässigste Quelle
            if oa.get("is_retracted"):
                retracted = True

    # Flags setzen
    if retracted:
        flags.append("⚠️ ZURÜCKGEZOGEN (CrossRef/OpenAlex)")
    if citation_count is not None and citation_count < 5:
        flags.append(f"⚠️ niedrige Zitationsanzahl (n={citation_count})")
    if peer_reviewed is False:
        flags.append("⚠️ nicht peer-reviewed")
    if doi is None:
        flags.append("⚠️ kein DOI — Qualität nicht automatisch prüfbar")

    return QualityReport(
        peer_reviewed=peer_reviewed,
        citation_count=citation_count,
        retracted=retracted,
        flags=flags,
        crossref_title=crossref_title,
        crossref_author=crossref_author,
        crossref_year=crossref_year,
        doi_from_title_match=doi_from_title_match,
    )
