"""Kapitel-DOI über den Seitenbereich auflösen (Layer 3 der Edition-Verifikation).

Wurzel-Fix gegen Edition-Verwechslung: Ein Buchkapitel-Titel ist oft generisch
(„Wissensorganisation") und CrossRef liefert mehrere Auflagen als Treffer — der
naive Top-Treffer ist nicht zwingend die richtige Edition (KSS-6 2013 vs. KSS-7
2022 desselben Reimer-Kapitels). Der **Seitenbereich** beweist die Edition: das
2013er Kapitel beginnt auf Druckseite 172, das 2022er auf 147.

Der Extraktions-Schritt ruft `resolve_chapter_doi(...)` mit Titel + Autor +
**bekannter Startseite** + Jahr auf und pinnt die zurückgegebene DOI via
`--doi` an die Pipeline. Dann ist die Zitation CrossRef-belegt statt
dateiname-geraten, und eine falsche Auflage fällt sofort auf (kein Match → None).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from typing import Optional

_CROSSREF = "https://api.crossref.org/works"


def _start_page(item: dict) -> Optional[int]:
    """Erste Zahl aus dem CrossRef-`page`-Feld (z.B. '172-182' → 172)."""
    page = item.get("page")
    if not page:
        return None
    m = re.match(r"\s*(\d+)", str(page))
    return int(m.group(1)) if m else None


def _year(item: dict) -> Optional[int]:
    pub = item.get("issued") or item.get("published-print") or item.get("published-online") or {}
    parts = (pub.get("date-parts") or [[None]])[0]
    return parts[0] if parts and parts[0] else None


def pick_chapter_doi(items: list[dict], start_page: int, year: Optional[str] = None) -> Optional[dict]:
    """Wählt aus CrossRef-Treffern das Kapitel, dessen Seitenbereich auf
    ``start_page`` beginnt — und, falls ``year`` gegeben, dessen Jahr passt.

    Fail-closed: kein passender Seitenbereich → ``None`` (lieber kein DOI als die
    falsche Auflage). Der Seitenbereich ist der eigentliche Edition-Beweis; der
    Titel-Score von CrossRef wird bewusst NICHT als Tie-Breaker genutzt, weil er
    bei generischen Kapiteltiteln die falsche Auflage nach oben sortiert.
    """
    want_year = int(year) if year and str(year).isdigit() else None
    matches = []
    for item in items:
        if not item.get("DOI"):
            continue  # ohne DOI nicht pinbar — überspringen statt KeyError
        if _start_page(item) != start_page:
            continue
        if want_year is not None and _year(item) != want_year:
            continue
        matches.append(item)
    # Genau EIN Treffer — sonst fail-closed None (0 = keiner; >1 = mehrdeutig, z.B.
    # CrossRef-Dublette oder zwei Werke mit gleicher Startseite). (Codex-Review.)
    return matches[0] if len(matches) == 1 else None


def resolve_chapter_doi(
    title: str, author: Optional[str], start_page: int, year: Optional[str] = None, rows: int = 10
) -> Optional[dict]:
    """Fragt CrossRef nach ``title``(+``author``) und disambiguiert per Startseite.

    Gibt das matchende CrossRef-Item (inkl. ``DOI``) zurück oder ``None``. I/O —
    die reine Auswahllogik steckt in :func:`pick_chapter_doi` (testbar ohne Netz).
    """
    params = {"query.bibliographic": title, "rows": str(rows)}
    if author:
        params["query.author"] = author
    url = f"{_CROSSREF}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "atomic-notes (mailto:noreply@example.com)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    items = (data.get("message") or {}).get("items", [])
    return pick_chapter_doi(items, start_page, year)


def _main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Kapitel-DOI über den Seitenbereich auflösen (Edition-Beweis).")
    p.add_argument("--title", required=True, help="Kapitel- oder Werk-Titel")
    p.add_argument("--author", default=None, help="Autor (Nachname genügt)")
    p.add_argument(
        "--start-page", type=int, required=True, help="Bekannte Druck-Startseite des Kapitels (beweist die Auflage)"
    )
    p.add_argument("--year", default=None, help="Erwartetes Jahr (fail-closed-Filter)")
    a = p.parse_args(argv)
    hit = resolve_chapter_doi(a.title, a.author, a.start_page, a.year)
    if not hit:
        print("kein eindeutiger Treffer (Seitenbereich passt zu keiner Auflage) — kein DOI gepinnt", file=sys.stderr)
        return 1
    print(hit["DOI"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
