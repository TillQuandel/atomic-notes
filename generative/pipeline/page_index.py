"""Page-Index + Claim-Quellfenster (Faithfulness-Gate E1, #69).

Zwei pure, deterministische Helfer ohne Pipeline-Verdrahtung. Das
Faithfulness-Gate nutzt sie später, um pro Claim das echte Quellfenster
(verankerte Druckseite + Nachbarseiten) zu liefern statt des gekürzten
`concept_text_window`.

Keine I/O, keine Abhängigkeit von orchestrator/agents.
"""

from __future__ import annotations

from generative.anchor_patterns import PAGE_MARKER_LINE_RE


def build_page_index(full_text: str) -> dict[int, str]:
    """Splittet PDF-Volltext mit `[S. N]`-Markern in `{seitennummer: seitentext}`.

    Nutzt die zeilen-isolierte `PAGE_MARKER_LINE_RE` — Inline-Verweise wie
    „vgl. [S. 12]" im Fließtext zählen nicht als Seitengrenze. Text vor dem
    ersten Marker wird verworfen. Kein Marker im Text → leeres Dict. Doppelte
    Seitennummer (defensiv, sollte nicht vorkommen) wird mit `"\n"` konkateniert.
    """
    matches = list(PAGE_MARKER_LINE_RE.finditer(full_text))
    index: dict[int, str] = {}
    for i, match in enumerate(matches):
        page = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        index[page] = f"{index[page]}\n{text}" if page in index else text
    return index


def claim_source_window(page_index: dict[int, str], page: int, neighbors: int = 1) -> str | None:
    """Liefert das Quellfenster für einen auf `page` verankerten Claim.

    Enthält `page` selbst plus `neighbors` Nachbarseiten davor/danach — als
    Nachbarn in der sortierten Key-Liste des Index, NICHT arithmetisch
    `page ± 1` (Seitennummern können Lücken haben, z.B. bei Kapitel-Auszügen).
    `page` nicht im Index → `None` (abstain-Signal, kein Raten).
    """
    if page not in page_index:
        return None
    keys = sorted(page_index)
    pos = keys.index(page)
    window_keys = keys[max(0, pos - neighbors) : pos + neighbors + 1]
    return "\n\n".join(f"[S. {n}]\n{page_index[n]}" for n in window_keys)
