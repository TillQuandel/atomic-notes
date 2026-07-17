"""#331: `confirmation_rate` im Verifier-Trace konnte > 1.0 (100 %) werden --
Coverage-Serie 2: Lauf 3 (Poege) 2.0 nach einem Refine-Zyklus, Lauf 5 (Kok) 3.0
OHNE Refine. Der Kok-Beleg widerlegt die urspruengliche Refine-Hypothese: die
Ursache liegt in EINEM einzelnen `verifier.run()`-Aufruf.

Root Cause: `_log_anchor_stats(title, total_in, final_anchors)` teilte
`confirmed` (gezaehlt aus `final_anchors`, NACH dem Lauf) durch `total_in`
(gezaehlt VOR dem Lauf, in `run()`). `_run_inner()` ruft an JEDEM Ausgang
`sync_anchors_from_body()` auf, das aus `„..." (S. N)`-Zitaten im Note-Body
NEUE, bereits bestaetigte Anker an `draft.source_anchors` anhaengt -- Zaehler
(post-sync) und Nenner (pre-sync) beziehen sich dadurch auf unterschiedliche
Mengen, ohne dass ein Refine-Zyklus noetig ist.

Fix: Nenner = `len(final_anchors)` (dieselbe Menge wie der Zaehler) statt der
pre-run `total_in`-Momentaufnahme. Rate ist damit mathematisch garantiert <= 1.0.

RED vor dem Fix: confirmation_rate == 3.0 (1 pre-pass-bestaetigter Original-
Anker + 2 body-synced Anker, geteilt durch total_in=1).
"""

from __future__ import annotations

import json

import generative.agents.tracing as tracing
import generative.agents.verifier as verifier
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor


def test_confirmation_rate_never_exceeds_one_when_body_sync_adds_anchors(tmp_path, monkeypatch):
    backend = tracing.JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)

    draft = AtomicNoteDraft(
        title="Test Note",
        body=(
            "Diese Formulierung erscheint identisch im Originaltext. "
            "„Erstes zusaetzliches Body-Zitat aus dem Volltext” (S. 3) "
            "Ein Uebergangssatz dazwischen. "
            "„Zweites zusaetzliches Body-Zitat, ebenfalls belegt” (S. 5)"
        ),
        source_anchors=[
            TextAnchor(
                quote="Diese Formulierung erscheint identisch im Originaltext",
                page=None,
                fuzzy_page=None,
            )
        ],
        related=[],
        tags=[],
        synthesis_confidence="medium",
    )
    chunk_text = (
        "[S. 3] Diese Formulierung erscheint identisch im Originaltext. Mehr Kontext fuer die Marker-Erkennung."
    )

    verifier.run(draft, chunk_text=chunk_text)

    trace_file = tmp_path / "test-run.jsonl"
    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    ev = next(e for e in events if e.get("type") == "anchor_stats")

    # Body-Sync hat tatsaechlich neue Anker angehaengt -- sonst testet dieser
    # Fall den Bug gar nicht (Kontrolle gegen ein stillschweigend geaendertes Fixture).
    assert len(draft.source_anchors) > ev["total_in"], "Fixture-Bug: Body-Sync hat nichts ergaenzt"

    assert ev["confirmation_rate"] <= 1.0, f"confirmation_rate > 100%: {ev}"
    assert ev["confirmation_rate"] == round(ev["confirmed"] / len(draft.source_anchors), 3)
