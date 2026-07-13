"""Cache-Auswertung eines Run-Traces — Messwerkzeug für #147 (Prefix-Cache).

Aggregiert pro Agent die echten LLM-Calls (Response-Cache-Hits separat),
Input-/Output-Tokens und cache_read vs. cache_creation. Der Creation-Anteil
an der Cache-Seite ist die Kennzahl des #147-Umbaus: gesund ist Creation nur
in der ersten Fan-out-Welle, danach Reads.

Aufruf:
    uv run python -m generative.tools.cache_report [trace.jsonl]

Ohne Argument wird der neueste Trace unter generative/.cache/runs/ genommen.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from generative.config import CACHE_DIR


def aggregate(trace_path: Path) -> dict[str, dict]:
    agg: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "resp_cache_hits": 0, "in": 0, "out": 0, "read": 0, "create": 0}
    )
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "model" not in r:
                continue
            a = agg[r.get("agent", "?")]
            if r.get("cached"):
                a["resp_cache_hits"] += 1
                continue
            a["calls"] += 1
            a["in"] += r.get("input_tokens", 0) or 0
            a["out"] += r.get("output_tokens", 0) or 0
            a["read"] += r.get("cache_read_tokens", 0) or 0
            a["create"] += r.get("cache_creation_tokens", 0) or 0
    return dict(agg)


def render(agg: dict[str, dict]) -> str:
    lines = [
        f"{'agent':<16}{'calls':>6}{'hits':>6}{'input':>10}{'output':>9}{'cache_read':>12}{'cache_create':>13}{'create%':>9}"
    ]
    tot = {"calls": 0, "in": 0, "out": 0, "read": 0, "create": 0}
    for agent, a in sorted(agg.items()):
        cache_side = a["read"] + a["create"]
        pct = (100 * a["create"] / cache_side) if cache_side else 0.0
        lines.append(
            f"{agent:<16}{a['calls']:>6}{a['resp_cache_hits']:>6}{a['in']:>10}{a['out']:>9}"
            f"{a['read']:>12}{a['create']:>13}{pct:>8.1f}%"
        )
        for k in tot:
            tot[k] += a[k]
    cache_side = tot["read"] + tot["create"]
    pct = (100 * tot["create"] / cache_side) if cache_side else 0.0
    lines.append("-" * 81)
    lines.append(
        f"{'TOTAL':<16}{tot['calls']:>6}{'':>6}{tot['in']:>10}{tot['out']:>9}"
        f"{tot['read']:>12}{tot['create']:>13}{pct:>8.1f}%"
    )
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1:
        trace = Path(sys.argv[1])
    else:
        runs = sorted((CACHE_DIR / "runs").glob("*.jsonl"))
        if not runs:
            raise SystemExit("Kein Trace unter generative/.cache/runs/ gefunden.")
        trace = runs[-1]
    print(f"Trace: {trace}")
    print(render(aggregate(trace)))


if __name__ == "__main__":
    main()
