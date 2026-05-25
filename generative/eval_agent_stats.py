"""Per-Agent Statistik-Aggregation aus Run-Trace-JSONL.

Usage:
  python eval_agent_stats.py                          # neuester Run
  python eval_agent_stats.py .cache/runs/20260518-123456.jsonl
  python eval_agent_stats.py .cache/runs/*.jsonl      # Multi-Run-Vergleich
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path


def aggregate(trace_path: Path) -> dict[str, dict]:
    """Liest JSONL und gibt per-Agent-Stats-Dict zurück.

    LLM-Call-Entries haben kein 'type'-Feld (base._trace()-Format).
    Strukturierte Events haben type: run_start | anchor_stats | score_result | ...
    """
    entries = [
        json.loads(l)
        for l in trace_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    llm_calls: dict[str, list] = defaultdict(list)
    anchor_rates: dict[str, list] = defaultdict(list)
    scores: dict[str, list] = defaultdict(list)
    outcomes: list[dict] = []
    plan: dict = {}

    for e in entries:
        agent = e.get("agent", "")
        etype = e.get("type")

        if etype is None and agent:      # LLM-Call (kein type-Feld)
            llm_calls[agent].append(e)
        elif etype == "anchor_stats":
            anchor_rates[agent].append(e.get("confirmation_rate", 0.0))
        elif etype == "score_result":
            scores[agent].append(e.get("score", 0))
        elif etype == "note_outcome":
            outcomes.append(e)
        elif etype == "plan_stats":
            plan = e

    stats: dict[str, dict] = {}

    for agent, calls in llm_calls.items():
        errors = [c for c in calls if c.get("error")]
        not_cached = [c for c in calls if not c.get("cached", False)]
        stats.setdefault(agent, {}).update({
            "calls": len(calls),
            "cached_calls": len(calls) - len(not_cached),
            "error_count": len(errors),
            "error_rate": round(len(errors) / len(calls), 3) if calls else 0.0,
            "total_input_tokens": sum(c.get("input_tokens", 0) for c in not_cached),
            "total_output_tokens": sum(c.get("output_tokens", 0) for c in not_cached),
            "avg_duration_ms": (
                int(sum(c.get("duration_ms", 0) for c in not_cached) / len(not_cached))
                if not_cached else 0
            ),
        })

    for agent, rates in anchor_rates.items():
        stats.setdefault(agent, {})["avg_confirmation_rate"] = (
            round(sum(rates) / len(rates), 3) if rates else 0.0
        )

    for agent, sc in scores.items():
        stats.setdefault(agent, {})["avg_score"] = (
            round(sum(sc) / len(sc), 2) if sc else 0.0
        )

    if outcomes:
        vault_n = sum(1 for o in outcomes if o.get("destination") == "vault")
        stats.setdefault("orchestrator", {}).update({
            "notes_total": len(outcomes),
            "notes_vault": vault_n,
            "vault_rate": round(vault_n / len(outcomes), 3),
        })

    if plan:
        stats.setdefault("orchestrator", {}).update({
            "vault_rate": plan.get("vault_rate", 0.0),
            "written": plan.get("written", 0),
        })

    return stats


def _find_latest_trace() -> Path | None:
    runs_dir = Path(__file__).parent / ".cache" / "runs"
    if not runs_dir.exists():
        return None
    files = sorted(runs_dir.glob("*.jsonl"))
    return files[-1] if files else None


def _print_table(stats: dict, model_config: dict | None, run_id: str) -> None:
    print(f"\n=== Agent-Stats: Run {run_id} ===")
    if model_config:
        print("Model-Config:", " | ".join(f"{k}={v}" for k, v in model_config.items()))
    print()
    print(f"{'Agent':<18} {'Calls':>6} {'Err%':>6} {'In-Tok':>9} {'Out-Tok':>8} {'avg ms':>7} {'avg Score':>9} {'Anker%':>8} {'Vault%':>7}")
    print("-" * 82)
    for agent, s in sorted(stats.items()):
        err_pct = f"{s['error_rate']:.0%}" if "error_rate" in s else ""
        score   = f"{s['avg_score']:.1f}"  if "avg_score" in s else ""
        anker   = f"{s['avg_confirmation_rate']:.0%}" if "avg_confirmation_rate" in s else ""
        vault   = f"{s['vault_rate']:.0%}" if "vault_rate" in s else ""
        print(
            f"{agent:<18}"
            f" {s.get('calls', ''):>6}"
            f" {err_pct:>6}"
            f" {s.get('total_input_tokens', ''):>9}"
            f" {s.get('total_output_tokens', ''):>8}"
            f" {s.get('avg_duration_ms', ''):>7}"
            f" {score:>8}"
            f" {anker:>8}"
            f" {vault:>7}"
        )


def main() -> None:
    import glob as _glob
    if len(sys.argv) > 1:
        paths = [Path(p) for pattern in sys.argv[1:] for p in _glob.glob(pattern)]
    else:
        latest = _find_latest_trace()
        paths = [latest] if latest else []
    if not paths:
        print("Kein Run-Trace gefunden unter .cache/runs/")
        sys.exit(1)

    if len(paths) == 1:
        trace_path = paths[0]
        entries = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        run_start = next((e for e in entries if e.get("type") == "run_start"), {})
        stats = aggregate(trace_path)
        _print_table(stats, run_start.get("model_config"), run_start.get("run_id", trace_path.stem))
    else:
        print(f"\n=== Multi-Run-Vergleich ({len(paths)} Runs) ===")
        print(f"{'Run':<24} {'Agent':<14} {'Calls':>5} {'In-Tok':>8} {'Out-Tok':>8} {'avg ms':>7} {'Vault%':>7}")
        print("-" * 75)
        for p in sorted(paths):
            entries = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            run_start = next((e for e in entries if e.get("type") == "run_start"), {})
            run_id = run_start.get("run_id", p.stem)
            stats = aggregate(p)
            for agent, s in sorted(stats.items()):
                vault = f"{s['vault_rate']:.0%}" if "vault_rate" in s else ""
                print(
                    f"{run_id[:24]:<24}"
                    f" {agent[:14]:<14}"
                    f" {s.get('calls', ''):>5}"
                    f" {s.get('total_input_tokens', ''):>8}"
                    f" {s.get('total_output_tokens', ''):>8}"
                    f" {s.get('avg_duration_ms', ''):>7}"
                    f" {vault:>7}"
                )


if __name__ == "__main__":
    main()
