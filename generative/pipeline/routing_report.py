"""Trust-/Routing-Sichtbarkeit (#45).

Macht das gebaute Trust-Rückgrat (fail-closed, Critic-Routing, keine PDF-Mutation)
im Lauf sichtbar — ohne Erklär-Dashboard und ohne Confidence-Display (passives
Anzeigen kalibriert Vertrauen nicht; wirksam ist NL-Rahmung + sichtbarer Status,
siehe [[Aktives-Engagement-schlägt-passives-Vertrauens-Display]]).

Pure Helper: keine Seiteneffekte, kein I/O. Der Orchestrator/vault_writer druckt
die Rückgaben. Friction-Gating (nur Low-Confidence/Inbox-Pfad) liegt beim Caller —
diese Helper beschreiben nur, was angezeigt wird.
"""
from __future__ import annotations

from generative.schemas.atomic_note import AtomicNoteDraft


def _decide(note: AtomicNoteDraft) -> tuple[bool, str]:
    # Lazy import: vault_writer importiert dieses Modul → Zirkel vermeiden.
    from generative.pipeline.vault_writer import auto_write_decision
    return auto_write_decision(note)


def routing_status_line(note: AtomicNoteDraft) -> str:
    """Ein-Zeilen-Status pro Note für den echten Lauf — Grund + konkrete Flags.

    Bislang erschienen die konkreten Quality-Flags (die *Gründe*) nur im
    `--dry-run`. Diese Zeile zeigt sie auch live, damit das Critic-Routing
    nicht stumm bleibt.
    """
    auto, reason = _decide(note)
    gates = "OK" if note.hard_gates_pass else "fail"
    status = "[Vault]" if auto else f"[Inbox: {reason}]"
    if note.action == "hub":
        status = f"[MoC] {status}"
    line = (f"Score: {note.critic_score}/5 | Hard-Gates: {gates} | "
            f"Confidence: {note.synthesis_confidence} {status}")
    if note.quality_flags:
        line += f"\n        Gründe: {', '.join(note.quality_flags)}"
    return line


def summarize_routing(drafts: list[AtomicNoteDraft]) -> dict:
    """Aggregat über alle Drafts für den Final-Report-Gründe-Block.

    pdfs_modified ist konstruktiv immer 0 (rename=False — die Pipeline mutiert
    Eingabe-PDFs nie); explizit berichtet, weil "0 PDFs verändert" genau die
    Sorglos-Zusicherung ist, die im Lauf bisher unsichtbar blieb.
    """
    vault = inbox = inbox_score_gates = source_unresolved = 0
    for d in drafts:
        auto, _ = _decide(d)
        if auto:
            vault += 1
        else:
            inbox += 1
            inbox_score_gates += 1
        if d.source_status == "unresolved":
            source_unresolved += 1
    return {
        "vault": vault,
        "inbox": inbox,
        "inbox_score_gates": inbox_score_gates,
        "source_unresolved": source_unresolved,
        "pdfs_modified": 0,
    }


def final_report_lines(drafts: list[AtomicNoteDraft]) -> list[str]:
    """Zeilen für den Final-Report: Routing-Verteilung + Gründe-Aufschlüsselung.

    "Quelle offen" steht auf einer EIGENEN Zeile, nicht als Inbox-Unterposten:
    source-status ist orthogonal zum Routing (eine Vault-Note kann unresolved
    sein), sonst entstünde eine irreführende Doppelzählung (X+Y>Inbox).
    """
    s = summarize_routing(drafts)
    return [
        f"   -> Vault:  {s['vault']}",
        f"   -> Inbox:  {s['inbox']} (manuell pruefen)",
        f"   -> Quelle offen: {s['source_unresolved']} (source-status: unresolved)",
        f"   -> {s['pdfs_modified']} PDFs verändert",
    ]


def source_status_framing(source_status: str | None, source_name: str) -> str | None:
    """First-person-NL-Zeile bei unsicherer Quelle — sonst None.

    Bewusst in natürlicher Sprache, ehrlich, erste Person (zu testende Hypothese:
    NL-Unsicherheit senkt Over-Reliance, FAccT 2024). Nur auf dem fail-closed-Pfad
    aktiv — High-Confidence/aufgelöste Quellen bleiben frictionless.
    """
    if source_status != "unresolved":
        return None
    # Wahrheitsgemäß: source-status ist ein Sichtbarkeits-Flag, kein Routing-Gate.
    # Garantiert ist nur: PDF nicht umbenannt (rename=False) + Note markiert.
    return (f"  [Quelle] Ich konnte die Quelle von '{source_name}' nicht "
            f"zuverlässig auflösen (Autor/Jahr/DOI) — ich habe die Eingabedatei "
            f"nicht umbenannt und die betroffenen Notes mit "
            f"`source-status: unresolved` zur Prüfung markiert.")
