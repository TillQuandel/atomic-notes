"""Kanonisches JSON-Export-Schema für AtomicNoteDraft — Single Source of Truth
für nachgelagerte Export-Formate (Output-Projekt F2: portable Markdown-Renderer,
F4: CLI/GUI-Formatwahl). Dieses Modul verdrahtet nichts in die Pipeline; es baut
nur reine dict/JSON-Repräsentationen aus bereits vorhandenen Draft-Objekten.

Feld-Semantik
-------------
- `schema_version`: int, siehe Versionierungs-Regel unten.
- `generated_at`: ISO-Datum des Exports (nicht des Pipeline-Laufs).
- `agent_version`: `config.AGENT_VERSION` zum Zeitpunkt des Exports.
- `source`: Quelldatei + `CitationMeta` 1:1 (Autor/Jahr/Titel/DOI), ergänzt um
  die zwei abgeleiteten (`@property`) Felder `short_label` und `display_year` —
  beide werden bei jedem Export frisch aus `CitationMeta` berechnet, nicht
  zwischengespeichert.
- `note`: `dataclasses.asdict(draft)` — ALLE Felder von `AtomicNoteDraft`
  automatisch, ohne Feld-Enumeration hier (Contract bleibt synchron, wenn
  künftig Felder dazukommen). Einzige Abweichung: `note["body"]` wird durch
  `vault_writer.strip_legacy_sections(draft.body)` ersetzt (nur im Export-Dict,
  der Draft selbst bleibt unangetastet). Der Body bleibt sonst ROH — Inline-
  `(S. N)`-Anker, KEINE Footnote-Konvertierung (das ist Aufgabe des jeweiligen
  Renderers, nicht dieses Contracts).
- `routing`: `auto_write_decision(draft)` frisch berechnet — unabhängig vom
  Draft-Feld `note.auto_vault_recommended`, das nur die zum Zeitpunkt der
  Draft-Erzeugung gültige Einschätzung ist (und `None` sein kann, wenn der
  Export vor `write_note` läuft). Bewusst werden BEIDE exportiert:
  `note.auto_vault_recommended` (Draft-Snapshot, kann null sein) und
  `routing.auto_vault_recommended` (immer frisch, nie null).

Versionierungs-Regel
---------------------
`SCHEMA_VERSION` wird bei jeder rückwärts-inkompatiblen Änderung erhöht (Felder
entfernt/umbenannt, Typ eines Felds geändert, Struktur eines verschachtelten
Objekts geändert). Rein additive Änderungen (neues optionales Feld) erfordern
keinen Bump.

Abgrenzung zu `orchestrator._save_draft_state`
-----------------------------------------------
`_save_draft_state` ist ein internes JSON-Resume-Format OHNE
Stabilitätsgarantie — es dient ausschließlich dazu, einen unterbrochenen
Pipeline-Lauf fortzusetzen, und darf sich zwischen Versionen beliebig ändern. `note.json` (dieses Modul) ist dagegen der stabile
EXTERNE Contract für Konsumenten außerhalb der Pipeline (F2-Renderer, F4-CLI/
GUI, externe Tools). Die beiden Formate dürfen nicht verwechselt oder
ineinander verschmolzen werden.

Naming-Konvention: JSON-Keys sind snake_case (Programmier-Contract, wie der
Rest der Codebase) — im Gegensatz zur kebab-case-Frontmatter der Obsidian-
Notes (Vault-Konvention, siehe Schema-Base). Das ist bewusst zwei verschiedene
Namensräume für zwei verschiedene Konsumenten.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from typing import Optional

from generative import config
from generative.pipeline.vault_writer import auto_write_decision, strip_legacy_sections
from generative.schemas.atomic_note import AtomicNoteDraft
from generative.schemas.citation import CitationMeta

SCHEMA_VERSION = 1


def _source_block(citation: CitationMeta) -> dict:
    return {
        "file": citation.source_file,
        "citation": {
            "author": citation.author,
            "year": citation.year,
            "title": citation.title,
            "doi": citation.doi,
            "short_label": citation.short_label,
            "display_year": citation.display_year,
        },
    }


def _note_and_routing(draft: AtomicNoteDraft) -> dict:
    """Baut den `{note, routing}`-Eintrag für einen einzelnen Draft — gemeinsamer
    Helper für `note_to_json_dict` (einzeln) und `run_to_json_dict` (Sammel-Form),
    damit beide Formen nicht auseinanderdriften."""
    note_dict = dataclasses.asdict(draft)
    note_dict["body"] = strip_legacy_sections(note_dict["body"])
    auto, reason = auto_write_decision(draft)
    return {
        "note": note_dict,
        "routing": {
            "auto_vault_recommended": auto,
            "reason": reason,
        },
    }


def note_to_json_dict(
    draft: AtomicNoteDraft,
    citation: CitationMeta,
    *,
    generated_at: Optional[str] = None,
) -> dict:
    """Export-Contract für einen einzelnen Draft. Pure: mutiert `draft` nie."""
    entry = _note_and_routing(draft)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or date.today().isoformat(),
        "agent_version": config.AGENT_VERSION,
        "source": _source_block(citation),
        "note": entry["note"],
        "routing": entry["routing"],
    }


def run_to_json_dict(
    drafts: list[AtomicNoteDraft],
    citation: CitationMeta,
    *,
    generated_at: Optional[str] = None,
) -> dict:
    """Sammel-Contract für einen kompletten Pipeline-Lauf (mehrere Drafts, eine
    Quelle). Gleicher Umschlag wie `note_to_json_dict`, aber `notes`-Liste statt
    `note`/`routing` — kein per-Note-`source`-Duplikat, da die Quelle für den
    gesamten Lauf identisch ist."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or date.today().isoformat(),
        "agent_version": config.AGENT_VERSION,
        "source": _source_block(citation),
        "notes": [_note_and_routing(draft) for draft in drafts],
    }


def dumps(obj: dict) -> str:
    """Serialisiert einen Export-Dict als lesbares JSON: Umlaute/Sonderzeichen
    literal (kein `\\uXXXX`-Escaping), 2-Space-Einrückung, trailing newline."""
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
