# atomic-notes/foss/ Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** foss/ Pipeline im atomic-notes Monorepo -- PDF zu Atomic Notes ohne LLM-API.

**Architecture:** pdfplumber -> GLiNER (Zero-Shot NER) -> LexRank (Sentence-Cluster) -> FOSS Stage 6 -> Pydantic AtomicNoteFoss -> Jinja2 Adapter.

**Tech Stack:** pdfplumber, gliner, sumy (LexRank), nltk, sentence-transformers, rank-bm25, rapidfuzz, pydantic, jinja2, langdetect

---

## File Map

| Datei | Verantwortung |
|---|---|
| foss/pipeline/pdf_chunker.py | pdfplumber -> Chunks |
| foss/pipeline/gliner_planner.py | GLiNER -> Konzeptliste |
| foss/pipeline/sentence_extractor.py | LexRank -> Body-Saetze |
| foss/pipeline/adapter.py | Pydantic -> Jinja2 |
| foss/eval/foss_eval.py | Anchor-Rate, Halluzinationsrate, JSONL |
| foss/templates/*.jinja2 | obsidian / generic / json |
| foss/orchestrator.py | CLI |
| foss/requirements.txt | Dependencies |
| shared/schemas/atomic_note_foss.py | Pydantic AtomicNoteFoss |
| tests/foss/ | Unit + E2E Tests |

Siehe Spec: docs/superpowers/specs/2026-05-20-atomic-notes-monorepo-design.md

---

## Tasks: T1 Requirements, T2 pdf_chunker, T3 gliner_planner, T4 sentence_extractor, T5 Pydantic+Jinja2, T6 foss_eval, T7 orchestrator, T8 E2E-Tests, T9 README

Vollstaendiger Plan mit Code: wird in Implementierungssession ausgefuellt via superpowers:executing-plans.

Implementierungsreihenfolge: T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> T8 -> T9
Jeder Task: failing test schreiben, FAIL verifizieren, implementieren, PASS verifizieren, committen.
