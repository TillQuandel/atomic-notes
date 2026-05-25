# atomic-notes

PDF → atomare Obsidian-Notes via Multi-Agent-Pipeline.

## Stand

**v0.3.66** · **foss-v0.2.0** · letzte Änderung: 2026-05-25

## Struktur

```
agent/          LLM-Pipeline (Claude Opus, 7 Stages)
foss/           FOSS-Pipeline (GLiNER, kein LLM)
shared/         Gemeinsame DB-Schema, Utilities
dashboard/      Eval-Dashboard
```

## Schnellstart

```bash
# Agent-Pipeline
cd agent/
python orchestrator.py --source <pdf> --dry-run    # Test ohne Schreiben
python orchestrator.py --source <pdf>              # Live-Run → 00-inbox/
python orchestrator.py --source <pdf> --no-llm     # FOSS-only

# Dashboard
python eval_dashboard_server.py                    # http://127.0.0.1:8051
```

## Agent-Pipeline (7 Stages)

| Stage | Was | Status |
|---|---|---|
| 1 | PDF → Text + Chunks | stabil |
| 2 | Context-Builder: Vault-Scan → Relevanz-Profil | stabil |
| 3 | Quality-Agent: CrossRef/OpenAlex | stabil |
| 4 | Planner: ConceptPlan (Sonnet) | stabil |
| 5 | Extractor × N parallel (Sonnet) | stabil |
| 6 | Verifier + CrossRef + Critic pro Note (Haiku / `--no-llm`) | stabil |
| 7 | Vault-Writer → 00-inbox/ oder 04-wissen/ | stabil |
| 8 | Eval (LLM-as-Judge via `eval_quality_v4`) | stabil (v0.3.66) |

## FOSS-Pipeline

GLiNER-basierte Extraktion ohne LLM. Stand v0.2.0: 18 PDFs, Ø 8.5% Halluzinationsrate (LLM-Judge v4.1). Extractive-vs-Generative-Gap offen.

Offene TODOs: LexRank-Definitional-Filter (v0.2.1), ER-Stage-1-Threshold, KeyBERT-Fallback-Tuning, Bates-Regression (EN hall 28%).

## Qualitätsziele (agent)

| Metrik | Aktuell | Ziel kurz | Ziel lang |
|---|---|---|---|
| Halluzinationsrate | ~16% (Eval v4.1, margin-TOP_K) | <10% | <5% |
| Inbox-Rate | ~30% | <20% | <10% |
| Coverage | ~52% | >60% | >80% |

Referenz: Claude Sonnet-4.6 Baseline = 10.6% (Vectara Leaderboard Mai 2026).
no-LLM (`--no-llm`): vergleichbare Metriken bei 3× schneller, 4× weniger Tokens — echter A/B-Vergleich ausstehend (`--save-drafts`/`--load-drafts` geplant).

## Calibration (offen)

30 Hybrid-Notes mit Status `to-label` — manuelles Labeling ~3-5h ausstehend.
Nach Labeln: `collect.py` → `adversarial.py` → `kappa.py`. Ziel: AC1 ≥ 0.70.

## GitHub

[TillQuandel/atomic-notes](https://github.com/TillQuandel/atomic-notes) — Issues #1–#14
Milestones: v0.4.0 Bugfixes · v0.5.0 PDF-Extraction-Upgrade · v1.0.0 Public Release
