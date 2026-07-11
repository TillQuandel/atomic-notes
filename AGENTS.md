# AGENTS.md — Kontext für KI-Sessions in diesem Repo

Diese Datei ist die erste Pflichtlektüre jeder KI-Session (Claude Code, Codex,
Cursor, …). Sie beantwortet: Was ist dieses Projekt, für wen, und nach welchen
Maximen werden Entscheidungen getroffen.

## Was

atomic-notes verwandelt Quell-PDFs in **atomare, quellenverankerte
Konzept-Notes**: ein generativer LLM-Pfad (7-Agenten-Pipeline mit
Verifier/Critic/Faithfulness-Gates) und ein extraktiver Offline-Pfad (0 LLM).
Output: Obsidian-Markdown, JSON, portables Markdown, docx/pdf/html/odt/epub.
Lokale Web-GUI. Eigenständiges Tool, **lokal-first** — kein Hosted-Dienst in v1.

## Für wen

- **Primär:** Studierende in der Abschlussarbeit (Bachelor bis PhD) mit einem
  Quellen-Stapel von Dutzenden bis wenigen hundert PDFs über die
  Projektlaufzeit, Atomic-Notes-Workflow bekannt oder gewünscht.
- **Sekundär:** Postdocs/Forschende mit Lit-Review-Last; PKM-Power-User mit
  Lesestapel.
- **Bewusst nicht (v1):** B2B/Unternehmens-Wissenskataloge.

Das heißt konkret: Nutzer sind NICHT der Maintainer. Sie haben beliebige
Hardware (8-GB-Laptops sind normal), englische Systeme (UI ist EN-Default,
DE via `ATOMIC_AGENT_UI_LANGUAGE`), und entweder eine Claude-Subscription
ODER einen API-Key (litellm-Backend: Anthropic/OpenAI/Gemini/Ollama).
Die Pipeline ist bewusst modell-agnostisch.

## Design-Maximen (gegen jede Empfehlung/Entscheidung halten)

1. **Produkt vor Maintainer-Workaround:** Eine Lösung muss für einen fremden
   Nutzer auf fremder Hardware funktionieren. ENV-Tuning, „andere Programme
   schließen" o. Ä. sind Diagnose-Hilfen — die Produkt-Antwort ist adaptives,
   selbsterklärendes Verhalten (erkennen → drosseln → klar melden) statt
   stillem Scheitern.
2. **Keine privaten Annahmen:** Keine Maintainer-Pfade, -Vault-Konventionen
   oder -Setups in Code, CLI-Texten oder Doku (siehe Entpersonalisierung in
   #158). Was ein Nutzer sieht, muss ohne fremden Kontext verständlich sein.
3. **Faktentreue ist der USP:** Anker-Verifikation, Quellen-Bindung und
   Eval-Gates sind der Kern des Produkts — nie für Convenience oder Tempo
   aufweichen. Prompt-/Eval-Umbauten nie ungemessen als „neutral" deklarieren.
4. **Privacy-Kontrakt ehrlich halten:** Metadaten-Lookups (Crossref, OpenAlex
   u. a.) laufen in jedem Backend-Modus; Grenzen stehen in README §Privacy und
   SECURITY.md — Doku-Aussagen dazu müssen dem Code entsprechen.

## Arbeits-Konventionen

- Kanonische Test-Suite: `pytest generative lib/decision_engine/tests
  shared/tests -q` (bare `pytest` sammelt dasselbe). CI-Gates: `ruff format
  --check .` und `ruff check .` — vor jedem Commit lokal laufen lassen.
- TDD für Fixes (RED zeigen, dann fixen); reale Befunde als GitHub-Issues
  persistieren.
- Commits/PRs ohne Attribution-Footer (kein Co-Authored-By, kein
  Generated-with).

## Offene Grundsatzentscheidungen (nicht vorgreifen)

- Distributionsmodell (Voraussetzung u. a. für weitere Provider-Zugänge).
- Umfang des v1.0.0-Releases (#11).

Mehr Kontext: README.md (Nutzer-Doku), ARCHITECTURE.md, SECURITY.md,
CONTRIBUTING.md.
