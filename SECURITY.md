# Security & Privacy

## Threat model

atomic-notes is a **local, single-user tool**. There is no authentication, by
design: the GUI (`atomic-notes gui`) and the eval dashboard bind to
`127.0.0.1` and defend against DNS-rebinding/CSRF with Host-header allowlists
and same-origin checks, but they trust any process on the same machine.
Concretely, **running the GUI makes every `.md` file in the configured vault
readable to any local process via `localhost`** (`GET /api/outputs/file`
streams any `.md` under the vault root; the path whitelist blocks traversal but
not local access). Treat the vault, the config, and the `.cache` directory as
data you are willing to expose to anything else running under your user account.

## What leaves your machine

The pipeline is local-first. The only outbound traffic is the LLM call itself
and read-only bibliographic lookups; everything else is opt-in.

| Data | Leaves to | When |
| --- | --- | --- |
| PDF text (chunks/prompts) | the configured provider (Anthropic, OpenAI, Ollama, …) | `litellm` backend (`ATOMIC_AGENT_BACKEND=litellm`) |
| PDF text (via the local `claude` CLI) | Anthropic | default `subscription` backend, through your own Claude account |
| Nothing (source sentences only, no generation) | — | `extractive` pipeline, or a local `litellm` provider such as Ollama |
| Bibliographic **identifiers and titles only** (DOI, ISBN, arXiv ID, PMID, work title — never PDF body text) | Crossref, OpenAlex, arXiv, PubMed, Open Library, Google Books | **every backend mode**, whenever the quality agent / PDF enrichment resolves a citation |
| Trace metadata (note titles, model/agent names, token counts, prompt hashes, scores) **plus HTTP Basic-Auth credentials** | `LANGFUSE_HOST` | only if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set and `ATOMIC_AGENT_TRACING=langfuse` |
| LLM spans incl. full prompt and completion text | local Phoenix server (`http://localhost:6006`) | only if Phoenix tracing is enabled — stays on the machine |

Notes:

- **Metadata lookups always run**, independent of the LLM backend. They send
  only identifiers and titles, never source text
  (`generative/agents/quality.py`, `generative/tools/pdf_enrich.py`).
- **Langfuse does not receive raw prompts.** The custom REST backend forwards
  only hashes and metadata (litellm's native Langfuse callback is disabled).
  It *does* send your Basic-Auth keys. If `LANGFUSE_HOST` is a non-local
  `http://` endpoint, those keys and the metadata travel in cleartext — the
  backend logs a one-time warning in that case
  (`generative/agents/langfuse_backend.py`).

## Local artifacts (nothing leaves the machine, but note the contents)

Under `generative/.cache/`:

- **`runs/<run-id>.jsonl`** — per-call trace lines. Carry `prompt_hash`, token
  counts and stats, **not** the prompts themselves.
- **`llm/<hash>.json`** — LLM response cache. Holds full model completions
  (the generated note text), keyed by hash; no prompt text.
- **`failed/<run-id>/<slug>.json`** — Stage-6 crash reports. These contain the
  **full prompt and PDF text excerpt** (`prompt`, `raw_output`, `draft_body`)
  of the note that crashed, so a failure stays diagnosable
  (`generative/orchestrator.py`, `generative/pipeline/crash_report.py`). No
  secrets, but real source content — delete before sharing a `.cache` folder.
- **`gui/uploads/`** — PDFs uploaded through the GUI are stored here.

## Reporting a vulnerability

This is a solo project, so there is no separate private-disclosure mailbox:
please **open a GitHub issue** at
<https://github.com/TillQuandel/atomic-notes/issues>. For a sensitive finding
you would rather not post publicly, contact the maintainer directly via GitHub
(profile [@TillQuandel](https://github.com/TillQuandel)) instead of filing a
public issue.
