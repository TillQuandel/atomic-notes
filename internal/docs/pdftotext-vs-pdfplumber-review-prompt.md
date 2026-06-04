# pdftotext vs pdfplumber Review Prompt

Use this prompt before reopening the pdfplumber Stage-0 adapter. Keep the input P0/public:
technical repo facts, synthetic excerpts, and aggregate metrics only.

```text
Context: atomic-notes generative PDF pipeline.

Current baseline:
- Stage 0 uses pdftotext through pdf_chunker.pdf_to_text(source_path).
- pdf_chunker owns page markers [S. N], frontmatter stripping, split_by_chapters,
  extract_overview, and concept_text_window.
- pdftotext is already integrated and tested.

Prior probe:
- A June 2026 A/B probe compared pdftotext, pdfplumber, and GROBID on Bates and
  Porst PDFs.
- The N>=10 acceptance criterion was not met because the Claude subscription backend
  hit a 5h/session limit.
- Hallucination rates did not separate tools reliably; most cells were N=1.
- GROBID sections increased yield but blew up token cost through 16-18 drafts/run.
- pdfplumber was cheap and simple, but on a two-column Bates PDF it produced glued
  words such as "Thelibraryisagrowing" and 5784 words vs 8846 with pdftotext.

Question:
Evaluate whether pdfplumber should replace or supplement pdftotext in Stage 0.
Do not compare pdfplumber mainly against GROBID; the relevant incumbent is pdftotext.

Input data to inspect:
1. Word counts per extractor.
2. Page-marker count and monotonicity.
3. Sampled text from the same pages, especially multi-column pages.
4. Glued-word indicators, e.g. long mixed-case tokens or missing whitespace around
   common word boundaries.
5. Planner overview differences from extract_overview().
6. Concept yield: planned concepts, produced drafts, written notes, dropped notes.
7. Grounding metrics: anchor hit rate, hallucination rate, coverage/factuality metrics.

Decision options:
A) Keep pdftotext as baseline; do not build pdfplumber now.
B) Add pdfplumber only as an experimental flag, not documented as productive.
C) Build pdfplumber with a guard/fallback to pdftotext.
D) Prefer another lever first, such as overlap chunking or planner recall.

Strict criteria:
- A pdfplumber build is justified only if it shows a yield or grounding gain over
  pdftotext beyond run noise.
- A guard/fallback is justified only if its false-negative and false-positive behavior
  is testable with representative PDFs.
- A naked pdfplumber flag is not acceptable if it can silently ship glued-word
  regressions on two-column PDFs.
- Do not recommend a new dependency unless the advantage over pdftotext is explicit.

Return:
| Severity | Finding | Evidence needed or observed | Recommendation impact |
|---|---|---|---|

End with one final recommendation: A, B, C, or D.
```
