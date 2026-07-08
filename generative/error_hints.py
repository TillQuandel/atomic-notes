"""Handlungsanleitende Fehler-Meldungen (#48).

clig.dev: „Catch errors and rewrite them for humans." Jede Meldung nennt den
nächsten konkreten Schritt; bei Setup-/Umgebungsproblemen den `doctor`-Verweis.
Pure Funktionen — der Caller druckt/erhebt.

Zweisprachig (EN default, DE via ATOMIC_AGENT_UI_LANGUAGE, #157): die Wortlaute
leben in `generative.ui_strings`. Die Fehler-ERKENNUNG (Passwort/Auth-Substrings)
bleibt sprachinvariant, ebenso der `doctor`-Verweis.
"""

from __future__ import annotations

from generative.ui_strings import DOCTOR_POINTER, msg

# Substrings, die auf ein Key-/Auth-/Backend-Konfigurationsproblem hindeuten.
_AUTH_MARKERS = (
    "auth",
    "api key",
    "api_key",
    "apikey",
    "401",
    "403",
    "unauthorized",
    "permission",
    "credential",
    "invalid key",
)


def scanned_pdf_hint(pdf_name: str, words_per_page: float | None = None) -> str:
    """Gescanntes/textloses PDF — erklärt das Problem + OCR-Schritt.

    ``words_per_page`` gesetzt → dünner (nicht leerer) Text: die Meldung sagt
    „kaum extrahierbaren Text (nur ~N Wörter/Seite)" statt „keinen" (G6/#27).
    """
    out_name = (pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name) + ".ocr.pdf"
    if words_per_page is not None:
        problem = msg("error.scanned_pdf.problem_thin", wpp=words_per_page)
    else:
        problem = msg("error.scanned_pdf.problem_empty")
    return msg("error.scanned_pdf.body", pdf=pdf_name, problem=problem, out=out_name)


def pdftotext_error_hint(stderr: str | None) -> str:
    """pdftotext-Fehler handlungsanleitend + doctor-Verweis (roher stderr bleibt)."""
    detail = (stderr or "").strip() or msg("error.pdftotext.no_stderr")
    out = msg("error.pdftotext.body", doctor=DOCTOR_POINTER, detail=detail)
    if "password" in detail.lower() or "encrypted" in detail.lower():
        out += msg("error.pdftotext.encrypted")
    return out


def litellm_error_hint(agent: str, model: str, exc: object) -> str:
    """litellm-Fehler handlungsanleitend; Key-/Auth-Fehler bekommen gezielten Hinweis."""
    detail = str(exc)
    base = msg("error.litellm.base", agent=agent, model=model, detail=detail)
    if any(m in detail.lower() for m in _AUTH_MARKERS):
        return base + msg("error.litellm.auth", doctor=DOCTOR_POINTER)
    return base + msg("error.litellm.generic", doctor=DOCTOR_POINTER)
