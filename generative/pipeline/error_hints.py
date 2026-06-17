"""Handlungsanleitende Fehler-Meldungen (#48).

clig.dev: „Catch errors and rewrite them for humans." Jede Meldung nennt den
nächsten konkreten Schritt; bei Setup-/Umgebungsproblemen den `doctor`-Verweis.
Pure Funktionen — der Caller druckt/erhebt.
"""
from __future__ import annotations

_DOCTOR = "→ atomic-notes doctor"

# Substrings, die auf ein Key-/Auth-/Backend-Konfigurationsproblem hindeuten.
_AUTH_MARKERS = ("auth", "api key", "api_key", "apikey", "401", "403",
                 "unauthorized", "permission", "credential", "invalid key")


def scanned_pdf_hint(pdf_name: str) -> str:
    """Gescanntes/textloses PDF — erklärt das Problem + OCR-Schritt."""
    out_name = (pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name) + ".ocr.pdf"
    return (
        f"  [Warnung] '{pdf_name}' enthält keinen extrahierbaren Text — "
        f"vermutlich ein gescanntes PDF. Die Pipeline braucht Text und liefert "
        f"sonst leere/dünne Notes.\n"
        f"  Nächster Schritt: OCR in eine neue Datei ausführen, z. B. "
        f"`ocrmypdf '{pdf_name}' '{out_name}'`, dann mit '{out_name}' erneut starten."
    )


def pdftotext_error_hint(stderr: str | None) -> str:
    """pdftotext-Fehler handlungsanleitend + doctor-Verweis (roher stderr bleibt)."""
    detail = (stderr or "").strip() or "<kein stderr ausgegeben>"
    msg = (
        f"pdftotext konnte das PDF nicht lesen.\n"
        f"  Prüfe, ob poppler-utils installiert und das PDF nicht beschädigt ist "
        f"({_DOCTOR}).\n"
        f"  Original-Fehler: {detail}"
    )
    if "password" in detail.lower() or "encrypted" in detail.lower():
        msg += ("\n  Das PDF ist passwortgeschützt/verschlüsselt — Schutz entfernen, "
                "z. B. `qpdf --decrypt 'input.pdf' 'output.pdf'`, dann erneut starten.")
    return msg


def litellm_error_hint(agent: str, model: str, exc: object) -> str:
    """litellm-Fehler handlungsanleitend; Key-/Auth-Fehler bekommen gezielten Hinweis."""
    detail = str(exc)
    base = f"litellm-Backend-Fehler ({agent}/{model}): {detail}"
    if any(m in detail.lower() for m in _AUTH_MARKERS):
        return (
            f"{base}\n"
            f"  Sieht nach einem Problem mit dem API-Key/Backend aus. Prüfe den "
            f"Provider-Key (z. B. ANTHROPIC_API_KEY/OPENAI_API_KEY) und "
            f"ATOMIC_AGENT_BACKEND ({_DOCTOR})."
        )
    return f"{base}\n  Prüfe Backend/Netzwerk/Modellnamen ({_DOCTOR})."
