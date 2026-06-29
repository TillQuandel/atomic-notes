"""Tests für handlungsanleitende Fehler-Meldungen (#48).

clig.dev: „catch errors and rewrite them for humans" — jede Meldung nennt den
nächsten Schritt; bei Setup-Problemen den `doctor`-Verweis.
"""


class TestScannedPdfHint:
    def test_names_file_and_suggests_ocr(self):
        from generative.pipeline.error_hints import scanned_pdf_hint

        msg = scanned_pdf_hint("Buch.pdf")
        assert "Buch.pdf" in msg
        assert "ocrmypdf" in msg
        # erklärt das Problem (kein Text / gescannt)
        assert "gescannt" in msg.lower() or "kein text" in msg.lower()


class TestScannedHintThinVariant:
    def test_thin_text_says_kaum_not_keinen(self):
        # Dünner (nicht leerer) Text: die "enthält keinen Text"-Formulierung wäre
        # sachlich falsch (G6/#27). words_per_page wird genannt, OCR-Schritt bleibt.
        from generative.pipeline.error_hints import scanned_pdf_hint

        msg = scanned_pdf_hint("Scan.pdf", words_per_page=12.0)
        assert "kaum" in msg.lower()
        assert "12" in msg
        assert "ocrmypdf" in msg

    def test_empty_default_unchanged(self):
        # Ohne words_per_page bleibt die bestehende "keinen Text"-Meldung.
        from generative.pipeline.error_hints import scanned_pdf_hint

        msg = scanned_pdf_hint("Scan.pdf")
        assert "keinen extrahierbaren text" in msg.lower()


class TestPdftotextErrorHint:
    def test_actionable_with_doctor_pointer(self):
        from generative.pipeline.error_hints import pdftotext_error_hint

        msg = pdftotext_error_hint("some raw stderr")
        assert "doctor" in msg
        assert "pdftotext" in msg
        # roher stderr bleibt zur Diagnose enthalten
        assert "some raw stderr" in msg


class TestScannedHintUsesDistinctOcrOutput:
    def test_ocr_command_does_not_overwrite_input_in_place(self):
        from generative.pipeline.error_hints import scanned_pdf_hint

        msg = scanned_pdf_hint("Buch.pdf")
        # ocrmypdf akzeptiert nicht denselben In-/Out-Pfad → distinkter Output
        assert "Buch.pdf' 'Buch.pdf'" not in msg
        assert ".ocr.pdf" in msg


class TestPdftotextHintRobust:
    def test_none_stderr_does_not_crash(self):
        from generative.pipeline.error_hints import pdftotext_error_hint

        msg = pdftotext_error_hint(None)
        assert "doctor" in msg

    def test_encrypted_pdf_gets_password_hint(self):
        from generative.pipeline.error_hints import pdftotext_error_hint

        msg = pdftotext_error_hint("Command Line Error: Incorrect password")
        assert "qpdf" in msg or "passwort" in msg.lower()


class TestPdfToPagesMissingBinary:
    def test_missing_pdftotext_gives_actionable_exit(self, monkeypatch):
        # fehlendes pdftotext wirft OSError VOR dem returncode-Check → muss
        # trotzdem die handlungsanleitende Meldung (+ doctor) liefern.
        import subprocess
        from generative.pipeline import pdf_chunker

        def _raise(*a, **k):
            raise FileNotFoundError("pdftotext not found")

        monkeypatch.setattr(subprocess, "run", _raise)
        import pytest

        with pytest.raises(SystemExit) as exc:
            pdf_chunker.pdf_to_pages(__import__("pathlib").Path("x.pdf"))
        assert "doctor" in str(exc.value)


class TestLitellmErrorHint:
    def test_key_error_gets_targeted_hint(self):
        from generative.pipeline.error_hints import litellm_error_hint

        msg = litellm_error_hint("extractor", "gpt-4", "AuthenticationError: invalid api key")
        assert "doctor" in msg
        # weist auf den Key/das Backend hin
        assert "key" in msg.lower() or "schlüssel" in msg.lower()

    def test_generic_error_still_actionable(self):
        from generative.pipeline.error_hints import litellm_error_hint

        msg = litellm_error_hint("planner", "claude-x", "Timeout reading response")
        assert "doctor" in msg
        assert "planner" in msg or "claude-x" in msg
        assert "Timeout reading response" in msg
