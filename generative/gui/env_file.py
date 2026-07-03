"""Merge-Write fuer `generative/.env` (B1b): litellm-API-Key aus der GUI setzen.

Reine Datei-/Str-Logik, kein FastAPI (analog gui_settings.py/run_history.py).
Schreibt/aktualisiert GENAU eine `NAME=WERT`-Zeile, alle anderen Zeilen
(Kommentare, Leerzeilen, andere Variablen) bleiben unveraendert und in
Reihenfolge erhalten. Der Pfad ist beim Aufrufer (app.py, `create_app`) fest
verdrahtet -- dieses Modul kennt keinen Request/Client-Pfad.

Sicherheit: `validate_key_value` blockt Control-Chars (inkl. `\r`/`\n`/`\0`)
im key-Wert -- sonst koennte ein `\n` im Wert eine zweite `.env`-Zeile
injizieren (z.B. `ATOMIC_AGENT_BACKEND=evil`). Kein Quoting beim Schreiben
noetig, da Control-Chars bereits ausgeschlossen sind.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

# < 0x20 (Tab, CR, LF, NUL, ...) oder DEL (0x7f) -- deckt alle gaengigen
# Zeilenumbruch-/Injection-Zeichen ab, ohne normale Key-Zeichen (auch
# Sonderzeichen wie -, _, /, :, .) einzuschraenken.
_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20)) | {"\x7f"}

# Read-Modify-Write der .env-Datei ist NICHT parallelsicher ohne Lock --
# zwei gleichzeitige POSTs koennten sich sonst gegenseitig ueberschreiben
# (Lost Update). Modulweiter Lock statt pro-Pfad, da es in dieser App nur
# eine `.env`-Datei gibt.
_LOCK = threading.Lock()


def validate_key_value(value: object) -> tuple[str | None, str | None]:
    """Prueft einen key-Wert aus `POST /api/access/litellm-key`.

    Rueckgabe: (bereinigter_wert, fehlermeldung). Control-Chars werden VOR dem
    Trimmen auf dem Rohwert geprueft (auch ein Trailing-`\n` waere sonst durch
    `.strip()` unsichtbar geworden, obwohl er als Eingabe abgelehnt werden
    soll). Fuehrende/nachfolgende Leerzeichen werden getrimmt; ein danach
    leerer String gilt als "kein Key".
    """
    if not isinstance(value, str):
        return None, "key fehlt oder ist kein String."
    if any(ch in _CONTROL_CHARS for ch in value):
        return None, "ungueltige Zeichen im Key."
    stripped = value.strip()
    if not stripped:
        return None, "key darf nicht leer sein."
    return stripped, None


def write_env_var(name: str, value: str, path: Path | str) -> Path:
    """Schreibt/aktualisiert `NAME=WERT` in der `.env`-Datei unter `path`.

    Existiert die Zeile bereits (erste Fundstelle per `NAME=`-Praefix), wird
    NUR sie ersetzt -- alle anderen Zeilen (Kommentare, Leerzeilen, andere
    Variablen) bleiben unveraendert und in Reihenfolge. Existiert sie nicht,
    wird sie ans Ende angehaengt. Existiert die Datei nicht, wird sie neu
    angelegt. Atomar (Tempfile im selben Verzeichnis + `os.replace`), analog
    `gui_settings.write_settings`/`run_history.write_run_record`.
    """
    path = Path(path)
    new_line = f"{name}={value}\n"
    prefix = f"{name}="
    with _LOCK:
        if path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        else:
            lines = []
        out_lines: list[str] = []
        replaced = False
        for line in lines:
            if not replaced and line.startswith(prefix):
                out_lines.append(new_line)
                replaced = True
            else:
                out_lines.append(line)
        if not replaced:
            # Falls die letzte bestehende Zeile ohne Newline endet (Datei ohne
            # abschliessenden Zeilenumbruch): erst trennen, sonst verschmelzen
            # die neue Zeile mit der vorherigen.
            if out_lines and not out_lines[-1].endswith("\n"):
                out_lines[-1] += "\n"
            out_lines.append(new_line)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".env")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return path
