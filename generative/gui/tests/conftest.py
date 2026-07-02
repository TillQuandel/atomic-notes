"""Test-Isolation fuer die Run-Historie (P4).

`create_app()` schreibt am Ende jedes Laufs (auch in Tests, die den Fake-Lauf
bis `exited` durchlaufen lassen) einen Historie-Record. Ohne Isolation
landeten diese Records im echten `generative/.cache/gui/runs/` — dieser
autouse-Fixture zeigt den Default-Pfad fuer jeden Test auf sein eigenes
`tmp_path` um, ohne die ~30 bestehenden `create_app(...)`-Aufrufe einzeln
anfassen zu muessen (Scope: nur P4-Historie, keine unrelated Test-Edits).
"""

import pytest

from generative.gui import app as app_module


@pytest.fixture(autouse=True)
def _isolate_default_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_DEFAULT_RUNS_DIR", tmp_path / "gui-runs-default")
