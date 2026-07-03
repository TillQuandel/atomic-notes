"""Test-Isolation fuer Run-Historie (P4) und persistierte Einstellungen (P2).

`create_app()` schreibt am Ende jedes Laufs (auch in Tests, die den Fake-Lauf
bis `exited` durchlaufen lassen) einen Historie-Record, und `PUT /api/settings`
schreibt eine Settings-Datei. Ohne Isolation landeten beide im echten
`generative/.cache/gui/` — diese autouse-Fixtures zeigen die Default-Pfade fuer
jeden Test auf sein eigenes `tmp_path` um, ohne die bestehenden
`create_app(...)`-Aufrufe einzeln anfassen zu muessen.
"""

import pytest

from generative.gui import app as app_module


@pytest.fixture(autouse=True)
def _isolate_default_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_DEFAULT_RUNS_DIR", tmp_path / "gui-runs-default")


@pytest.fixture(autouse=True)
def _isolate_default_settings_path(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_DEFAULT_SETTINGS_PATH", tmp_path / "gui-settings-default" / "settings.json")


@pytest.fixture(autouse=True)
def _isolate_default_env_path(tmp_path, monkeypatch):
    # B1b: Sicherheitsnetz zusaetzlich zu `create_app(env_path=...)` -- selbst
    # ein Test, der `env_path` vergisst zu injizieren, darf NIE die echte
    # `generative/.env` schreiben.
    monkeypatch.setattr(app_module, "_DEFAULT_ENV_PATH", tmp_path / "gui-env-default" / ".env")
