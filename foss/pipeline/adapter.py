from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_ENV = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
_FORMAT_MAP = {
    "obsidian": "obsidian.md.jinja2",
    "md": "generic.md.jinja2",
    "json": "note.json.jinja2",
}


def render_note(note, output_format: str = "obsidian") -> str:
    return _ENV.get_template(_FORMAT_MAP.get(output_format, "obsidian.md.jinja2")).render(note=note)


def write_note(note, out_dir: Path, output_format: str = "obsidian") -> Path:
    ext = "json" if output_format == "json" else "md"
    slug = note.title.lower().replace(" ", "-")[:60]
    path = out_dir / f"{slug}.{ext}"
    path.write_text(render_note(note, output_format), encoding="utf-8")
    return path
