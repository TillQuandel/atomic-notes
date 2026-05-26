from __future__ import annotations
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from shared.path_safety import contained_child_path, safe_filename_stem

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_ENV = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR), encoding="utf-8-sig"))
_FORMAT_MAP = {
    "obsidian": "obsidian.md.jinja2",
    "md": "generic.md.jinja2",
    "json": "note.json.jinja2",
}

_PAGE_ANCHOR_RE = re.compile(r"\(S\.\s*(\d+(?:-\d+)?)\)")


def convert_anchors_to_footnotes(body: list[str], source_file: str) -> tuple[list[str], str]:
    """Konvertiert (S. N)-Anker zu Obsidian-Footnotes [^N] mit Definitions-Block.
    Analog zu atomic-agent vault_writer.convert_inline_to_footnotes().
    """
    counter = 1
    page_to_fn: dict[str, int] = {}
    converted: list[str] = []

    for sentence in body:
        def replace(m: re.Match) -> str:
            nonlocal counter
            page = m.group(1)
            if page not in page_to_fn:
                page_to_fn[page] = counter
                counter += 1
            return f"[^{page_to_fn[page]}]"
        converted.append(_PAGE_ANCHOR_RE.sub(replace, sentence))

    # Footnote-Definitions-Block
    stem = Path(source_file).stem
    defs = "\n".join(
        f"[^{fn}]: *{stem}* · S. {page}"
        for page, fn in sorted(page_to_fn.items(), key=lambda x: x[1])
    )
    return converted, defs


def render_note(note, output_format: str = "obsidian") -> str:
    if output_format == "obsidian":
        body_converted, footnote_defs = convert_anchors_to_footnotes(
            note.extracted_body, note.source_file
        )
        return _ENV.get_template("obsidian.md.jinja2").render(
            note=note,
            body_converted=body_converted,
            footnote_defs=footnote_defs,
        )
    return _ENV.get_template(_FORMAT_MAP.get(output_format, "obsidian.md.jinja2")).render(note=note)


def write_note(note, out_dir: Path, output_format: str = "obsidian") -> Path:
    ext = "json" if output_format == "json" else "md"
    slug = safe_filename_stem(note.title)
    path = contained_child_path(out_dir, f"{slug}.{ext}")
    path.write_text(render_note(note, output_format), encoding="utf-8")
    return path
