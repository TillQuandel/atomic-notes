"""a11y-Pass (P8): stdlib-only Struktur-Checks gegen `static/index.html`.

Kein axe-core/Browser noetig -- das eigentliche axe-Audit lief manuell (siehe
P8-Spec). Diese Tests halten die drei damit gefundenen/behobenen Muster fest,
damit sie nicht unbemerkt zurueckkommen:
  (a) jedes Formularfeld hat eine Label-Bindung,
  (b) Ueberschriften-Ebenen springen nicht (kein hN ohne h(N-1) zuvor),
  (c) kein `role="button"` auf `<li>` (axe: aria-allowed-role).
"""

from html.parser import HTMLParser
from pathlib import Path

STATIC_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"

# Void-Elemente: kein schliessendes Tag im Quelltext, daher nie auf den
# Nesting-Stack legen (sonst haengt der Stack falsch, weil handle_endtag
# fuer sie nie aufgerufen wird).
_VOID_TAGS = {"input", "meta", "link", "br", "hr", "img", "source", "col"}


class _A11yParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._stack = []
        self.label_fors = set()
        self.controls = []  # [{tag, attrs, in_label}]
        self.heading_levels = []  # Dokument-Reihenfolge, z.B. [1, 2, 2, ...]
        self.li_roles = []  # role-Attribut (oder None) jedes <li>

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "label" and d.get("for"):
            self.label_fors.add(d["for"])
        if tag in ("input", "select"):
            in_label = "label" in self._stack
            self.controls.append({"tag": tag, "attrs": d, "in_label": in_label})
        if tag == "li":
            self.li_roles.append(d.get("role"))
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self.heading_levels.append(int(tag[1]))
        if tag not in _VOID_TAGS:
            self._stack.append(tag)

    def handle_endtag(self, tag):
        if not self._stack:
            return
        if self._stack[-1] == tag:
            self._stack.pop()
        elif tag in self._stack:
            while self._stack and self._stack[-1] != tag:
                self._stack.pop()
            if self._stack:
                self._stack.pop()


def _parse():
    parser = _A11yParser()
    parser.feed(STATIC_INDEX.read_text(encoding="utf-8"))
    return parser


def _is_exempt(control):
    attrs = control["attrs"]
    if control["tag"] != "input":
        return False
    if attrs.get("type") == "hidden":
        return True
    # z.B. #file-input: per JS-Klick ausgeloest, fuer AT/Tab-Reihenfolge
    # bewusst unsichtbar -- Label waere fuer niemanden erreichbar.
    if attrs.get("type") == "file" and "hidden" in attrs:
        return True
    return False


def test_all_inputs_and_selects_have_label_binding():
    parser = _parse()
    assert parser.controls, "Parser hat keine input/select-Elemente gefunden -- Test kaputt?"
    unlabeled = []
    for control in parser.controls:
        if _is_exempt(control):
            continue
        attrs = control["attrs"]
        labeled = control["in_label"] or bool(attrs.get("aria-label")) or attrs.get("id") in parser.label_fors
        if not labeled:
            unlabeled.append((control["tag"], attrs.get("id"), attrs.get("type")))
    assert not unlabeled, f"Controls ohne Label-Bindung: {unlabeled}"


def test_heading_levels_do_not_skip():
    parser = _parse()
    assert parser.heading_levels, "Parser hat keine Ueberschriften gefunden -- Test kaputt?"
    max_seen = 0
    for level in parser.heading_levels:
        assert level <= max_seen + 1, f"Ueberschriften-Sprung: h{level} nach hoechstens h{max_seen} gesehen"
        max_seen = max(max_seen, level)


def test_no_role_button_on_list_items():
    parser = _parse()
    assert parser.li_roles, "Parser hat keine <li>-Elemente gefunden -- Test kaputt?"
    assert "button" not in parser.li_roles
