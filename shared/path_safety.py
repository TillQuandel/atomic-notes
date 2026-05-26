from __future__ import annotations

import os
import re
from pathlib import Path

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_DASH_RUN = re.compile(r"-+")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename_stem(value: str, *, max_len: int = 60, fallback: str = "note") -> str:
    stem = str(value or "").lower().replace(" ", "-")
    stem = _UNSAFE_FILENAME_CHARS.sub("-", stem)
    stem = _DASH_RUN.sub("-", stem).strip(" .-_")
    if not stem:
        stem = fallback
    stem = stem[:max_len].strip(" .-_") or fallback
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"{stem}-{fallback}"
    return stem


def contained_child_path(parent: Path, filename: str) -> Path:
    candidate = parent / filename
    base = os.path.abspath(os.path.normpath(os.fspath(parent)))
    target = os.path.abspath(os.path.normpath(os.path.join(base, filename)))
    if os.path.commonpath([base, target]) != base:
        raise ValueError(f"unsafe output path outside target directory: {filename}")
    return candidate
