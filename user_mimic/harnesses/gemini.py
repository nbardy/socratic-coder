from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from user_mimic.canonical import Thread

log = logging.getLogger(__name__)

GEMINI_ROOT = Path.home() / ".gemini" / "history"


def iter_session_files(root: Path = GEMINI_ROOT) -> Iterator[Path]:
    if not root.exists():
        return
    for _ in root.rglob("*"):
        pass
    return
    yield  # pragma: no cover


def load_threads(root: Path = GEMINI_ROOT) -> Iterator[Thread]:
    log.warning("gemini adapter not implemented (root=%s)", root)
    return
    yield  # pragma: no cover
