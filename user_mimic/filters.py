from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from user_mimic.canonical import UserTurn

_GIT_ROOT = Path(os.environ.get("USER_MIMIC_GIT_ROOT", Path.home() / "git"))

# Canonical hide-tag regexes from ~/git/unleashd/server/src/adapters/jsonl.ts
# (HIDE_TEST_RE, AI_WRITING_TOOL_RE, OOMPA_RE). Applied to the FIRST user
# message only — matches unleashd's thread-level classification semantics.
HIDE_TEST_RE = re.compile(r"^\s*(?:\"|')?\s*\[_HIDE_TEST_\]\s*")
AI_WRITING_TOOL_RE = re.compile(r"^\s*(?:\"|')?\s*\[ai-writing-tool\]\s*")
OOMPA_TAG_RE = re.compile(r"^\s*(?:\"|')?\s*\[oompa(?::([^:\]]+)(?::([^\]]+))?)?\]")

OOMPA_WORKTREE_RE = re.compile(r"\.(ww\d+-i\d+|ws[0-9a-f]+-w\d+-i\d+)(?:/|$)")
OOMPA_CWD_ROOTS = (
    str(_GIT_ROOT / "oompa"),
    str(_GIT_ROOT / "oompa_loompas"),
)
OOMPA_PROJECT_SLUG_MARK = "-private-tmp-claude-"

HOOK_INJECTED_PREFIXES = (
    "<system-reminder>",
    "<command-name>",
    "<user-prompt-submit-hook>",
    "Caveat:",
    "<local-command-stdout>",
    "<local-command-caveat>",
)

MIN_NICK_WORDS = 2


@dataclass
class DropCounts:
    hide_test_thread: int = 0
    ai_writing_thread: int = 0
    oompa_tag_thread: int = 0
    oompa_cwd: int = 0
    oompa_worktree: int = 0
    oompa_project_slug: int = 0
    hook_injected: int = 0
    too_short: int = 0
    too_long_target: int = 0
    sidechain: int = 0
    tool_result_user: int = 0

    def total(self) -> int:
        return (
            self.hide_test_thread
            + self.ai_writing_thread
            + self.oompa_tag_thread
            + self.oompa_cwd
            + self.oompa_worktree
            + self.oompa_project_slug
            + self.hook_injected
            + self.too_short
            + self.too_long_target
            + self.sidechain
            + self.tool_result_user
        )


def is_oompa_cwd(cwd: str) -> bool:
    if not cwd:
        return False
    for root in OOMPA_CWD_ROOTS:
        if cwd == root or cwd.startswith(root + "/"):
            return True
    return False


def is_oompa_worktree(cwd: str) -> bool:
    if not cwd:
        return False
    return bool(OOMPA_WORKTREE_RE.search(cwd))


def is_oompa_project_slug(slug: str) -> bool:
    return OOMPA_PROJECT_SLUG_MARK in (slug or "")


def classify_first_user_message(text: str) -> str | None:
    """Mirror unleashd's extractWorkerMetadata. Returns 'hide_test', 'ai_writing',
    'oompa', or None. Applied ONCE against the thread's first user message."""
    t = text or ""
    if HIDE_TEST_RE.match(t):
        return "hide_test"
    if AI_WRITING_TOOL_RE.match(t):
        return "ai_writing"
    if OOMPA_TAG_RE.match(t):
        return "oompa"
    return None


def is_hook_injected(text: str) -> bool:
    s = (text or "").lstrip()
    return any(s.startswith(p) for p in HOOK_INJECTED_PREFIXES)


def is_user_message(turn: UserTurn, cwd: str, counts: DropCounts) -> bool:
    if turn.is_sidechain:
        counts.sidechain += 1
        return False
    text = turn.text or ""
    if not text.strip():
        counts.too_short += 1
        return False
    if is_hook_injected(text):
        counts.hook_injected += 1
        return False
    if is_oompa_cwd(cwd):
        counts.oompa_cwd += 1
        return False
    if is_oompa_worktree(cwd):
        counts.oompa_worktree += 1
        return False
    if len(text.split()) < MIN_NICK_WORDS:
        counts.too_short += 1
        return False
    return True
