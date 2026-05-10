from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

HOME = Path.home()
GIT_ROOT = Path(os.environ.get("NICK_MIMIC_GIT_ROOT", HOME / "git"))

from nick_mimic.canonical import (
    AssistantTurn,
    Sample,
    SystemTurn,
    Thread,
    ToolCallTurn,
    ToolResultTurn,
    Turn,
    UserTurn,
)
from nick_mimic.filters import (
    DropCounts,
    classify_first_user_message,
    is_human_nick,
    is_oompa_cwd,
    is_oompa_project_slug,
    is_oompa_worktree,
)
from nick_mimic.harnesses import claude as claude_adapter
from nick_mimic.harnesses import codex as codex_adapter
from nick_mimic.harnesses import gemini as gemini_adapter
from nick_mimic.redact import REDACTED, redact_secrets, render_tool_result_body

log = logging.getLogger(__name__)

VAL_FRACTION = 0.05
WORKTREE_SEGMENT_RE = re.compile(r"\.(ww\d+-i\d+|ws[0-9a-f]+-w\d+-i\d+)$")

MAX_MSG_CHARS = 4000          # per-message cap in context; oversized pastes get head+tail excerpt
MAX_CONTEXT_MSGS = 150        # keep the last N turns only
MAX_TARGET_CHARS = 20000      # sanity ceiling; long targets are kept and down-weighted by the auto-respond labeler


def _truncate_msg_content(text: str) -> str:
    if len(text) <= MAX_MSG_CHARS:
        return text
    head = text[:2000]
    tail = text[-1000:]
    return f"{head}\n…[TRUNCATED {len(text) - 3000} chars]…\n{tail}"


@dataclass
class LoaderStats:
    threads_seen: int = 0
    threads_kept: int = 0
    threads_dropped_cwd: int = 0
    threads_dropped_slug: int = 0
    threads_dropped_date: int = 0
    samples_emitted: int = 0
    drops: DropCounts = field(default_factory=DropCounts)


def project_slug_from_cwd(cwd: str) -> str:
    if not cwd:
        return "_unknown"
    prefixes = (f"{GIT_ROOT}/", f"{HOME}/")
    rest = cwd
    for p in prefixes:
        if cwd.startswith(p):
            rest = cwd[len(p):]
            break
    parts = [seg for seg in rest.split("/") if seg]
    if not parts:
        return "_unknown"
    if parts[0] == "conductor-workspaces" and len(parts) >= 2:
        return parts[1]
    first = parts[0]
    first = WORKTREE_SEGMENT_RE.sub("", first)
    if not first or first.startswith("."):
        return "_unknown"
    return first


def _thread_split(thread_id: str) -> str:
    h = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < VAL_FRACTION else "train"


def _parse_since(since: str | None) -> dt.datetime | None:
    if since is None:
        return None
    try:
        d = dt.datetime.fromisoformat(since)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d
    except ValueError:
        log.warning("bad --since value %r, ignoring", since)
        return None


def _thread_started(thread: Thread) -> dt.datetime | None:
    if not thread.started_at:
        return None
    try:
        s = thread.started_at.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d
    except (ValueError, AttributeError):
        return None


def _render_turn(turn: Turn) -> dict | None:
    if isinstance(turn, UserTurn):
        return {"role": "user", "content": _truncate_msg_content(redact_secrets(turn.text))}
    if isinstance(turn, AssistantTurn):
        return {"role": "assistant", "content": _truncate_msg_content(redact_secrets(turn.text))}
    if isinstance(turn, SystemTurn):
        return {"role": "system", "content": _truncate_msg_content(redact_secrets(turn.text))}
    if isinstance(turn, ToolCallTurn):
        try:
            shell = json.dumps(turn.input, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            shell = str(turn.input)
        return {
            "role": "assistant",
            "content": _truncate_msg_content(f"[tool_use: {turn.name}({redact_secrets(shell)})]"),
        }
    if isinstance(turn, ToolResultTurn):
        body = render_tool_result_body(turn.body, turn.is_error)
        return {"role": "user", "content": f"[tool_result] {body}"}
    return None


_RENDERED_TURN_TYPES = (UserTurn, AssistantTurn, SystemTurn, ToolCallTurn, ToolResultTurn)


def _build_context_up_to(turns: Iterable[Turn], idx: int) -> list[dict]:
    msgs: list[dict] = []
    for i, t in enumerate(turns):
        if i >= idx:
            break
        if not isinstance(t, _RENDERED_TURN_TYPES):
            continue
        rendered = _render_turn(t)
        if rendered is not None:
            msgs.append(rendered)
    if len(msgs) > MAX_CONTEXT_MSGS:
        msgs = msgs[-MAX_CONTEXT_MSGS:]
    return msgs


def _thread_cwd_ok(thread: Thread, slug: str, stats: LoaderStats) -> bool:
    if is_oompa_cwd(thread.cwd):
        stats.threads_dropped_cwd += 1
        return False
    if is_oompa_worktree(thread.cwd):
        stats.threads_dropped_cwd += 1
        return False
    if is_oompa_project_slug(Path(thread.source_path).parent.name) or is_oompa_project_slug(slug):
        stats.threads_dropped_slug += 1
        return False
    return True


def _first_user_text(turns: Iterable[Turn]) -> str | None:
    for t in turns:
        if isinstance(t, UserTurn):
            return t.text or ""
    return None


def samples_from_thread(thread: Thread, stats: LoaderStats) -> Iterator[Sample]:
    slug = project_slug_from_cwd(thread.cwd)
    if not _thread_cwd_ok(thread, slug, stats):
        return
    first = _first_user_text(thread.turns)
    if first is not None:
        tag = classify_first_user_message(first)
        if tag == "hide_test":
            stats.drops.hide_test_thread += 1
            return
        if tag == "ai_writing":
            stats.drops.ai_writing_thread += 1
            return
        if tag == "oompa":
            stats.drops.oompa_tag_thread += 1
            return
    turns = thread.turns
    for i, t in enumerate(turns):
        if not isinstance(t, UserTurn):
            continue
        if not is_human_nick(t, thread.cwd, stats.drops):
            continue
        target = redact_secrets(t.text).strip()
        if not target:
            continue
        if len(target) > MAX_TARGET_CHARS:
            stats.drops.too_long_target += 1
            continue
        ctx = _build_context_up_to(turns, i)
        yield Sample(
            context_messages=ctx,
            target=target,
            project=slug,
            harness=thread.harness,
            model=thread.model,
            cwd=thread.cwd,
            source_path=thread.source_path,
            thread_id=thread.thread_id,
        )


def iter_all_threads(since: dt.datetime | None, stats: LoaderStats) -> Iterator[Thread]:
    sources = (
        ("claude", claude_adapter.load_threads),
        ("codex", codex_adapter.load_threads),
        ("gemini", gemini_adapter.load_threads),
    )
    for name, loader in sources:
        try:
            for thread in loader():
                stats.threads_seen += 1
                started = _thread_started(thread)
                if since is not None and started is not None and started < since:
                    stats.threads_dropped_date += 1
                    continue
                yield thread
        except Exception as e:
            log.warning("loader %s failed: %s", name, e)


def build_samples(since: dt.datetime | None = None) -> tuple[list[Sample], list[Sample], LoaderStats]:
    stats = LoaderStats()
    train: list[Sample] = []
    val: list[Sample] = []
    for thread in iter_all_threads(since, stats):
        emitted = 0
        split = _thread_split(thread.thread_id)
        for sample in samples_from_thread(thread, stats):
            (val if split == "val" else train).append(sample)
            emitted += 1
        if emitted:
            stats.threads_kept += 1
            stats.samples_emitted += emitted
    return train, val, stats


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def log_stats(stats: LoaderStats) -> None:
    d = stats.drops
    log.info(
        "threads seen=%d kept=%d dropped_cwd=%d dropped_slug=%d dropped_date=%d samples=%d",
        stats.threads_seen,
        stats.threads_kept,
        stats.threads_dropped_cwd,
        stats.threads_dropped_slug,
        stats.threads_dropped_date,
        stats.samples_emitted,
    )
    log.info(
        "drops hide_test=%d ai_writing=%d oompa_tag=%d oompa_cwd=%d oompa_worktree=%d oompa_slug=%d hook_injected=%d too_short=%d too_long_target=%d sidechain=%d tool_result_user=%d",
        d.hide_test_thread,
        d.ai_writing_thread,
        d.oompa_tag_thread,
        d.oompa_cwd,
        d.oompa_worktree,
        d.oompa_project_slug,
        d.hook_injected,
        d.too_short,
        d.too_long_target,
        d.sidechain,
        d.tool_result_user,
    )
