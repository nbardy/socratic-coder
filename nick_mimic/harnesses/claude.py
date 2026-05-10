from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from nick_mimic.canonical import (
    AssistantTurn,
    SystemTurn,
    Thread,
    ToolCallTurn,
    ToolResultTurn,
    Turn,
    UserTurn,
)

log = logging.getLogger(__name__)

CLAUDE_ROOT = Path.home() / ".claude" / "projects"

SKIP_TYPES = {
    "file-history-snapshot",
    "queue-operation",
    "last-prompt",
    "permission-mode",
    "summary",
}


def iter_session_files(root: Path = CLAUDE_ROOT) -> Iterator[Path]:
    if not root.exists():
        return
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        for path in project_dir.rglob("*.jsonl"):
            if "subagents" in path.parts:
                continue
            yield path


def _parse_user_content(content: Any) -> Turn | None:
    if isinstance(content, str):
        return UserTurn(text=content)
    if isinstance(content, list):
        texts: list[str] = []
        tool_results: list[ToolResultTurn] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                texts.append(item.get("text", ""))
            elif t == "tool_result":
                body = item.get("content", "")
                if isinstance(body, list):
                    body = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in body
                    )
                elif not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False)
                tool_results.append(
                    ToolResultTurn(
                        call_id=item.get("tool_use_id"),
                        body=body,
                        is_error=bool(item.get("is_error")),
                    )
                )
        if tool_results and not texts:
            return tool_results[0] if len(tool_results) == 1 else _Bundle(tool_results)
        if texts and not tool_results:
            return UserTurn(text="\n".join(texts))
        if texts or tool_results:
            return _Bundle([UserTurn(text="\n".join(texts))] + list(tool_results))
    return None


class _Bundle:
    __slots__ = ("turns",)

    def __init__(self, turns: list[Turn]):
        self.turns = turns


def _parse_assistant_content(content: Any, model: str | None) -> list[Turn]:
    out: list[Turn] = []
    if not isinstance(content, list):
        return out
    texts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            texts.append(item.get("text", ""))
        elif t == "thinking":
            continue
        elif t == "tool_use":
            if texts:
                out.append(AssistantTurn(text="\n".join(texts), model=model))
                texts = []
            out.append(
                ToolCallTurn(
                    name=item.get("name", ""),
                    input=item.get("input") or {},
                    call_id=item.get("id"),
                )
            )
    if texts:
        out.append(AssistantTurn(text="\n".join(texts), model=model))
    return out


def _iter_records(path: Path) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        log.warning("claude: cannot read %s: %s", path, e)


def load_thread(path: Path) -> Thread | None:
    cwd = ""
    model = ""
    started_at: str | None = None
    session_id: str | None = None
    turns: list[Turn] = []
    saw_any = False

    for rec in _iter_records(path):
        rtype = rec.get("type")
        if rtype in SKIP_TYPES:
            continue
        if rec.get("isSidechain"):
            continue
        cwd = rec.get("cwd") or cwd
        session_id = rec.get("sessionId") or session_id
        if not started_at:
            started_at = rec.get("timestamp")

        if rtype == "user":
            msg = rec.get("message") or {}
            parsed = _parse_user_content(msg.get("content"))
            if parsed is None:
                continue
            saw_any = True
            if isinstance(parsed, _Bundle):
                for t in parsed.turns:
                    if isinstance(t, UserTurn):
                        turns.append(
                            UserTurn(
                                text=t.text,
                                is_sidechain=False,
                                source_uuid=rec.get("uuid"),
                            )
                        )
                    else:
                        turns.append(t)
            elif isinstance(parsed, UserTurn):
                turns.append(
                    UserTurn(
                        text=parsed.text,
                        is_sidechain=False,
                        source_uuid=rec.get("uuid"),
                    )
                )
            else:
                turns.append(parsed)
        elif rtype == "assistant":
            msg = rec.get("message") or {}
            amodel = msg.get("model") or model
            model = amodel or model
            for t in _parse_assistant_content(msg.get("content"), amodel):
                turns.append(t)
            saw_any = True
        elif rtype == "system":
            content = rec.get("content")
            if isinstance(content, str):
                turns.append(SystemTurn(text=content))
        else:
            continue

    if not saw_any:
        return None

    return Thread(
        harness="claude",
        model=model or "",
        cwd=cwd,
        started_at=started_at,
        source_path=str(path),
        thread_id=session_id or str(path),
        turns=tuple(turns),
    )


def load_threads(root: Path = CLAUDE_ROOT) -> Iterator[Thread]:
    for path in iter_session_files(root):
        th = load_thread(path)
        if th is not None:
            yield th
