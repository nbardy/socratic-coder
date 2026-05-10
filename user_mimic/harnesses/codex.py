from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from user_mimic.canonical import (
    AssistantTurn,
    SystemTurn,
    Thread,
    ToolCallTurn,
    ToolResultTurn,
    Turn,
    UserTurn,
)

log = logging.getLogger(__name__)

CODEX_ROOT = Path.home() / ".codex" / "sessions"

PREAMBLE_THRESHOLD = 2000
AGENTS_MARKER = "# AGENTS.md instructions"


def iter_session_files(root: Path = CODEX_ROOT) -> Iterator[Path]:
    if not root.exists():
        return
    for path in root.rglob("rollout-*.jsonl"):
        yield path


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("type")
                if t in ("input_text", "output_text"):
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return ""


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
        log.warning("codex: cannot read %s: %s", path, e)


def _is_agents_preamble(text: str, seen_user: bool) -> bool:
    if seen_user:
        return False
    if not text:
        return False
    if text.lstrip().startswith(AGENTS_MARKER):
        return True
    return len(text) > PREAMBLE_THRESHOLD


def load_thread(path: Path) -> Thread | None:
    cwd = ""
    model = ""
    started_at: str | None = None
    session_id: str | None = None
    turns: list[Turn] = []
    seen_real_user = False

    for rec in _iter_records(path):
        rtype = rec.get("type")
        payload = rec.get("payload") or {}

        if rtype == "session_meta":
            cwd = payload.get("cwd") or cwd
            started_at = payload.get("timestamp") or started_at
            session_id = payload.get("id") or session_id
            model = payload.get("model") or model
            continue

        if rtype != "response_item":
            continue

        ptype = payload.get("type")

        if ptype == "message":
            role = payload.get("role")
            text = _flatten_text(payload.get("content"))
            if role == "developer":
                turns.append(SystemTurn(text=text))
            elif role == "user":
                if _is_agents_preamble(text, seen_real_user):
                    continue
                seen_real_user = True
                turns.append(UserTurn(text=text))
            elif role == "assistant":
                turns.append(AssistantTurn(text=text, model=model or None))
            continue

        if ptype in ("function_call", "custom_tool_call"):
            raw_args = payload.get("arguments")
            try:
                parsed_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                )
                if not isinstance(parsed_args, dict):
                    parsed_args = {"_raw": parsed_args}
            except json.JSONDecodeError:
                parsed_args = {"_raw": raw_args}
            turns.append(
                ToolCallTurn(
                    name=payload.get("name", ""),
                    input=parsed_args,
                    call_id=payload.get("call_id"),
                )
            )
            continue

        if ptype in ("function_call_output", "custom_tool_call_output"):
            body = payload.get("output", "")
            if not isinstance(body, str):
                body = json.dumps(body, ensure_ascii=False)
            turns.append(
                ToolResultTurn(
                    call_id=payload.get("call_id"),
                    body=body,
                    is_error=False,
                )
            )
            continue

    if not turns:
        return None

    return Thread(
        harness="codex",
        model=model or "",
        cwd=cwd,
        started_at=started_at,
        source_path=str(path),
        thread_id=session_id or str(path),
        turns=tuple(turns),
    )


def load_threads(root: Path = CODEX_ROOT) -> Iterator[Thread]:
    for path in iter_session_files(root):
        th = load_thread(path)
        if th is not None:
            yield th
