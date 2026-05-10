from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

from user_mimic.canonical import Sample
from user_mimic import template as tpl

MAX_CONTEXT_TOKENS = 8000

_WW_RE = re.compile(r"\.(ww\d+-i\d+|ws[0-9a-f]+-w\d+-i\d+)$")
_GIT_ROOT = f"{Path(os.environ.get('USER_MIMIC_GIT_ROOT', Path.home() / 'git'))}/"
_UNKNOWN = "_unknown"


def slug_from_cwd(cwd: str) -> str:
    if not cwd:
        return _UNKNOWN
    path = cwd
    idx = path.find(_GIT_ROOT)
    if idx >= 0:
        path = path[idx + len(_GIT_ROOT):]
    else:
        path = path.lstrip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return _UNKNOWN
    first = parts[0]
    if first == "conductor-workspaces" and len(parts) >= 2:
        first = parts[1]
    first = _WW_RE.sub("", first)
    if not first:
        return _UNKNOWN
    return first


def load_prompts(project: str, prompts_dir: str = "prompts") -> tuple[str, str | None]:
    base = Path(prompts_dir)
    prefix_path = base / "prefix.txt"
    if not prefix_path.exists():
        raise FileNotFoundError(f"missing prefix file: {prefix_path}")
    prefix = prefix_path.read_text(encoding="utf-8")
    suffix_path = base / "suffix" / f"{project}.txt"
    if not suffix_path.exists():
        return prefix, None
    return prefix, suffix_path.read_text(encoding="utf-8")


def _reply_anthropic(messages: list[dict], model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    system_content = ""
    convo: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_content = m.get("content", "")
            continue
        convo.append({"role": m["role"], "content": m.get("content", "")})
    resp = client.messages.create(
        model=model,
        system=system_content,
        messages=cast(Any, convo),
        max_tokens=1024,
    )
    parts = getattr(resp, "content", []) or []
    out = []
    for p in parts:
        text = getattr(p, "text", None)
        if isinstance(text, str):
            out.append(text)
    return "".join(out).strip()


def _reply_openai(messages: list[dict], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    model_name = model.split("/", 1)[1] if model.startswith("openai/") else model
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()


def reply(
    sample: Sample,
    prefix: str,
    suffix: str | None,
    model: str = "claude-haiku-4-5",
) -> str:
    messages, _ = tpl.render(
        sample,
        prefix=prefix,
        suffix=suffix,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    if model.startswith("openai/") or model.startswith("gpt-"):
        return _reply_openai(messages, model)
    return _reply_anthropic(messages, model)
