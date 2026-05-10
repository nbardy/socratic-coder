from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from user_mimic.canonical import Sample


TRUNC_MARKER = "…[truncated]…"


@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


def _count_message_tokens(msg: dict) -> int:
    enc = _enc()
    return len(enc.encode(msg.get("content", ""))) + len(enc.encode(msg.get("role", ""))) + 4


def _count_total(messages: list[dict]) -> int:
    return sum(_count_message_tokens(m) for m in messages) + 2


def _build_system_prompt(sample: Sample, prefix: str, suffix: str | None) -> str:
    mid = f"\n\n{suffix}" if suffix is not None else ""
    tail = (
        f"\n\nThis is a log from model {sample.model} in harness {sample.harness}, "
        f"cwd {sample.cwd}. Respond as the user."
    )
    return f"{prefix}{mid}{tail}"


def _left_truncate_single(content: str, budget_tokens: int) -> str:
    enc = _enc()
    marker_ids = enc.encode(TRUNC_MARKER)
    ids = enc.encode(content)
    keep = max(0, budget_tokens - len(marker_ids) - 8)
    if keep <= 0:
        return TRUNC_MARKER
    return TRUNC_MARKER + enc.decode(ids[-keep:])


def render(
    sample: Sample,
    prefix: str,
    suffix: str | None,
    max_context_tokens: int,
) -> tuple[list[dict], str]:
    system_msg = {"role": "system", "content": _build_system_prompt(sample, prefix, suffix)}
    system_cost = _count_message_tokens(system_msg)

    context = list(sample.context_messages)
    messages = [system_msg] + context

    while _count_total(messages) > max_context_tokens and len(messages) > 1:
        if len(messages) == 2:
            only = messages[1]
            remaining = max_context_tokens - system_cost - 6
            only = {**only, "content": _left_truncate_single(only.get("content", ""), max(remaining, 1))}
            messages = [system_msg, only]
            break
        messages.pop(1)

    return messages, sample.target


@dataclass(frozen=True)
class JudgePrompt:
    prompt: str
    gold_is: str


def _format_context(context_messages: list[dict]) -> str:
    lines = []
    for m in context_messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def render_for_judge(
    sample: Sample,
    generated: str,
    gold: str,
    rng: random.Random | None = None,
) -> JudgePrompt:
    r = rng if rng is not None else random.Random()
    gold_is_a = r.random() < 0.5
    a_text, b_text = (gold, generated) if gold_is_a else (generated, gold)
    context = _format_context(sample.context_messages)
    prompt = (
        "Given this conversation thread:\n\n"
        f"{context}\n\n"
        "Which reply is more in the style of the original author?\n"
        f"A) {a_text}\n"
        f"B) {b_text}\n"
        "Answer only A or B."
    )
    return JudgePrompt(prompt=prompt, gold_is="A" if gold_is_a else "B")
