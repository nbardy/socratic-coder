from __future__ import annotations

import random

from user_mimic.canonical import Sample
from user_mimic.template import JudgePrompt, render, render_for_judge


def _mk_sample(context: list[dict] | None = None, target: str = "finish the work") -> Sample:
    return Sample(
        context_messages=context if context is not None else [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "Working on it."},
        ],
        target=target,
        project="myrepo",
        harness="claude",
        model="claude-opus-4-6",
        cwd="/Users/nick/git/myrepo",
        source_path="/tmp/thread.jsonl",
        thread_id="t1",
    )


def test_render_returns_system_then_context_and_target():
    s = _mk_sample()
    messages, target = render(s, prefix="PREFIX", suffix="SUFFIX", max_context_tokens=4096)

    assert target == s.target
    assert messages[0]["role"] == "system"
    assert "PREFIX" in messages[0]["content"]
    assert "SUFFIX" in messages[0]["content"]
    assert "claude-opus-4-6" in messages[0]["content"]
    assert "harness claude" in messages[0]["content"]
    assert messages[1:] == s.context_messages


def test_render_skips_suffix_when_none():
    s = _mk_sample()
    messages, _ = render(s, prefix="PREFIX", suffix=None, max_context_tokens=4096)
    sys_content = messages[0]["content"]
    assert sys_content.startswith("PREFIX\n\nThis is a log")
    assert "SUFFIX" not in sys_content


def test_render_left_truncates_dropping_oldest_messages():
    context = [
        {"role": "user", "content": "oldest " + "x" * 2000},
        {"role": "assistant", "content": "middle " + "y" * 2000},
        {"role": "user", "content": "newest short"},
    ]
    s = _mk_sample(context=context)
    messages, _ = render(s, prefix="P", suffix=None, max_context_tokens=200)

    assert messages[0]["role"] == "system"
    remaining_contents = [m["content"] for m in messages[1:]]
    assert not any("oldest" in c for c in remaining_contents)
    assert any("newest short" in c for c in remaining_contents)


def test_render_truncates_single_oversized_message_from_left():
    huge = "START_OF_MESSAGE " + ("tok " * 5000) + "END_OF_MESSAGE"
    context = [{"role": "user", "content": huge}]
    s = _mk_sample(context=context)
    messages, _ = render(s, prefix="P", suffix=None, max_context_tokens=200)

    assert len(messages) == 2
    kept = messages[1]["content"]
    assert "END_OF_MESSAGE" in kept
    assert "START_OF_MESSAGE" not in kept
    assert "…[truncated]…" in kept


def test_render_system_is_always_kept():
    context = [{"role": "user", "content": "x " * 10000}]
    s = _mk_sample(context=context)
    messages, _ = render(s, prefix="KEEPME", suffix=None, max_context_tokens=50)
    assert messages[0]["role"] == "system"
    assert "KEEPME" in messages[0]["content"]


def test_render_for_judge_randomizes_and_reports_slot():
    s = _mk_sample()
    rng_a = random.Random(0)
    rng_b = random.Random(1)
    jp_a = render_for_judge(s, generated="gen reply", gold="gold reply", rng=rng_a)
    jp_b = render_for_judge(s, generated="gen reply", gold="gold reply", rng=rng_b)

    for jp in (jp_a, jp_b):
        assert isinstance(jp, JudgePrompt)
        assert jp.gold_is in ("A", "B")
        assert "gen reply" in jp.prompt
        assert "gold reply" in jp.prompt
        assert "A)" in jp.prompt and "B)" in jp.prompt


def test_render_for_judge_gold_slot_matches_content():
    s = _mk_sample()
    for seed in range(20):
        jp = render_for_judge(s, generated="G_GEN", gold="G_GOLD", rng=random.Random(seed))
        a_line = [ln for ln in jp.prompt.splitlines() if ln.startswith("A) ")][0]
        b_line = [ln for ln in jp.prompt.splitlines() if ln.startswith("B) ")][0]
        gold_line = a_line if jp.gold_is == "A" else b_line
        assert "G_GOLD" in gold_line
