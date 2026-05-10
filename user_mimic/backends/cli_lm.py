"""DSPy LM that shells out to `claude -p` or `codex exec` per call.

Uses the user's existing CLI auth (subscription) instead of API keys.
~2-5s subprocess overhead per call vs HTTP API; fine for overnight runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Literal

import dspy

CLITool = Literal["claude", "codex"]


def _flatten_messages(messages: list[dict] | None, prompt: str | None) -> str:
    if messages:
        return "\n\n".join(
            f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}"
            for m in messages
        )
    return prompt or ""


def _run_claude(text: str, model: str, timeout: int) -> tuple[str, dict]:
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd.extend(["--model", model])
    cmd.append(text)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude -p exit {r.returncode}: {r.stderr[:300]}")
    obj = json.loads(r.stdout)
    if obj.get("is_error"):
        raise RuntimeError(f"claude -p error: {str(obj.get('result'))[:300]}")
    usage = obj.get("usage") or {}
    return obj["result"], {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
    }


def _parse_codex_usage(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "turn.completed":
            u = obj.get("usage") or {}
            return {
                "prompt_tokens": int(u.get("input_tokens") or 0),
                "completion_tokens": int(u.get("output_tokens") or 0),
            }
    return {"prompt_tokens": 0, "completion_tokens": 0}


def _run_codex(text: str, model: str, timeout: int) -> tuple[str, dict]:
    # Read text from stdin (avoids argv length limits on large thread contexts).
    # `--ignore-user-config` skips the user's interactive Codex preferences (e.g.
    # xhigh reasoning) which would make every optimization call extremely slow.
    fd, last_path = tempfile.mkstemp(suffix=".txt", prefix="codex_last_")
    os.close(fd)
    try:
        cmd = [
            "codex", "exec",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--json",
            "-c", 'model_reasoning_effort="low"',
            "--output-last-message", last_path,
        ]
        if model:
            cmd.extend(["--model", model])
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"codex exec exit {r.returncode}: {r.stderr[:300]}")
        with open(last_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content, _parse_codex_usage(r.stdout)
    finally:
        try:
            os.unlink(last_path)
        except FileNotFoundError:
            pass


_RUNNERS = {"claude": _run_claude, "codex": _run_codex}


class CLIModelLM(dspy.BaseLM):
    """DSPy LM that calls a CLI tool subprocess for each request."""

    def __init__(self, tool: CLITool, cli_model: str, *, timeout: int = 300, **kwargs):
        if tool not in _RUNNERS:
            raise ValueError(f"unknown CLI tool: {tool!r}")
        super().__init__(model=f"{tool}/{cli_model}", **kwargs)
        self._tool: CLITool = tool
        self._cli_model = cli_model
        self._timeout = timeout

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **_kwargs):
        text = _flatten_messages(messages, prompt)
        content, usage = _RUNNERS[self._tool](text, self._cli_model, self._timeout)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                logprobs=None,
            )],
            usage=usage,
            model=self.model,
            _hidden_params={},
        )
