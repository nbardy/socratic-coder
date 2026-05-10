"""DSPy LM that shells out to `claude -p` or `codex exec` per call.

Uses the user's existing CLI auth (subscription) instead of API keys.
~1-2s subprocess overhead per call vs HTTP API; fine for overnight runs.
"""
from __future__ import annotations

import json
import subprocess
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
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", text]
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


def _run_codex(text: str, model: str, timeout: int) -> tuple[str, dict]:
    cmd = ["codex", "exec", "--model", model, text]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"codex exec exit {r.returncode}: {r.stderr[:300]}")
    return r.stdout.strip(), {"prompt_tokens": 0, "completion_tokens": 0}


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
