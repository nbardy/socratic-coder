"""JudgeLM impl that calls a CLI tool directly (no DSPy roundtrip)."""
from __future__ import annotations

from user_mimic.backends.cli_lm import CLITool, _RUNNERS


class CLIJudge:
    def __init__(self, tool: CLITool, cli_model: str, *, timeout: int = 120):
        if tool not in _RUNNERS:
            raise ValueError(f"unknown CLI tool: {tool!r}")
        self._tool: CLITool = tool
        self._cli_model = cli_model
        self._timeout = timeout

    def compare(self, prompt: str) -> str:
        content, _ = _RUNNERS[self._tool](prompt, self._cli_model, self._timeout)
        return content
