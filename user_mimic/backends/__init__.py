"""Pluggable LM / judge / embedder backends.

Two LM paths exist:
- API (OpenAI/Anthropic HTTP via DSPy's default `dspy.LM`) — needs API keys.
- CLI (`claude -p` or `codex exec` subprocess) — uses subscription auth, slower.

Embedder path:
- OpenAI HTTP (`metric.OpenAIEmbedder`) — needs API key.
- Local (sentence-transformers, CPU) — free, ~80MB model.
"""
from user_mimic.backends.cli_lm import CLIModelLM
from user_mimic.backends.cli_judge import CLIJudge
from user_mimic.backends.local_embedder import LocalEmbedder

__all__ = ["CLIModelLM", "CLIJudge", "LocalEmbedder"]
