# Socratic Coder

A "you-bot" that learns to drive autonomous agents the same way you already do — by nudging. Train on your own steering messages from Claude Code, Codex, and Gemini CLI sessions, then have it speak in your voice ("did you take notes?", "spawn 5 subagents on X", "finish the work").

The name is the thesis: a good agent operator mostly asks pointed questions. Socratic Coder learns *your* questions.

## What it does

1. **Imports** your past conversations from Claude Code / Codex / Gemini CLI.
2. **Builds samples** of `(long agent thread) → (your short reply)`.
3. **Trains** one of two ways:
   - **Prompt optimization** (primary) — DSPy + GEPA evolves a system prompt over a cloud model.
   - **SFT** (fallback) — Qwen MoE LoRA fine-tune on a cloud GPU.

## Run

```bash
uv venv && uv sync

# 1. Import + build samples
uv run python -m user_mimic.loader --out data/samples/

# 2a. Prompt optimization
uv run python -m user_mimic.optimize --stage prefix
uv run python -m user_mimic.optimize --stage suffix --project <slug>

# 2b. SFT (cloud GPU required)
uv sync --extra train
uv run python -m user_mimic.train --data data/samples/train.jsonl
```

---

## Internals

<details>
<summary><b>Sample shape</b></summary>

For a thread with the user's messages at positions `i₁…iₖ`, emit `k` samples. Sample `j` = (context up to but not including msg `j`) → (msg `j`).

System prompt: `"This is a log from model {model} in harness {harness}, cwd {cwd}. Respond as the user."`

Size caps applied during sample build:
- Per-message context: 4000 chars (head 2000 + tail 1000, marker in the middle)
- Context turn count: last 150
- Tool-result body ≥ 200 chars → replaced with `[tool_result: N bytes]`
- Target: 20000-char sanity ceiling, kept and down-weighted by the auto-respond labeler
</details>

<details>
<summary><b>Auto-respond labeling</b></summary>

Every sample gets a `CAN_AUTO_RESPOND: 0–100%` score, computed by `scripts/label_samples.py` in three zones by target length:
- ≤150 chars → 90–100 (heuristic; short = generic steering)
- 150–800 chars → 50–90 (LLM grades how generic-bot-reproducible)
- \>800 chars → exponential decay 50→0 (long = specific, don't auto-respond)

Labels keyed by `{thread_id}:{sha1(target)[:12]}` in `data/samples/labels.jsonl`.
</details>

<details>
<summary><b>Track A — prompt optimization</b></summary>

- **Metric** — weighted combo of LLM-as-judge (pairwise), embedding cosine, length sanity.
- **Optimizer** — DSPy `GEPA` (reflective prompt mutation).
- **Two-stage**: optimize `PREFIX` on the full corpus, then freeze it and optimize per-project `SUFFIX` for projects with ≥N messages.
- **Budget** — start on Haiku/Sonnet; Opus only for final rounds.
</details>

<details>
<summary><b>Track B — SFT</b></summary>

Unsloth LoRA on Qwen3-30B-A3B (or smaller), long-context, cloud CUDA. Loss masking: everything except the user's message is `label=-100`.
</details>

<details>
<summary><b>Harness sources</b></summary>

| Harness | Location | Format |
|---|---|---|
| Claude Code | `~/.claude/projects/<slug>/*.jsonl` | `type:user/assistant`, tool_use / tool_result blocks |
| Codex | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | `response_item`, OpenAI-style `input_text` / tool calls |
| Gemini CLI | `~/.gemini/history/<project-slug>/` | format still being probed |
</details>

<details>
<summary><b>Portability</b></summary>

- Harness paths via `Path.home()`; no hardcoded usernames.
- Code root defaults to `~/git/` — override with `USER_MIMIC_GIT_ROOT`.
- Date cutoff is `dt.date.today()`.
- Mac/Linux, Python 3.11+, `uv`.
</details>
