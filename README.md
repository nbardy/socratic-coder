# Socratic Coder

A "you-bot" that learns to drive autonomous agents the same way you already do — by nudging. Train on your own steering messages from Claude Code, Codex, and Gemini CLI sessions, then have it speak in your voice ("did you take notes?", "spawn 5 subagents on X", "finish the work").

The name is the thesis: a good agent operator mostly asks pointed questions. Socratic Coder learns *your* questions.

## What it does

1. **Imports** your past conversations from Claude Code / Codex / Gemini CLI.
2. **Builds samples** of `(long agent thread) → (your short reply)`.
3. **Trains** one of two ways:
   - **Prompt optimization** (primary) — DSPy + GEPA evolves a system prompt over a cloud model.
   - **SFT** (fallback) — Qwen MoE LoRA fine-tune on a cloud GPU.

## Generate your own prompt

Three commands. No API keys — uses your existing Codex (or Claude Code) subscription via the CLI.

```bash
uv venv && uv sync --extra optimize

# 1. Build samples from your local agent-session history
uv run python -m user_mimic.loader --out data/samples/

# 2. Optimize a prompt for your steering style (~12 min wall, $0)
uv run python scripts/optimize_prompts.py --stage prefix \
  --lm-backend cli --cli-tool codex \
  --num-threads 4 \
  --train-limit 300 --val-limit 80 --max-metric-calls 120
```

Output lands at [`prompts/prefix.txt`](prompts/prefix.txt) — drop it in front of any model at inference time.

→ **[See an example prefix](prompts/prefix.txt)** evolved on Nick's `nbardy` corpus (4.4KB, valset score 0.42 → 0.44, captured patterns like "subagent fanout for parallel work", "demand verification not just edits", "strip large files from history over Git LFS").

### Other knobs

```bash
# Use Claude Code instead of Codex
... --cli-tool claude  # defaults to haiku for student/judge, sonnet for reflection

# Per-project tailoring (run after prefix is good)
uv run python scripts/optimize_prompts.py --stage suffix --project <slug>

# SFT path (cloud GPU required, see Track B internals below)
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

- **Metric** — weighted combo (0.6/0.3/0.1) of LLM-as-judge (pairwise A/B), embedding cosine, length sanity.
- **Optimizer** — DSPy `GEPA` (reflective prompt mutation).
- **Two-stage**: optimize `PREFIX` on the full corpus, then freeze it and optimize per-project `SUFFIX` for projects with ≥30 messages.
- **LM backends**: `--lm-backend cli` shells out to `claude -p` or `codex exec` (subscription auth, no API key); `--lm-backend api` uses `dspy.LM` (needs OpenAI/Anthropic key).
- **Embedder**: `--embedder local` uses `sentence-transformers/all-MiniLM-L6-v2` on CPU (free); `--embedder openai` uses `text-embedding-3-small` (needs key).
- **Parallelism**: `--num-threads 4` ≈ 2× speedup; 8+ risks subscription rate limits.
</details>

<details>
<summary><b>Track B — SFT</b></summary>

Unsloth LoRA on Qwen2.5-7B-Instruct (4-bit, 16k seq) or larger Qwen3-30B-A3B, cloud CUDA only. Loss masking: everything except the user's message is `label=-100`. ~$12/epoch on a runpod A100-80G.
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
