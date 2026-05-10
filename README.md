# Socratic Coder

A "you-bot" that learns to drive autonomous agents the same way you already do — by nudging. Train on your own steering messages from Claude Code, Codex, and Gemini CLI sessions, then have it speak in your voice ("did you take notes?", "spawn 5 subagents on X", "finish the work").

The name is the thesis: a good agent operator mostly asks pointed questions. Socratic Coder learns *your* questions.

> Internal package name: `nick_mimic` — this repo was bootstrapped on Nick's own (`nbardy`) message corpus. Everything below is corpus-agnostic; swap your harness paths and it trains on you.

## Idea

You barely write in your agent sessions — you just nudge. The nudges are patterned enough to learn. Each historical conversation on disk is (long agent thread) → (short user message).

## Two tracks

**Track A (primary): prompt optimization on top of a cloud model.**
Use DSPy + GEPA (or TextGrad) to evolve two prompts against Nick's message corpus:
1. A global `PREFIX` — optimized across **all** projects, encodes Nick's general steering style.
2. A per-project `SUFFIX` — optimized with `PREFIX` frozen, encodes project-specific steering.

Serve as `{PREFIX}\n{SUFFIX[project]}` in front of whatever cloud model (Claude / GPT).

**Track B (fallback): fine-tune a small local Qwen MoE.**
Same data pipeline. Use only if prompt optimization plateaus below usable quality, or if local/offline inference is required.

## Shared pipeline (both tracks use this)

1. **Loader** — walks each harness's on-disk session store, normalizes to a canonical thread type.
2. **Sample shape** — for a thread with Nick messages at positions `i₁, i₂, …, iₖ`, emit `k` samples. Sample `j` = (context up to but not including Nick msg `j`) → (Nick msg `j`). The context is the input; the Nick message is the gold target.
3. **Context prompt** — `"This is a log from model {model} in harness {harness}, cwd {cwd}. Respond as Nick."`

### Size caps

To keep samples trainable/sendable without losing real Nick behavior:

| Limit | Value | What happens when exceeded |
|---|---|---|
| Per-message content (context) | 4000 chars | Head 2000 + tail 1000 excerpt with `…[TRUNCATED N chars]…` marker. Nick's pasted terminal logs stay readable at both ends. |
| Context message count | 150 turns | Keep the last 150 only. Deep threads stay recent-biased. |
| Tool-result body | 200 chars | Bodies ≥ 200 chars replaced with `[tool_result: N bytes]`. Short statuses/errors pass through. |
| Target (Nick's message) | 20000 chars | Sanity ceiling only (beyond this is almost certainly a full file dump). Long targets are **kept** and down-weighted by the auto-respond labeler (see below) so the model learns they're the "don't auto-respond" class. |

## Auto-respond labeling

Output contract is two lines:
```
CAN_AUTO_RESPOND: <0-100>%
AUTO_RESPONSE: <nick's reply>
```

The score is pre-computed per training sample by `scripts/label_samples.py`. Three zones by target length:

| Zone | Target length | Score | How |
|---|---|---|---|
| short  | ≤ 150 chars | 90–100 (linear 100→90) | heuristic — almost all short Nick replies are generic steering ("keep going", "did you take notes?") |
| middle | 150–800 chars | 50–90 | synthetic label: `claude -p` reads the convo + Nick's reply and grades how generic-bot-reproducible it is |
| long   | > 800 chars | 50 → 0 (`50 · exp(-0.002 · (len − 800))`) | heuristic — long replies are specific, can't safely auto-respond |

Labels live in `data/samples/labels.jsonl`, keyed by `{thread_id}:{sha1(target)[:12]}`. Run `--dry-run` to see zone distribution and tune thresholds before calling the labeler.

## Portability

Hardcoded paths are avoided. All system-specific roots resolve at runtime:
- Harness data: `~/.claude/projects/`, `~/.codex/sessions/`, `~/.gemini/history/` via `Path.home()`.
- Your code root: defaults to `~/git/` — override with `NICK_MIMIC_GIT_ROOT=/path/to/your/code` env var. Used for the oompa-cwd drop list and the project slug extractor.
- Date cutoff: `dt.date.today()`, not a hardcoded constant.
- No pinned usernames. Should work on any Mac/Linux with Python 3.11+ and `uv`.

## Track A details

- **Metric** — we can't train without one. Candidates (combine with weights):
  - LLM-as-judge: "Which of these two replies is more Nick-like given this thread?" pairwise.
  - Embedding cosine between generated reply and true Nick reply.
  - Length sanity (penalize responses >2× or <½ the true reply length — Nick is terse).
- **Optimizer** — DSPy `GEPA` (reflective mutation on nat-lang prompts). Alt: TextGrad, promptim.
- **Two-stage search**:
  1. Split corpus into all-project train/val. Optimize `PREFIX` as a single DSPy `Signature` against the full mixed set.
  2. For each project with ≥N Nick messages, freeze `PREFIX`, optimize a short `SUFFIX` on that project's train split only.
  3. Projects with < N messages → no suffix, just use `PREFIX` alone.
- **Budget** — GEPA is reflection-heavy and expensive. Cap rollouts; start on Haiku/Sonnet as the judge+candidate model, promote to Opus only for final rounds.

## Track B details

- Unsloth LoRA on Qwen3-30B-A3B (or smaller), long-context, cloud CUDA.
- Sample shape adds loss-masking: everything except Nick's message is label=-100.

## Harness sources

| Harness | Location | Format |
|---|---|---|
| Claude Code | `~/.claude/projects/<slug>/*.jsonl` | one JSON object per line; `type: user/assistant`, rich tool_use / tool_result blocks, Read tool results embed file contents with line-number prefixes |
| Codex | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | `session_meta` / `response_item` / `event_msg`; OpenAI-style `input_text` / tool calls |
| Gemini CLI | `~/.gemini/history/<project-slug>/` | needs more probing — dirs contain `.project_root` markers |

## Design answers

- **Do Read tool calls contain file contents?** Claude: yes, full with line-number prefixes. Codex: need to check. For v1 we strip file contents from tool_result (keep the call, drop the payload) — the bot just needs to learn Nick's steering patterns, not reproduce file contents.
- **Why loss only on Nick's message?** The agent text is context, not a label. We want the LM head to specialize on producing Nick-style short steering.
- **Why small MoE?** Fast local inference; active params low; good for a lightweight sidecar/steerer.

## Layout (planned)

```
nick_mimic/
├── README.md
├── TODO.md
├── pyproject.toml          # uv-managed
├── nick_mimic/
│   ├── harnesses/          # one adapter per source (claude.py, codex.py, gemini.py)
│   ├── canonical.py        # canonical Thread / Message / ToolCall types
│   ├── loader.py           # walk + adapt + emit samples
│   ├── redact.py           # PII / secret stripping
│   ├── template.py         # render prompt + target for a sample
│   ├── metric.py           # Track A metric (judge / embed / length)
│   ├── optimize.py         # Track A: DSPy + GEPA two-stage optimizer
│   └── train.py            # Track B: unsloth + qwen + lora
├── prompts/
│   ├── prefix.txt          # optimized global PREFIX
│   └── suffix/<project>.txt
├── data/
│   ├── raw_index.jsonl     # pointers to source files (not copies)
│   └── samples/            # materialized samples (shared by both tracks)
└── scripts/
    ├── build_dataset.py
    ├── optimize_prompts.py # Track A
    └── train_cloud.py      # Track B
```

## Run (Track A)

```bash
uv venv && uv sync
uv run python -m nick_mimic.loader --out data/samples/
uv run python -m nick_mimic.optimize --stage prefix
uv run python -m nick_mimic.optimize --stage suffix --project <slug>
```
