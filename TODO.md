# TODO

## 0. Decide the obvious
- [ ] **Primary track**: prompt optimization (Track A). Fine-tune is fallback (Track B).
- [ ] Pick optimizer lib: DSPy+GEPA (leaning) vs TextGrad vs promptim.
- [ ] Pick cloud model for inference (Claude Sonnet? GPT-5? Both?).
- [ ] Decide: include tool_use/tool_result *shells* but drop content bodies? (recommended v1 — true for both tracks)
- [ ] Track B only — base model (Qwen3-30B-A3B vs smaller) and context length (8k/16k/32k).

## 1. Canonical types (`canonical.py`)
- [ ] `Thread = list[Turn]` with fields: `harness`, `model`, `cwd`, `started_at`, `source_path`.
- [ ] `Turn = User | Assistant | ToolCall | ToolResult | System` — sum type, one clean handler per variant in rendering.

## 2. Harness adapters (`harnesses/`)
- [ ] `claude.py` — parse `~/.claude/projects/<slug>/*.jsonl`. Map `type:user`/`type:assistant`; inside assistant messages unpack `content[]` into `text` / `thinking` / `tool_use`; inside user messages unpack `tool_result`. Skip `file-history-snapshot`.
- [ ] `codex.py` — parse `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`. Skip `session_meta` / `event_msg`; keep `response_item`. Distinguish the `developer` role (system-ish) from real user messages — the *first* human message often comes after a big developer preamble.
- [ ] `gemini.py` — probe `~/.gemini/history/<slug>/` further; file format still unknown (only `.project_root` markers surfaced in our scan).
- [ ] Skip sidechain / subagent threads (`isSidechain: true` in Claude) — those are agent-to-agent, not Nick.
- [ ] Skip hook-injected user messages (e.g. `[_HIDE_TEST_]`, `<system-reminder>`) — they're not Nick.

## 3. Identifying "Nick" messages
- [ ] Claude: `type:user` where `userType:"external"` AND content is plain string (not `tool_result`) AND not starting with system-reminder/hook markers.
- [ ] Codex: `role:"user"` where content is `input_text` AND not the `AGENTS.md` / developer preamble.
- [ ] Add a `is_human_nick(turn) -> bool` predicate per harness; unit-test on a handful of known sessions.

### 3a. Hidden-thread filter — use unleashd's canonical list
Source of truth: `~/git/unleashd/server/src/adapters/jsonl.ts` — `extractWorkerMetadata`. Three tags matched against the **first user message only** (matches unleashd semantics):
- `[_HIDE_TEST_]` — probe/validation runs: `^\s*(?:"|')?\s*\[_HIDE_TEST_\]\s*`
- `[ai-writing-tool]` — ai-writing-tool jobs: `^\s*(?:"|')?\s*\[ai-writing-tool\]\s*`
- `[oompa(:swarmId(:workerId)?)?]` — oompa workers: `^\s*(?:"|')?\s*\[oompa(?::([^:\]]+)(?::([^\]]+))?)?\]`
All three tolerate a leading quote and whitespace (prompts get wrapped in quotes by some launchers).

Complementary cwd-based drops (for threads whose tag was stripped upstream):
- `cwd` under `~/git/oompa` or `~/git/oompa_loompas`.
- `cwd` matches `\.(ww\d+-i\d+|ws[0-9a-f]+-w\d+-i\d+)(/|$)` — oompa-spawned worker worktrees inside other repos (e.g. `.ww3-i70`, `.wsf0be61ac-w8-i4`).
- Project slug contains `-private-tmp-claude-`.

Implemented in `nick_mimic/filters.py::classify_first_user_message` + thread-level check in `loader.py::samples_from_thread`. Per-signal drop counts logged.

## 4. Sample builder (`template.py`)
- [ ] Emit canonical sample dict: `{context_messages: [...], target: "<nick msg>", project: "<slug>", harness, model}`. Both tracks consume this.
- [ ] Truncate from the *left* (drop oldest context) so the target and system are always intact.
- [ ] Length budget configurable (cloud-model context for Track A, `max_seq_len` for Track B).
- [ ] Track A: render to provider-native chat format at optimize time.
- [ ] Track B: tokenize, set labels to `-100` on everything except target tokens, emit HF Arrow shards.

## 5. System prompt
- [ ] Template: `This is a log from model {model} in harness {harness}, cwd {cwd}. Respond as Nick.`
- [ ] Include repo name only (not full path) for generalization? Both variants — mix.

## 6. Redaction (`redact.py`) — **don't skip**
- [ ] Strip obvious secrets: `sk-ant-…`, `sk-…`, AWS keys, GitHub tokens, Bearer tokens, `.env` contents, private SSH keys.
- [ ] Strip email addresses except Nick's own (optional).
- [ ] Strip tool_result payloads by default (keep the call signature, drop body). Makes samples 10-100× smaller.
- [ ] Log per-sample redaction counts for audit.

## 7. Filtering / balance
- [ ] Drop Nick messages of length < 2 tokens ("ok", "y") — they'll dominate and teach nothing.
- [ ] Optionally drop trivial one-word confirmations above some frequency cap.
- [ ] De-dup near-identical Nick messages across threads (hash-based).
- [ ] Date filter: only include sessions from last N months so the model learns *current* Nick.

## 8. Splits & eval
- [ ] Hold out ~5% of threads (not messages) for eval — prevents leakage.
- [ ] Eval metric: loss on held-out Nick messages + a few qualitative generations.
- [ ] Write `eval.py` that samples completions given a real prior context.

## 9a. Track A — Metric (`metric.py`)
- [ ] Implement LLM-as-judge: pairwise "which reply is more Nick-like given this thread?" between (generated, gold). Use a cheap judge (Haiku / GPT-5-mini).
- [ ] Implement embedding cosine (sentence-transformers or cloud embed API) between generated and gold.
- [ ] Length sanity bonus/penalty — Nick is terse; reward being within 0.5×–2× gold length.
- [ ] Combined score = weighted sum. Start with weights (0.6 judge, 0.3 embed, 0.1 length) and tune.
- [ ] Sanity-check metric manually: does it rank obviously-Nick replies above generic ones?

## 9b. Track A — Optimizer (`optimize.py`)
- [ ] Wire DSPy `Signature`: inputs `{system, thread}` → outputs `{reply}`.
- [ ] Stage 1: optimize `PREFIX` with GEPA on all-project train split. Cap rollouts/budget.
- [ ] Stage 2: for each project with ≥ N (e.g. 30) Nick msgs, freeze PREFIX, optimize a short SUFFIX on that project's train split. Project split defined by `cwd` → repo-root slug.
- [ ] Hold out per-project val; report final metric on held-out test.
- [ ] Log every candidate prompt + score (GEPA will produce many) — the history is useful for manual review.
- [ ] Cost cap: hard ceiling on total USD per run; fail loudly if exceeded.

## 9. Track B — Trainer (`train.py`)
- [ ] Unsloth `FastLanguageModel.from_pretrained(...)` with 4-bit + LoRA.
- [ ] LoRA rank 16–32, alpha 32, target `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- [ ] `SFTTrainer` with `dataset_text_field=None` (we already tokenized) and `packing=False` (labels masking).
- [ ] Gradient checkpointing on, bf16, cosine lr schedule, warmup 3%, 1–2 epochs.
- [ ] Long-context: rope scaling if > 8k.
- [ ] Save LoRA adapter + merged FP16 both.

## 10. Cloud training
- [ ] `scripts/train_cloud.py` — spins up a runpod/lambdalabs/modal box, rsyncs `data/samples/`, runs `train.py`, syncs adapters back.
- [ ] Pin CUDA / torch / unsloth versions. Don't use `pip` — `uv` for local, but cloud image can use bundled pip if faster.

## 11. Inference / how do I actually use this?
- [ ] Decide the shape: standalone CLI that watches a live Claude Code jsonl and suggests Nick-style replies? A wrapper harness? A VS Code thing? — **pick one before v2.**
- [ ] Track A: `infer.py` loads `prompts/prefix.txt` + `prompts/suffix/<project>.txt` (if exists), prepends to context, calls cloud model.
- [ ] Project routing: `cwd` → repo-root slug → suffix file. Fall back to PREFIX-only for unknown projects.
- [ ] Track B: load LoRA adapter, same wrapper.

## Open questions for Nick
- [ ] Include thinking blocks in the context, or strip? (Leaning: strip — they're huge.)
- [ ] Train on Gemini data at all, given we don't yet know its on-disk schema?
- [ ] Include `sidechain` subagent threads? (Leaning: no — Nick isn't the user there.)
- [ ] Should system prompt also include the *current* Nick memory (`~/.claude/.../memory/*.md`)? Could be strong steering signal.
