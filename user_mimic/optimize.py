from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

import dspy

from user_mimic.canonical import Sample
from user_mimic import template as tpl
from user_mimic import metric as metric_mod

log = logging.getLogger(__name__)

SEED_PREFIX = (
    "You are responding as the user, steering an AI agent. "
    "Be terse. Ask for notes, completion, subagent fanout. "
    "Be pushy when they stall."
)

SEED_SUFFIX = "Project-specific steering."

MAX_CONTEXT_TOKENS = 8000

# USD per 1M tokens (input, output). Keep conservative; optimizer aborts when
# estimated spend exceeds max_cost_usd so slightly high is safer than low.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-5": (1.25, 10.00),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-4-5": (3.00, 15.00),
    "anthropic/claude-opus-4-5": (15.00, 75.00),
}


@dataclass(frozen=True)
class CLISpec:
    """Run via `claude -p` / `codex exec` subprocess. Subscription billing."""
    tool: str
    student_model: str
    reflection_model: str
    judge_model: str


@dataclass(frozen=True)
class APISpec:
    """Run via OpenAI/Anthropic HTTP API. Needs API key in env."""
    student_lm: str
    reflection_lm: str
    judge_lm: str


BackendSpec = Union[CLISpec, APISpec]


@dataclass
class Backends:
    student_lm: Any
    reflection_lm: Any
    judge: Any
    embedder: Any
    cost_model: str  # key into PRICE_TABLE; for CLI specs, a sentinel


class UserReply(dspy.Signature):
    """Read the thread and reply as the user would: terse, pattern-steering, pushy."""

    system: str = dspy.InputField()
    thread: str = dspy.InputField()
    reply: str = dspy.OutputField()


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostTracker:
    model: str
    max_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: Any = field(default_factory=threading.Lock)

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            if self.estimate_usd() > self.max_usd:
                raise BudgetExceeded(
                    f"cost cap hit: ${self.estimate_usd():.2f} > ${self.max_usd:.2f}"
                )

    def estimate_usd(self) -> float:
        p_in, p_out = PRICE_TABLE.get(self.model, (1.0, 3.0))
        return (self.prompt_tokens * p_in + self.completion_tokens * p_out) / 1_000_000.0


def _format_thread(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = (m.get("role") or "unknown").upper()
        content = m.get("content") or ""
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def _sample_to_example(sample: Sample, prefix: str, suffix: str | None, key: str) -> dspy.Example:
    messages, target = tpl.render(sample, prefix=prefix, suffix=suffix, max_context_tokens=MAX_CONTEXT_TOKENS)
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    rest = [m for m in messages if m.get("role") != "system"]
    system_text = system_msg["content"] if system_msg else ""
    thread_text = _format_thread(rest)
    ex = dspy.Example(system=system_text, thread=thread_text, reply=target, sample_key=key)
    return ex.with_inputs("system", "thread")


@dataclass
class CandidateLogger:
    path: Path
    stage: str
    project: str | None
    start: float

    def log(self, candidate: str, score: float, extra: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "stage": self.stage,
            "project": self.project,
            "candidate": candidate,
            "score": score,
            "wall_time": time.time() - self.start,
        }
        if extra:
            row.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_metric(
    samples_by_key: dict[str, Sample],
    judge: Any,
    embedder: Any,
    cost: CostTracker,
    candidate_log: CandidateLogger,
) -> Callable[..., float]:
    def _fn(example: dspy.Example, pred: dspy.Prediction, *_args, **_kwargs) -> float:
        if cost.estimate_usd() > cost.max_usd:
            raise BudgetExceeded(f"cost cap hit mid-metric: ${cost.estimate_usd():.2f}")
        generated = getattr(pred, "reply", "") or ""
        gold = example.reply
        key = getattr(example, "sample_key", None)
        sample = samples_by_key.get(key) if key else None
        context = sample.context_messages if sample is not None else []
        score = metric_mod.score(
            generated=generated,
            gold=gold,
            context=context,
            judge=judge,
            embedder=embedder,
        )
        combined = float(getattr(score, "combined", 0.0))
        candidate_log.log(candidate="<pred>", score=combined, extra={"gold": gold, "gen": generated})
        return combined

    return _fn


def _make_gepa(
    metric_fn: Callable[..., float],
    reflection_lm: Any,
    max_metric_calls: int | None,
    num_threads: int | None,
) -> tuple[Any, str]:
    gepa_cls = getattr(dspy, "GEPA", None)
    if gepa_cls is None:
        mipro_cls = getattr(dspy, "MIPROv2", None)
        if mipro_cls is None:
            raise RuntimeError("Neither dspy.GEPA nor dspy.MIPROv2 available")
        log.warning("dspy.GEPA not available; falling back to MIPROv2 (no reflection_lm)")
        return mipro_cls(metric=metric_fn, auto="light", num_threads=num_threads), "MIPROv2"

    common = {"metric": metric_fn, "reflection_lm": reflection_lm, "num_threads": num_threads}
    if max_metric_calls is None:
        return gepa_cls(auto="light", **common), "GEPA"
    return gepa_cls(max_metric_calls=max_metric_calls, **common), "GEPA"


def _build_lm(spec: BackendSpec, role: str) -> Any:
    """Build a DSPy LM for `role` ∈ {'student', 'reflection'}."""
    if isinstance(spec, CLISpec):
        from user_mimic.backends import CLIModelLM
        model = spec.student_model if role == "student" else spec.reflection_model
        return CLIModelLM(tool=spec.tool, cli_model=model)
    if isinstance(spec, APISpec):
        model = spec.student_lm if role == "student" else spec.reflection_lm
        if role == "reflection":
            return dspy.LM(model=model, temperature=1.0, max_tokens=32000)
        return dspy.LM(model=model)
    raise TypeError(f"unknown backend spec: {type(spec).__name__}")


def _build_judge(spec: BackendSpec) -> Any:
    if isinstance(spec, CLISpec):
        from user_mimic.backends import CLIJudge
        return CLIJudge(tool=spec.tool, cli_model=spec.judge_model)
    if isinstance(spec, APISpec):
        return metric_mod.OpenAIJudge.from_env()
    raise TypeError(f"unknown backend spec: {type(spec).__name__}")


def _build_embedder(kind: str) -> Any:
    if kind == "local":
        from user_mimic.backends import LocalEmbedder
        return LocalEmbedder()
    if kind == "openai":
        return metric_mod.OpenAIEmbedder.from_env()
    raise ValueError(f"unknown embedder kind: {kind!r}")


def _cost_model(spec: BackendSpec) -> str:
    if isinstance(spec, CLISpec):
        return f"cli/{spec.tool}/{spec.student_model}"
    return spec.student_lm


def _build_backends(spec: BackendSpec, embedder_kind: str) -> Backends:
    return Backends(
        student_lm=_build_lm(spec, "student"),
        reflection_lm=_build_lm(spec, "reflection"),
        judge=_build_judge(spec),
        embedder=_build_embedder(embedder_kind),
        cost_model=_cost_model(spec),
    )


def _wrap_lm_for_cost(lm: Any, cost: CostTracker) -> None:
    orig_call = lm.__call__

    def wrapped_call(*args, **kwargs):
        result = orig_call(*args, **kwargs)
        usage = (lm.history[-1].get("usage") if lm.history else None) or {}
        cost.add(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
        return result

    lm.__call__ = wrapped_call  # type: ignore[method-assign]


def _extract_prefix_from_program(program: Any, fallback: str) -> str:
    predictors = list(getattr(program, "predictors", lambda: [])()) or []
    for pred in predictors:
        sig = getattr(pred, "signature", None)
        if sig is None:
            continue
        instr = getattr(sig, "instructions", None)
        if isinstance(instr, str) and instr.strip():
            return instr
    return fallback


def _filter_by_project(samples: list[Sample], project: str) -> list[Sample]:
    return [s for s in samples if s.project == project]


def _sample_key(sample: Sample, tag: str, idx: int) -> str:
    return f"{tag}:{sample.thread_id}:{idx}"


def _build_examples(samples: list[Sample], prefix: str, suffix: str | None, tag: str) -> tuple[list[dspy.Example], dict[str, Sample]]:
    examples: list[dspy.Example] = []
    by_key: dict[str, Sample] = {}
    for idx, s in enumerate(samples):
        key = _sample_key(s, tag, idx)
        examples.append(_sample_to_example(s, prefix, suffix, key))
        by_key[key] = s
    return examples, by_key


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def optimize_prefix(
    train_samples: list[Sample],
    val_samples: list[Sample],
    *,
    backend: BackendSpec,
    embedder: str = "local",
    max_cost_usd: float = math.inf,
    max_metric_calls: int | None = None,
    num_threads: int | None = None,
    out_path: str = "prompts/prefix.txt",
) -> str:
    if not train_samples:
        raise ValueError("optimize_prefix: train_samples is empty")
    bks = _build_backends(backend, embedder)
    cost = CostTracker(model=bks.cost_model, max_usd=max_cost_usd)
    _wrap_lm_for_cost(bks.student_lm, cost)
    dspy.configure(lm=bks.student_lm)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    clog = CandidateLogger(
        path=logs_dir / f"optimizer_prefix_{_timestamp()}.jsonl",
        stage="prefix",
        project=None,
        start=time.time(),
    )

    train_ex, train_idx = _build_examples(train_samples, SEED_PREFIX, None, "train")
    val_ex, val_idx = _build_examples(val_samples, SEED_PREFIX, None, "val")
    by_key = {**train_idx, **val_idx}

    metric_fn = _build_metric(by_key, bks.judge, bks.embedder, cost, clog)

    program = dspy.Predict(UserReply)
    program.signature = program.signature.with_instructions(SEED_PREFIX)

    optimizer, name = _make_gepa(metric_fn, bks.reflection_lm, max_metric_calls, num_threads)
    log.info("prefix optimize: using %s, %d train, %d val", name, len(train_ex), len(val_ex))

    try:
        optimized = optimizer.compile(program, trainset=train_ex, valset=val_ex)
    except BudgetExceeded as e:
        log.error("prefix optimize aborted: %s", e)
        optimized = program

    final_prefix = _extract_prefix_from_program(optimized, SEED_PREFIX)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_prefix, encoding="utf-8")
    log.info("prefix optimize done: est $%.2f, wrote %s", cost.estimate_usd(), out)
    return final_prefix


def optimize_suffix(
    project: str,
    prefix: str,
    train_samples: list[Sample],
    val_samples: list[Sample],
    *,
    backend: BackendSpec,
    embedder: str = "local",
    max_cost_usd: float = math.inf,
    max_metric_calls: int | None = None,
    out_dir: str = "prompts/suffix",
) -> str:
    proj_train = _filter_by_project(train_samples, project)
    proj_val = _filter_by_project(val_samples, project)
    if len(proj_train) < 30:
        raise ValueError(
            f"optimize_suffix: project {project!r} has only {len(proj_train)} train samples; need >=30"
        )

    bks = _build_backends(backend, embedder)
    cost = CostTracker(model=bks.cost_model, max_usd=max_cost_usd)
    _wrap_lm_for_cost(bks.student_lm, cost)
    dspy.configure(lm=bks.student_lm)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    clog = CandidateLogger(
        path=logs_dir / f"optimizer_suffix_{project}_{_timestamp()}.jsonl",
        stage="suffix",
        project=project,
        start=time.time(),
    )

    # Suffix stage: prefix is frozen. Seed instructions = PREFIX + SEED_SUFFIX; GEPA evolves
    # the whole instruction string; we strip PREFIX back off afterward by prefix-match.
    seeded = f"{prefix}\n\n{SEED_SUFFIX}"

    train_ex, train_idx = _build_examples(proj_train, prefix, SEED_SUFFIX, f"train:{project}")
    val_ex, val_idx = _build_examples(proj_val, prefix, SEED_SUFFIX, f"val:{project}")
    by_key = {**train_idx, **val_idx}

    metric_fn = _build_metric(by_key, bks.judge, bks.embedder, cost, clog)

    program = dspy.Predict(UserReply)
    program.signature = program.signature.with_instructions(seeded)

    optimizer, name = _make_gepa(metric_fn, bks.reflection_lm, max_metric_calls, num_threads)
    log.info("suffix[%s] optimize: using %s, %d train, %d val", project, name, len(train_ex), len(val_ex))

    try:
        optimized = optimizer.compile(program, trainset=train_ex, valset=val_ex)
    except BudgetExceeded as e:
        log.error("suffix optimize aborted: %s", e)
        optimized = program

    final_full = _extract_prefix_from_program(optimized, seeded)
    final_suffix = final_full[len(prefix):].lstrip("\n") if final_full.startswith(prefix) else final_full

    out = Path(out_dir) / f"{project}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_suffix, encoding="utf-8")
    log.info("suffix[%s] optimize done: est $%.2f, wrote %s", project, cost.estimate_usd(), out)
    return final_suffix
