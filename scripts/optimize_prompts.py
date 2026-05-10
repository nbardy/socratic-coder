from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from user_mimic.canonical import Sample
from user_mimic.optimize import (
    APISpec,
    BackendSpec,
    CLISpec,
    optimize_prefix,
    optimize_suffix,
)


def _load_samples(path: Path, limit: int | None = None) -> list[Sample]:
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                Sample(
                    context_messages=rec["context_messages"],
                    target=rec["target"],
                    project=rec["project"],
                    harness=rec["harness"],
                    model=rec["model"],
                    cwd=rec["cwd"],
                    source_path=rec.get("source_path", ""),
                    thread_id=rec.get("thread_id", ""),
                )
            )
            if limit is not None and len(out) >= limit:
                break
    return out


def _build_backend(args: argparse.Namespace) -> BackendSpec:
    if args.lm_backend == "cli":
        return CLISpec(
            tool=args.cli_tool,
            student_model=args.student_model,
            reflection_model=args.reflection_model,
            judge_model=args.judge_model,
        )
    if args.lm_backend == "api":
        return APISpec(
            student_lm=args.student_model,
            reflection_lm=args.reflection_model,
            judge_lm=args.judge_model,
        )
    raise ValueError(f"unknown --lm-backend: {args.lm_backend!r}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prefix", "suffix"], required=True)
    p.add_argument("--project", default=None)
    p.add_argument("--train", default="data/samples/train.jsonl")
    p.add_argument("--val", default="data/samples/val.jsonl")
    p.add_argument("--train-limit", type=int, default=None, help="cap loaded train samples (smoke test)")
    p.add_argument("--val-limit", type=int, default=None, help="cap loaded val samples (smoke test)")
    p.add_argument("--lm-backend", choices=["cli", "api"], default="cli")
    p.add_argument("--cli-tool", choices=["claude", "codex"], default="claude")
    p.add_argument("--student-model", default="haiku", help="cli: 'haiku'/'sonnet'; api: 'openai/gpt-4.1-mini' etc.")
    p.add_argument("--reflection-model", default="sonnet")
    p.add_argument("--judge-model", default="haiku")
    p.add_argument("--embedder", choices=["local", "openai"], default="local")
    p.add_argument("--max-cost-usd", type=float, default=None, help="api mode only; CLI mode is subscription-billed")
    p.add_argument("--max-metric-calls", type=int, default=None, help="hard cap on GEPA metric calls (None → auto='light')")
    p.add_argument("--num-threads", type=int, default=None, help="parallel metric calls (None → DSPy default; CLI subprocess works well at 4-8)")
    p.add_argument("--prefix-path", default="prompts/prefix.txt")
    p.add_argument("--suffix-dir", default="prompts/suffix")
    args = p.parse_args(argv)

    train = _load_samples(Path(args.train), limit=args.train_limit)
    val = _load_samples(Path(args.val), limit=args.val_limit)
    backend = _build_backend(args)

    common: dict = {
        "backend": backend,
        "embedder": args.embedder,
        "max_metric_calls": args.max_metric_calls,
        "num_threads": args.num_threads,
    }
    if args.max_cost_usd is not None:
        common["max_cost_usd"] = args.max_cost_usd

    if args.stage == "prefix":
        final = optimize_prefix(train, val, out_path=args.prefix_path, **common)
        print(f"wrote prefix ({len(final)} chars) to {args.prefix_path}")
        return 0

    if not args.project:
        print("--project is required for --stage suffix", file=sys.stderr)
        return 2
    prefix_text = Path(args.prefix_path).read_text(encoding="utf-8")
    final = optimize_suffix(args.project, prefix_text, train, val, out_dir=args.suffix_dir, **common)
    print(f"wrote suffix for {args.project} ({len(final)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
