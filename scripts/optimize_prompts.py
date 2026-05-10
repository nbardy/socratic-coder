from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nick_mimic.canonical import Sample
from nick_mimic.optimize import optimize_prefix, optimize_suffix


def _load_samples(path: Path) -> list[Sample]:
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
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prefix", "suffix"], required=True)
    p.add_argument("--project", default=None)
    p.add_argument("--train", default="data/samples/train.jsonl")
    p.add_argument("--val", default="data/samples/val.jsonl")
    p.add_argument("--max-cost-usd", type=float, default=None)
    p.add_argument("--max-rollouts", type=int, default=None)
    p.add_argument("--student-lm", default="openai/gpt-4.1-mini")
    p.add_argument("--judge-lm", default="openai/gpt-4.1-mini")
    p.add_argument("--prefix-path", default="prompts/prefix.txt")
    p.add_argument("--suffix-dir", default="prompts/suffix")
    args = p.parse_args(argv)

    train = _load_samples(Path(args.train))
    val = _load_samples(Path(args.val))

    if args.stage == "prefix":
        kw = {
            "student_lm": args.student_lm,
            "judge_lm": args.judge_lm,
            "out_path": args.prefix_path,
        }
        if args.max_cost_usd is not None:
            kw["max_cost_usd"] = args.max_cost_usd
        if args.max_rollouts is not None:
            kw["max_rollouts"] = args.max_rollouts
        final = optimize_prefix(train, val, **kw)
        print(f"wrote prefix ({len(final)} chars) to {args.prefix_path}")
        return 0

    if not args.project:
        print("--project is required for --stage suffix", file=sys.stderr)
        return 2
    prefix_text = Path(args.prefix_path).read_text(encoding="utf-8")
    kw = {
        "student_lm": args.student_lm,
        "judge_lm": args.judge_lm,
        "out_dir": args.suffix_dir,
    }
    if args.max_cost_usd is not None:
        kw["max_cost_usd"] = args.max_cost_usd
    if args.max_rollouts is not None:
        kw["max_rollouts"] = args.max_rollouts
    final = optimize_suffix(args.project, prefix_text, train, val, **kw)
    print(f"wrote suffix for {args.project} ({len(final)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
