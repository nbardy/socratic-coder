from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from user_mimic.canonical import Sample
from user_mimic.infer import load_prompts, reply, slug_from_cwd


def _load_samples(path: Path, n: int) -> list[Sample]:
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
            if len(out) >= n:
                break
    return out


def _last_user_turn(sample: Sample) -> str:
    for m in reversed(sample.context_messages):
        if m.get("role") in ("user", "assistant"):
            c = m.get("content") or ""
            return c if len(c) <= 400 else c[:400] + "…"
    return ""


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples/val.jsonl")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--prompts-dir", default="prompts")
    args = p.parse_args(argv)

    samples = _load_samples(Path(args.samples), args.n)
    for i, s in enumerate(samples):
        project = s.project or slug_from_cwd(s.cwd)
        prefix, suffix = load_prompts(project, prompts_dir=args.prompts_dir)
        generated = reply(s, prefix=prefix, suffix=suffix, model=args.model)
        print(f"--- sample {i} project={project} harness={s.harness} ---")
        print(f"[context tail]\n{_last_user_turn(s)}")
        print(f"[gold]\n{s.target}")
        print(f"[generated]\n{generated}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
