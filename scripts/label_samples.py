#!/usr/bin/env python3
"""Label training samples with CAN_AUTO_RESPOND scores.

Three zones by target (user-reply) length:
  short  (len <= LOW):          deterministic 100 -> 90 (linear)
  middle (LOW < len <= HIGH):   synthetic label via `claude -p`, range 50-90
  long   (len > HIGH):          deterministic 50 -> 0 (exponential decay)

Run with --dry-run first to tune the thresholds against the real
distribution. The labeler is serialized on purpose: it only emits ~3
tokens per call, so parallelism isn't worth the rate-limit risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

LOW_THRESHOLD = 150      # <= this many target chars -> "short" zone
HIGH_THRESHOLD = 800     # > this many target chars -> "long" zone
DECAY_SCALE = 0.002      # long-zone exp decay: 50% at HIGH, ~5% at HIGH+1150
CONTEXT_TURNS = 30       # recent convo turns shown to the labeler
CONTEXT_MSG_MAX = 1000   # per-message cap in labeler prompt
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SCORE_RE = re.compile(r"CAN_AUTO_RESPOND:\s*(\d{1,3})")


def sample_key(thread_id: str, target: str) -> str:
    h = hashlib.sha1(target.encode("utf-8")).hexdigest()[:12]
    return f"{thread_id}:{h}"


def zone_of(length: int) -> str:
    if length <= LOW_THRESHOLD:
        return "short"
    if length <= HIGH_THRESHOLD:
        return "middle"
    return "long"


def short_score(length: int) -> int:
    frac = length / LOW_THRESHOLD if LOW_THRESHOLD else 0.0
    return int(round(100 - 10 * frac))


def long_score(length: int) -> int:
    excess = length - HIGH_THRESHOLD
    return max(0, int(round(50 * math.exp(-DECAY_SCALE * excess))))


def _iter_jsonl(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        if not p.exists():
            logging.warning("samples file missing: %s", p)
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def render_convo(sample: dict) -> str:
    lines: list[str] = []
    for m in sample["context_messages"][-CONTEXT_TURNS:]:
        role = m.get("role", "?")
        content = m.get("content", "")
        if len(content) > CONTEXT_MSG_MAX:
            head = CONTEXT_MSG_MAX * 2 // 3
            tail = CONTEXT_MSG_MAX - head
            content = f"{content[:head]}\n…[truncated]…\n{content[-tail:]}"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


LABELER_PROMPT = """\
You are labeling how much a reply is "generic steering" (a simple bot
could have produced it) vs. "specific technical feedback" (required a
human who understood the situation).

Conversation (recent turns):
{convo}

the user's actual reply:
{reply}

Generic bot replies look like: "keep going", "try 6 more things",
"spawn 5 subagents on X", "did you take notes?", "ok", "no", "why?",
"looks good", "fix it", "finish the work". Anything that names specific
code, decisions, file paths, or contains real reasoning is NOT generic.

Rate CAN_AUTO_RESPOND in range 50-90 inclusive:
  90 = fully generic steering; interchangeable with the bot replies above
  70 = mostly generic, minor situational tweak
  50 = mostly specific; real technical content

Output EXACTLY one line and nothing else:
CAN_AUTO_RESPOND: <number>%
"""


def _claude_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDECODE_SESSION_ID", None)
    return env


def run_claude(prompt: str, model: str, timeout: float = 60.0) -> int | None:
    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("claude CLI not found on PATH")
    try:
        res = subprocess.run(
            [claude, "-p", prompt, "--model", model],
            capture_output=True, text=True, timeout=timeout,
            check=False, env=_claude_env(),
        )
    except subprocess.TimeoutExpired:
        return None
    m = SCORE_RE.search(res.stdout or "")
    if not m:
        return None
    return max(50, min(90, int(m.group(1))))


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def _load_existing_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    seen: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seen[rec["key"]] = rec
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default="data/samples")
    parser.add_argument("--labels-out", default="data/samples/labels.jsonl")
    parser.add_argument("--model", default=CLAUDE_MODEL)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print zone distribution only; do not call claude.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap middle-zone claude calls (for testing).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    samples_dir = Path(args.samples_dir)
    labels_path = Path(args.labels_out)
    sources = [samples_dir / "train.jsonl", samples_dir / "val.jsonl"]

    seen = _load_existing_labels(labels_path)

    buckets: dict[str, list[tuple[str, dict]]] = {"short": [], "middle": [], "long": []}
    for s in _iter_jsonl(sources):
        key = sample_key(s["thread_id"], s["target"])
        buckets[zone_of(len(s["target"]))].append((key, s))

    total = sum(len(v) for v in buckets.values())
    short_n, middle_n, long_n = len(buckets["short"]), len(buckets["middle"]), len(buckets["long"])

    print("\n=== label_samples thresholds ===")
    print(f"  LOW_THRESHOLD  = {LOW_THRESHOLD}  chars")
    print(f"  HIGH_THRESHOLD = {HIGH_THRESHOLD} chars")
    print(f"  DECAY_SCALE    = {DECAY_SCALE}")
    print(f"  model          = {args.model}")

    print("\n=== sample zone distribution ===")
    print(f"  short  (len <= {LOW_THRESHOLD}):            {short_n:>7,}  ({_pct(short_n, total)})")
    print(f"  middle ({LOW_THRESHOLD} < len <= {HIGH_THRESHOLD}):   {middle_n:>7,}  ({_pct(middle_n, total)})")
    print(f"  long   (len >  {HIGH_THRESHOLD}):            {long_n:>7,}  ({_pct(long_n, total)})")
    print(f"  total:                           {total:>7,}")
    print(f"  already labeled:                 {len(seen):>7,}")

    if total and long_n:
        lengths = [len(s["target"]) for _, s in buckets["long"]]
        print(f"\n=== long-zone length spot-check ===")
        print(f"  max length: {max(lengths):,}")
        print(f"  long_score(HIGH+100)  = {long_score(HIGH_THRESHOLD + 100)}")
        print(f"  long_score(HIGH+500)  = {long_score(HIGH_THRESHOLD + 500)}")
        print(f"  long_score(HIGH+1500) = {long_score(HIGH_THRESHOLD + 1500)}")
        print(f"  long_score(HIGH+3000) = {long_score(HIGH_THRESHOLD + 3000)}")

    if args.dry_run:
        print("\n[dry-run] not calling claude; exiting.")
        return

    labels_path.parent.mkdir(parents=True, exist_ok=True)

    with labels_path.open("a", encoding="utf-8") as out:
        def emit(key: str, score: int, method: str, sample: dict) -> None:
            if key in seen:
                return
            rec = {
                "key": key,
                "score": score,
                "method": method,
                "target_len": len(sample["target"]),
                "thread_id": sample["thread_id"],
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            seen[key] = rec

        for key, s in buckets["short"]:
            emit(key, short_score(len(s["target"])), "heuristic_short", s)
        for key, s in buckets["long"]:
            emit(key, long_score(len(s["target"])), "heuristic_long", s)

        middle = buckets["middle"]
        if args.limit is not None:
            middle = middle[: args.limit]
        for i, (key, s) in enumerate(middle):
            if key in seen:
                continue
            prompt = LABELER_PROMPT.format(
                convo=render_convo(s),
                reply=s["target"],
            )
            logging.info(
                "middle %d/%d key=%s len=%d",
                i + 1, len(middle), key, len(s["target"]),
            )
            score = run_claude(prompt, args.model)
            if score is None:
                logging.warning("parse/timeout failure on %s; skipping", key)
                continue
            emit(key, score, f"claude:{args.model}", s)


if __name__ == "__main__":
    main()
