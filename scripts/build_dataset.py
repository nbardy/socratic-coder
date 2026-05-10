#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from user_mimic.loader import (
    _parse_since,
    build_samples,
    log_stats,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build user-bot training dataset.")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; default = all time")
    parser.add_argument(
        "--out",
        default="data/samples",
        help="output dir; writes train.jsonl and val.jsonl",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    since = _parse_since(args.since)
    out_dir = Path(args.out)
    train, val, stats = build_samples(since=since)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    log_stats(stats)
    logging.info("wrote %d train, %d val samples to %s", len(train), len(val), out_dir)


if __name__ == "__main__":
    main()
