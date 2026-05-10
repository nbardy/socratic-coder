#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

from nick_mimic.loader import _parse_since, build_samples


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def _row(label: str, n: int, total: int) -> str:
    return f"  {label:<32}{n:>8,}  ({_pct(n, total)})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print nick_mimic corpus metrics.")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; default = all time")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    since = _parse_since(args.since)
    train, val, stats = build_samples(since=since)

    d = stats.drops
    threads_total = stats.threads_seen
    samples_total = stats.samples_emitted + d.too_short + d.too_long_target + d.hook_injected
    hidden_threads = d.hide_test_thread + d.ai_writing_thread + d.oompa_tag_thread

    print(f"\n=== nick_mimic corpus metrics (since={args.since or 'all time'}) ===\n")
    print(f"Total conversations seen:        {threads_total:>8,}")
    print(f"Total conversations kept:        {stats.threads_kept:>8,}  ({_pct(stats.threads_kept, threads_total)})")
    print(f"Total response data points:      {stats.samples_emitted:>8,}")
    print(f"  train/val split:               {len(train):>8,} / {len(val):,}")

    print("\nThread-level drops:")
    print(_row("oompa cwd / worktree", stats.threads_dropped_cwd, threads_total))
    print(_row("date cutoff", stats.threads_dropped_date, threads_total))
    print(_row("[_HIDE_TEST_] tag", d.hide_test_thread, threads_total))
    print(_row("[ai-writing-tool] tag", d.ai_writing_thread, threads_total))
    print(_row("[oompa:...] tag", d.oompa_tag_thread, threads_total))
    print(_row("all hidden tags combined", hidden_threads, threads_total))

    print(f"\nSample-level drops (within kept threads, of {samples_total:,} candidates):")
    print(_row("long paste (target > 4000 chars)", d.too_long_target, samples_total))
    print(_row("too-short (< 2 words)", d.too_short, samples_total))
    print(_row("hook-injected reminders", d.hook_injected, samples_total))
    print()


if __name__ == "__main__":
    main()
