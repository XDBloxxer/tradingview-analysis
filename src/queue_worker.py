#!/usr/bin/env python3
"""
Queue worker for the shared "tv-analysis-backfill" run queue.

Why this exists
----------------
GitHub Actions concurrency groups only ever hold ONE pending (queued) run per
group, no matter how many times you trigger a workflow that shares the group.
`cancel-in-progress: false` only protects the run that is *currently
executing* -- it does nothing for runs that are merely waiting in line. If you
queue run B, then queue run C before B has started, GitHub silently drops B
and keeps only C. So "queue up a bunch of actions" never worked reliably.

The fix is to stop asking GitHub to hold multiple pending runs at all. Instead:
  - Every trigger (schedule or manual) just appends a small JSON job
    description to data/run_queue.jsonl and commits it. This step has no
    concurrency restriction, so any number of these can happen back-to-back
    without anything being dropped.
  - A single "queue_runner" workflow (which DOES use the shared concurrency
    group) drains that file one job at a time, in order, looping until it's
    empty. Only one runner ever needs to be "in the queue" at once, because
    whichever runner actually gets to execute will pick up everything that
    has piled up in the file -- nothing is lost even if a redundant runner
    trigger gets dropped by GitHub's one-pending-run limit.

Commands
--------
peek:
    Print the next queued job as JSON (id first, for logging), or exit 1 with
    no output if the queue is empty. Used by the workflow to decide whether to
    keep looping.

run-next:
    Pop the single oldest job off the queue, execute it (daily_winners_main.py
    or daily_non_winners_main.py, with the same arguments the old inline shell
    logic used to build), then remove that job from the queue file. Re-reads
    the file immediately before writing back so that any jobs appended by a
    concurrent enqueue step (which can legitimately happen while this job is
    running) are preserved rather than clobbered.

    Exits with the subprocess's return code. The job is removed from the
    queue whether it succeeded or failed -- same as the old behavior, where a
    failed run just failed and wasn't auto-retried. Use the enqueue workflow
    again to retry.

enqueue-range:
    Like `enqueue`, but understands --chunk-days. If start_date/end_date are
    given along with a positive --chunk-days, the range is split into
    consecutive pieces of that many calendar days each (the last piece may be
    shorter) and one queue item is created per piece -- e.g. a 90-day
    backfill with --chunk-days 7 becomes ~13 separate queued jobs instead of
    one job trying to cover all 90 days in a single workflow run. Each piece
    still runs through the normal one-at-a-time queue, so nothing changes
    about ordering or the anti-repeat selection-count persistence -- it just
    means a single "queue this backfill" click produces many small jobs
    instead of one big one, which keeps any individual job well under
    GitHub's 6-hour job timeout and gives you visibility/retry granularity
    per chunk. If --chunk-days is 0/empty, or start_date/end_date aren't both
    set, this behaves exactly like a single `enqueue` call.
"""

import argparse
import json
import subprocess
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

QUEUE_PATH = Path("data/run_queue.jsonl")


def _read_lines():
    if not QUEUE_PATH.exists():
        return []
    items = []
    for line_no, raw in enumerate(QUEUE_PATH.read_text().splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            items.append(json.loads(raw))
        except json.JSONDecodeError:
            print(f"WARNING: skipping malformed queue line {line_no}: {raw!r}", file=sys.stderr)
    return items


def _write_lines(items):
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("w") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")


def _make_item(job_type, date_str, start_date, end_date, top_n, allow_append, queued_by):
    return {
        "id": uuid.uuid4().hex[:12],
        "job_type": job_type,
        "date": date_str or "",
        "start_date": start_date or "",
        "end_date": end_date or "",
        "top_n": top_n or "",
        "allow_append": allow_append or "false",
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "queued_by": queued_by or "",
    }


def cmd_enqueue(args):
    item = _make_item(
        args.job_type, args.date, args.start_date, args.end_date,
        args.top_n, args.allow_append, args.queued_by,
    )
    items = _read_lines()
    items.append(item)
    _write_lines(items)
    print(item["id"])


def _split_range(start_date, end_date, chunk_days):
    """Yield (chunk_start, chunk_end) date pairs of at most chunk_days
    calendar days each, covering [start_date, end_date] inclusive."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        start, end = end, start

    step = timedelta(days=chunk_days - 1)  # chunk_days=7 -> 7 days inclusive
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + step, end)
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def cmd_enqueue_range(args):
    chunk_days = int(args.chunk_days) if args.chunk_days else 0

    if args.start_date and args.end_date and chunk_days > 0:
        pieces = list(_split_range(args.start_date, args.end_date, chunk_days))
    elif args.start_date and args.end_date:
        pieces = [(args.start_date, args.end_date)]
    else:
        pieces = [(args.date, args.date)] if args.date else [("", "")]
        # Single-date (or no-date) job: keep start/end empty, use --date as-is.
        pieces = None

    items = _read_lines()
    new_ids = []

    if pieces is None:
        item = _make_item(
            args.job_type, args.date, "", "",
            args.top_n, args.allow_append, args.queued_by,
        )
        items.append(item)
        new_ids.append(item["id"])
    else:
        for chunk_start, chunk_end in pieces:
            item = _make_item(
                args.job_type, "", chunk_start, chunk_end,
                args.top_n, args.allow_append, args.queued_by,
            )
            items.append(item)
            new_ids.append(item["id"])

    _write_lines(items)
    for job_id in new_ids:
        print(job_id)
    print(f"Enqueued {len(new_ids)} job(s).", file=sys.stderr)


def cmd_peek(_args):
    items = _read_lines()
    if not items:
        return 1
    print(json.dumps(items[0]))
    return 0


def _build_args(item):
    job_type = item["job_type"]
    if job_type == "winners":
        script = "daily_winners_main.py"
        default_top_n = "15"
    elif job_type == "non_winners":
        script = "daily_non_winners_main.py"
        default_top_n = "60"
    else:
        raise ValueError(f"Unknown job_type: {job_type!r}")

    cli = [sys.executable, script, "--verbose", "--top-n", item.get("top_n") or default_top_n]

    if item.get("allow_append") == "true":
        cli.append("--allow-append")

    if item.get("start_date") and item.get("end_date"):
        cli += ["--start-date", item["start_date"], "--end-date", item["end_date"]]
    elif item.get("date"):
        cli += ["--date", item["date"]]

    return cli


def cmd_run_next(_args):
    items = _read_lines()
    if not items:
        print("Queue is empty, nothing to run.")
        return 0

    item = items[0]
    cli = _build_args(item)
    print(f"=== Running queued job {item['id']} ({item['job_type']}) ===")
    print(f"    queued_at={item.get('queued_at')} queued_by={item.get('queued_by')!r}")
    print(f"    command: {' '.join(cli)}")

    result = subprocess.run(cli)

    # Re-read fresh right before writing back, so anything appended by a
    # concurrent enqueue step while this job was running isn't lost.
    fresh_items = _read_lines()
    remaining = [i for i in fresh_items if i["id"] != item["id"]]
    _write_lines(remaining)

    print(f"=== Job {item['id']} finished with exit code {result.returncode}; "
          f"{len(remaining)} job(s) still queued ===")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_enqueue = sub.add_parser("enqueue", help="Append a single job to the queue")
    p_enqueue.add_argument("--job-type", required=True, choices=["winners", "non_winners"])
    p_enqueue.add_argument("--date", default="")
    p_enqueue.add_argument("--start-date", default="")
    p_enqueue.add_argument("--end-date", default="")
    p_enqueue.add_argument("--top-n", default="")
    p_enqueue.add_argument("--allow-append", default="false")
    p_enqueue.add_argument("--queued-by", default="")
    p_enqueue.set_defaults(func=cmd_enqueue)

    p_enqueue_range = sub.add_parser(
        "enqueue-range",
        help="Append a job, splitting start_date..end_date into chunk-days-sized pieces",
    )
    p_enqueue_range.add_argument("--job-type", required=True, choices=["winners", "non_winners"])
    p_enqueue_range.add_argument("--date", default="")
    p_enqueue_range.add_argument("--start-date", default="")
    p_enqueue_range.add_argument("--end-date", default="")
    p_enqueue_range.add_argument("--chunk-days", default="")
    p_enqueue_range.add_argument("--top-n", default="")
    p_enqueue_range.add_argument("--allow-append", default="false")
    p_enqueue_range.add_argument("--queued-by", default="")
    p_enqueue_range.set_defaults(func=cmd_enqueue_range)

    p_peek = sub.add_parser("peek", help="Print next job, exit 1 if queue empty")
    p_peek.set_defaults(func=cmd_peek)

    p_run_next = sub.add_parser("run-next", help="Pop and execute the next job")
    p_run_next.set_defaults(func=cmd_run_next)

    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
