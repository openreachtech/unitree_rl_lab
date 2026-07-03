#!/usr/bin/env python3
"""Aggregate TensorBoard scalar logs at fixed iteration intervals."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively aggregate TensorBoard scalar event files under a log directory."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("logs/rsl_rl"),
        help="Root directory to search recursively for TensorBoard event files.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Iteration interval to keep. For example, 100 keeps steps divisible by 100.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="events.out.tfevents.*",
        help="Glob pattern for TensorBoard event files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aggregate files. By default, existing files are skipped.",
    )
    return parser.parse_args()


def aggregate_event_file(event_file: Path, interval: int) -> tuple[dict[int, dict[str, float]], list[str]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # pyright: ignore[reportMissingImports]

    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    data: collections.defaultdict[int, dict[str, float]] = collections.defaultdict(dict)

    for tag in scalar_tags:
        for event in accumulator.Scalars(tag):
            if event.step % interval == 0:
                data[event.step][tag] = event.value

    columns = [tag for tag in scalar_tags if any(tag in values for values in data.values())]
    return dict(data), columns


def format_scalar(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(output_path: Path, data: dict[int, dict[str, float]], columns: list[str]) -> None:
    with output_path.open("w") as output_file:
        for column_index, column in enumerate(columns):
            if column_index > 0:
                output_file.write("\n")
            output_file.write(f"## {column}\n")
            for step in sorted(data):
                values = data[step]
                if column not in values:
                    continue
                output_file.write(f"{step} iteration: {format_scalar(values[column])}\n")


def experiment_name_from_path(path: Path) -> str:
    """Extract experiment name from logs/rsl_rl/{experiment_name}/... layout."""
    parts = path.resolve().parts
    if "rsl_rl" in parts:
        index = parts.index("rsl_rl")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def max_model_iteration(run_dir: Path) -> int | None:
    """Return the largest iteration number from ``model_*.pt`` files in a run directory."""
    iterations = []
    for path in run_dir.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt", path.name)
        if match:
            iterations.append(int(match.group(1)))
    return max(iterations) if iterations else None


def output_path_for(event_file: Path, interval: int) -> Path:
    experiment_name = experiment_name_from_path(event_file)
    max_iteration = max_model_iteration(event_file.parent)
    if max_iteration is not None:
        return event_file.with_name(f"{experiment_name}_model_{max_iteration}.md")
    return event_file.with_name(f"{experiment_name}.md")


def main() -> None:
    args = parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be a positive integer.")

    logdir = args.logdir.expanduser()
    if not logdir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {logdir}")
    if not logdir.is_dir():
        raise NotADirectoryError(f"Log path is not a directory: {logdir}")

    event_files = sorted(
        path for path in logdir.rglob(args.pattern) if path.is_file() and ".metrics_" not in path.name
    )
    print(f"Searching TensorBoard event files under: {logdir}")
    print(f"Found {len(event_files)} event file(s).")

    if not event_files:
        return

    processed = 0
    skipped = 0
    failed = 0

    for event_file in event_files:
        output_path = output_path_for(event_file, args.interval)
        print(f"\n[FILE] {event_file}")

        if output_path.exists() and not args.overwrite:
            print(f"  Skip: aggregate already exists: {output_path}")
            skipped += 1
            continue

        try:
            data, columns = aggregate_event_file(event_file, args.interval)
        except Exception as exc:  # noqa: BLE001 - continue processing other log files.
            print(f"  Failed: {exc}")
            failed += 1
            continue

        if not data or not columns:
            print(f"  Skip: no scalar data found at every {args.interval} iterations.")
            skipped += 1
            continue

        write_markdown(output_path, data, columns)
        processed += 1
        print(f"  Rows: {len(data)}")
        print(f"  Metrics: {len(columns)}")
        print(f"  Saved: {output_path}")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
