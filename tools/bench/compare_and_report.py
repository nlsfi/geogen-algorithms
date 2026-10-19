#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class BenchmarkDiff:
    """Dataclass for diff data."""

    name: str
    base_mean_duration: float
    head_mean_duration: float
    delta_percentage: float
    status: str
    passes_threshold: bool


def format_duration(seconds: float) -> str:
    if seconds < 1e-6:  # noqa: PLR2004
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:  # noqa: PLR2004
        return f"{seconds * 1e6:.2f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def extract_means(report_path: Path) -> dict[str, float]:
    with report_path.open() as file:
        data = json.load(file)
    return {
        bench["name"]: bench["stats"]["mean"] for bench in data.get("benchmarks", [])
    }


def get_diffs(
    base_stats: dict[str, float],
    head_stats: dict[str, float],
    threshold_percentage: float,
) -> list[BenchmarkDiff]:
    diffs = []
    for name, head_mean in head_stats.items():
        if name not in base_stats:
            continue
        base_mean = base_stats[name]
        delta_percentage = (
            0.0 if base_mean == 0 else ((head_mean - base_mean) / base_mean) * 100.0
        )

        if delta_percentage < 0:
            status = "![](https://img.shields.io/badge/-Faster-238636)"
        else:
            status = "![](https://img.shields.io/badge/-Slower-da3633)"

        passes_threshold = abs(delta_percentage) >= threshold_percentage

        diffs.append(
            BenchmarkDiff(
                name=name,
                base_mean_duration=base_mean,
                head_mean_duration=head_mean,
                delta_percentage=delta_percentage,
                status=status,
                passes_threshold=passes_threshold,
            )
        )

    diffs.sort(key=lambda d: d.delta_percentage, reverse=True)
    return diffs


def render_markdown(
    all_diffs: Sequence[BenchmarkDiff],
    base_ref: str,
    head_ref: str,
    threshold_percentage: float,
) -> str:
    def _write_columns() -> str:
        return (
            "| Benchmark | Base Mean | PR Mean | Change | Status |\n"
            "| :--- | :--- | :--- | :--- | :--- |"
        )

    def _write_row(diff: BenchmarkDiff) -> str:
        sign = "+" if diff.delta_percentage > 0 else ""

        return (
            f"| `{diff.name}` | {format_duration(diff.base_mean_duration)} | "
            f"{format_duration(diff.head_mean_duration)} | "
            f"{sign}{diff.delta_percentage:.2f}% | {diff.status} |"
        )

    header = "## Benchmark Change Report\n\n"
    header += (
        f"Comparing `{head_ref}` against `{base_ref}` "
        "f(threshold at ±{threshold_percentage})\n"
    )

    passing_diffs = [diff for diff in all_diffs if diff.passes_threshold]
    non_passing_diffs = [diff for diff in all_diffs if not diff.passes_threshold]

    lines = [
        header,
    ]

    if not passing_diffs:
        lines.append(
            f"**No benchmarks changed by more than ±{threshold_percentage}%**.",
        )
    else:
        lines.append("Passes threshold:\n")
        lines.append(_write_columns())
        for diff in passing_diffs:
            lines.append(_write_row(diff))  # noqa: PERF401

    if non_passing_diffs:
        lines.append("\n<details>")
        lines.append("<summary><strong>Does not pass threshold</strong></summary>\n")
        lines.append(_write_columns())
        for diff in non_passing_diffs:
            lines.append(_write_row(diff))  # noqa: PERF401
        lines.append("</details>")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Base JSON report")
    parser.add_argument("--head", type=Path, required=True, help="Head JSON report")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Delta filter threshold percentage (default: 25.0)",
    )
    parser.add_argument("--base-ref", default="main", help="Base ref name")
    parser.add_argument("--head-ref", default="head", help="Head ref name")

    args = parser.parse_args(argv)

    base_stats = extract_means(args.base)
    head_stats = extract_means(args.head)
    diffs = get_diffs(base_stats, head_stats, args.threshold)
    report = render_markdown(diffs, args.base_ref, args.head_ref, args.threshold)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        sys.stdout.write(report + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
