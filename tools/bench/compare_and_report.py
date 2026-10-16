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


def get_meaningful_diffs(
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

        if abs(delta_percentage) < threshold_percentage:
            continue

        if delta_percentage < 0:
            status = "![](https://img.shields.io/badge/-Faster-238636)"
        else:
            status = "![](https://img.shields.io/badge/-Slower-da3633)"

        diffs.append(
            BenchmarkDiff(
                name=name,
                base_mean_duration=base_mean,
                head_mean_duration=head_mean,
                delta_percentage=delta_percentage,
                status=status,
            )
        )

    diffs.sort(key=lambda d: d.delta_percentage, reverse=True)
    return diffs


def render_markdown(
    diffs: Sequence[BenchmarkDiff],
    base_ref: str,
    head_ref: str,
    threshold_percentage: float,
) -> str:
    header = "## Benchmark Change Report\n"

    if not diffs:
        return (
            f"{header}\n"
            f"Comparing `{head_ref}` against `{base_ref}`:\n\n"
            f"**No benchmarks changed by more than ±{threshold_percentage}%**."
        )

    lines = [
        header,
        f"Comparing `{head_ref}` against `{base_ref}` "
        f"(Threshold ±{threshold_percentage})\n",
        "| Benchmark | Base Mean | PR Mean | Change | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for diff in diffs:
        sign = "+" if diff.delta_percentage > 0 else ""
        lines.append(
            f"| `{diff.name}` | {format_duration(diff.base_mean_duration)} | "
            f"{format_duration(diff.head_mean_duration)} | "
            f"{sign}{diff.delta_percentage:.2f}% | {diff.status} |"
        )
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
    diffs = get_meaningful_diffs(base_stats, head_stats, args.threshold)
    report = render_markdown(diffs, args.base_ref, args.head_ref, args.threshold)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        sys.stdout.write(report + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
