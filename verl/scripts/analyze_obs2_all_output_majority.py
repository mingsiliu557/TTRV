#!/usr/bin/env python3
"""Recompute Obs2 entropy bins without filtering invalid rollouts.

The input is the question-level CSV produced by
``analyze_diagnostic_stage_b_available.py``.  Its ``official_all_*`` columns
were computed from all 32 parser outputs, including invalid parser outputs.
This script bins that all-output entropy and scores an invalid majority as
wrong. Invalid strings are not filtered or remapped, matching the stored
official all-output Counter.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = (
    "mme20",
    "ai2d20",
    "seed20",
    "mathvista20_hq",
    "mathverse20_fixed",
    "realworldqa20_hq",
)
Z_95 = 1.959963984540054
POOLED_Q33 = 0.4656663294644227
POOLED_Q67 = 0.7540931063268035


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def entropy_bin(value: float, q33: float, q67: float) -> str:
    if value <= q33:
        return "low"
    if value <= q67:
        return "mid"
    return "high"


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    z2 = Z_95 * Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    spread = Z_95 * math.sqrt(
        (proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            benchmark = str(raw["benchmark"])
            if benchmark not in BENCHMARKS:
                continue
            majority_is_valid = int(raw["official_all_majority_is_valid"])
            valid_correct_text = str(raw.get("valid_majority_correct") or "").strip()
            if majority_is_valid and not valid_correct_text:
                raise ValueError(
                    f"{benchmark}/{raw['question_id']}: valid all-output majority has no correctness"
                )
            rows.append(
                {
                    "benchmark": benchmark,
                    "question_id": str(raw["question_id"]),
                    "entropy": float(raw["official_all_normalized_entropy"]),
                    "majority_answer": str(raw["official_all_majority_answer"]),
                    "majority_count": int(raw["official_all_majority_count"]),
                    "majority_is_invalid": 1 - majority_is_valid,
                    "majority_correct": int(
                        majority_is_valid and int(float(valid_correct_text)) == 1
                    ),
                }
            )
    return rows


def summarize_scope(scope: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entropy_values = [float(row["entropy"]) for row in rows]
    q33 = quantile(entropy_values, 1.0 / 3.0)
    q67 = quantile(entropy_values, 2.0 / 3.0)
    output: list[dict[str, Any]] = []
    for label in ("low", "mid", "high"):
        bucket = [row for row in rows if entropy_bin(float(row["entropy"]), q33, q67) == label]
        correct = sum(int(row["majority_correct"]) for row in bucket)
        invalid = sum(int(row["majority_is_invalid"]) for row in bucket)
        low, high = wilson_interval(correct, len(bucket))
        output.append(
            {
                "benchmark": scope,
                "h_bin": label,
                "q33": q33,
                "q67": q67,
                "h_mean": mean(float(row["entropy"]) for row in bucket),
                "majority_correct_successes": correct,
                "majority_correct_n": len(bucket),
                "majority_correct_rate": correct / len(bucket) if bucket else None,
                "majority_correct_ci95_low": low,
                "majority_correct_ci95_high": high,
                "invalid_majority_count": invalid,
                "invalid_majority_rate": invalid / len(bucket) if bucket else None,
            }
        )
    total_correct = sum(int(row["majority_correct"]) for row in rows)
    total_invalid = sum(int(row["majority_is_invalid"]) for row in rows)
    if sum(int(row["majority_correct_n"]) for row in output) != len(rows):
        raise AssertionError(f"{scope}: entropy bins do not partition all questions")
    if sum(int(row["majority_correct_successes"]) for row in output) != total_correct:
        raise AssertionError(f"{scope}: binned correct count does not match total")
    if sum(int(row["invalid_majority_count"]) for row in output) != total_invalid:
        raise AssertionError(f"{scope}: binned invalid-majority count does not match total")
    summary = {
        "benchmark": scope,
        "n": len(rows),
        "q33": q33,
        "q67": q67,
        "majority_correct_successes": total_correct,
        "majority_correct_rate": total_correct / len(rows),
        "invalid_majority_count": total_invalid,
        "invalid_majority_rate": total_invalid / len(rows),
    }
    return output, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_rate(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def write_markdown(path: Path, bin_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    labels = {
        "mme20": "MME20",
        "ai2d20": "AI2D20",
        "seed20": "SEED20",
        "mathvista20_hq": "MathVista20-HQ",
        "mathverse20_fixed": "MathVerse20-fixed",
        "realworldqa20_hq": "RealWorldQA20-HQ",
        "MCQ_MICRO": "MCQ micro",
    }
    by_scope = {(row["benchmark"], row["h_bin"]): row for row in bin_rows}
    lines = [
        "# Obs2 All-output Majority@32 Entropy Bins",
        "",
        "All 32 parser outputs enter both the answer Counter and entropy exactly as stored.",
        "Invalid outputs are neither filtered nor remapped; an invalid majority is scored",
        "as incorrect. Tertile",
        "cutpoints are recomputed from all-output entropy, so all-invalid questions remain",
        "in the analysis with entropy zero.",
        "",
        "| Dataset | n | Overall maj@32 | Invalid-majority | Low-H | Mid-H | High-H |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        scope = str(summary["benchmark"])
        rates = [by_scope[(scope, label)]["majority_correct_rate"] for label in ("low", "mid", "high")]
        lines.append(
            f"| {labels[scope]} | {summary['n']} | {format_rate(summary['majority_correct_rate'])} "
            f"| {summary['invalid_majority_count']}/{summary['n']} "
            f"({format_rate(summary['invalid_majority_rate'])}) | "
            f"{format_rate(rates[0])} | {format_rate(rates[1])} | {format_rate(rates[2])} |"
        )
    lines.extend(
        [
            "",
            "Intervals use `low: H <= q33`, `mid: q33 < H <= q67`, and",
            "`high: H > q67`. Equal entropy values are not split across bins, so tied",
            "cutpoints can produce unequal or empty bins.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = read_questions(args.questions)
    counts = {benchmark: sum(row["benchmark"] == benchmark for row in questions) for benchmark in BENCHMARKS}
    if any(count != 250 for count in counts.values()):
        raise ValueError(f"Expected 250 questions per benchmark, got {counts}")

    bin_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for benchmark in (*BENCHMARKS, "MCQ_MICRO"):
        selected = questions if benchmark == "MCQ_MICRO" else [
            row for row in questions if row["benchmark"] == benchmark
        ]
        scoped_bins, scoped_summary = summarize_scope(benchmark, selected)
        bin_rows.extend(scoped_bins)
        summaries.append(scoped_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_map = [
        {
            "benchmark": row["benchmark"],
            "question_id": row["question_id"],
            "step0_all_output_entropy": row["entropy"],
            "h_bin": entropy_bin(float(row["entropy"]), POOLED_Q33, POOLED_Q67),
            "pooled_q33": POOLED_Q33,
            "pooled_q67": POOLED_Q67,
        }
        for row in questions
    ]
    fixed_counts = {
        label: sum(row["h_bin"] == label for row in fixed_map)
        for label in ("low", "mid", "high")
    }
    if fixed_counts != {"low": 513, "mid": 487, "high": 500}:
        raise AssertionError(f"Unexpected fixed pooled-bin counts: {fixed_counts}")
    write_csv(args.output_dir / "obs2_fixed_pooled_bin_map.csv", fixed_map)
    write_csv(args.output_dir / "obs2_entropy_all_outputs.csv", bin_rows)
    write_csv(args.output_dir / "obs2_entropy_all_outputs_summary.csv", summaries)
    write_markdown(args.output_dir / "obs2_entropy_all_outputs.md", bin_rows, summaries)
    print(args.output_dir / "obs2_entropy_all_outputs.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
