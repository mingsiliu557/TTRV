#!/usr/bin/env python3
"""Run the Stage B analyses supported by the current diagnostic dumps.

The current Stage A protocol uses sampled N=32 responses for real images and
greedy N=1 responses for shuffled/blank images. A sampling-diversity gate
checks whether real N=32 is usable. Obs3 explicitly labels the mixed decoding
protocol and must not be reported as same-protocol VRS/BD.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VERL_ROOT = Path(__file__).resolve().parents[1]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score.ttrl.direct_answer import UNKNOWN_LABEL, normalize_direct_answer
from verl.utils.reward_score.ttrl.ttt_metrics import _frequency_entropy_rewards, _normalized_entropy


BENCHMARKS = (
    ("mme20", "closed", 2),
    ("ai2d20", "closed", 4),
    ("seed20", "closed", 4),
    ("mathvista20_hq", "closed", 6),
    ("mathverse20_fixed", "closed", 6),
    ("realworldqa20_hq", "closed", 4),
    ("visualsimpleqa", "open", None),
    ("ocrbench_v1", "open", None),
    ("aokvqa_da_val", "open", None),
)
BENCHMARK_KIND = {key: kind for key, kind, _ in BENCHMARKS}
ANSWER_SPACE = {key: size for key, _, size in BENCHMARKS}
CONDITIONS = ("real", "shuffle", "blank")
EXPECTED_SAMPLES = {"real": 32, "shuffle": 1, "blank": 1}
MIXED_PROTOCOL = "real=sampled-majority@32;counterfactual=greedy@1"
Z_95 = 1.959963984540054
MIN_RAW_DIVERSE_GROUP_RATE = 0.05


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    z2 = Z_95 * Z_95
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    spread = Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def proportion_fields(successes: int, total: int, prefix: str = "") -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {
        f"{prefix}successes": successes,
        f"{prefix}n": total,
        f"{prefix}rate": successes / total if total else None,
        f"{prefix}ci95_low": low,
        f"{prefix}ci95_high": high,
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def safe_mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def safe_stdev(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.stdev(items) if len(items) > 1 else 0.0 if items else None


def canonical_answer(record: dict[str, Any], benchmark: str) -> str | None:
    if not bool(record["is_valid"]):
        return None
    answer = str(record.get("parsed_answer") or "").strip()
    if BENCHMARK_KIND[benchmark] == "open":
        answer = normalize_direct_answer(answer)
        return None if answer == UNKNOWN_LABEL else answer
    return answer.upper() if answer else None


def load_condition(
    dump_root: Path,
    benchmark: str,
    condition: str,
    expected_qids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path = dump_root / benchmark / f"{condition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    expected_samples = EXPECTED_SAMPLES[condition]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, int]] = set()
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("benchmark") != benchmark or raw.get("condition") != condition:
                raise ValueError(f"{path}:{line_number}: benchmark/condition mismatch")
            qid = str(raw.get("question_id"))
            sample_idx = int(raw.get("sample_idx"))
            key = (qid, sample_idx)
            if key in seen_keys:
                raise ValueError(f"{path}:{line_number}: duplicate key {key}")
            seen_keys.add(key)
            compact = {
                "question_id": qid,
                "sample_idx": sample_idx,
                "raw_text": str(raw.get("raw_text") or ""),
                "parsed_answer": str(raw.get("parsed_answer") or ""),
                "is_valid": bool(raw.get("is_valid")),
                "is_correct": int(raw.get("is_correct") or 0),
                "verifier_reward": float(raw.get("verifier_reward") or 0.0),
                "shuffle_src_qid": raw.get("shuffle_src_qid"),
                "ground_truth": str(raw.get("ground_truth") or ""),
            }
            groups[qid].append(compact)
            records += 1

    expected_set = set(expected_qids)
    actual_set = set(groups)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)[:10]
        extra = sorted(actual_set - expected_set)[:10]
        raise ValueError(f"{benchmark}/{condition}: qid mismatch missing={missing} extra={extra}")
    for qid, samples in groups.items():
        samples.sort(key=lambda item: item["sample_idx"])
        indices = [item["sample_idx"] for item in samples]
        if indices != list(range(expected_samples)):
            raise ValueError(f"{benchmark}/{condition}/{qid}: sample indices={indices}")
    expected_records = len(expected_qids) * expected_samples
    if records != expected_records:
        raise ValueError(f"{benchmark}/{condition}: expected {expected_records} records, got {records}")
    return groups, {
        "benchmark": benchmark,
        "condition": condition,
        "questions": len(groups),
        "records": records,
        "samples_per_question": expected_samples,
        "status": "complete",
    }


def summarize_real_group(benchmark: str, qid: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_texts = [str(record["raw_text"]) for record in records]
    parsed_all = [str(record["parsed_answer"]) for record in records]
    official_rewards, official_counter, official_h = _frequency_entropy_rewards(parsed_all, entropy_coef=0.75)
    official_majority, official_count = official_counter.most_common(1)[0]
    valid_pairs = [
        (canonical_answer(record, benchmark), record)
        for record in records
        if canonical_answer(record, benchmark) is not None
    ]
    valid_answers = [answer for answer, _ in valid_pairs]
    valid_counter = Counter(valid_answers)
    n_valid = len(valid_answers)
    if valid_counter:
        majority, majority_count = valid_counter.most_common(1)[0]
        majority_records = [record for answer, record in valid_pairs if answer == majority]
        majority_correct = int(any(record["is_correct"] for record in majority_records))
        majority_verifier_reward = max(record["verifier_reward"] for record in majority_records)
        valid_rewards, _, valid_h = _frequency_entropy_rewards(valid_answers, entropy_coef=0.75)
        valid_majority_reward = valid_rewards[valid_answers.index(majority)]
        consensus = majority_count / n_valid
    else:
        majority = None
        majority_count = 0
        majority_correct = None
        majority_verifier_reward = None
        valid_h = None
        valid_majority_reward = None
        consensus = None

    official_majority_is_valid = any(
        str(record["parsed_answer"]) == official_majority and canonical_answer(record, benchmark) is not None
        for record in records
    )
    return {
        "benchmark": benchmark,
        "benchmark_kind": BENCHMARK_KIND[benchmark],
        "question_id": qid,
        "n_total": len(records),
        "n_valid": n_valid,
        "n_invalid": len(records) - n_valid,
        "invalid_ratio": (len(records) - n_valid) / len(records),
        "raw_response_support": len(set(raw_texts)),
        "parsed_all_support": len(official_counter),
        "valid_support": len(valid_counter),
        "valid_majority_answer": majority,
        "valid_majority_count": majority_count,
        "valid_consensus_strength": consensus,
        "valid_normalized_entropy": valid_h,
        "valid_majority_correct": majority_correct,
        "valid_majority_verifier_reward": majority_verifier_reward,
        "valid_frequency_entropy_reward": valid_majority_reward,
        "official_all_majority_answer": official_majority,
        "official_all_majority_count": official_count,
        "official_all_majority_is_valid": int(official_majority_is_valid),
        "official_all_normalized_entropy": official_h,
        "official_all_frequency_entropy_reward": official_rewards[parsed_all.index(official_majority)],
    }


def summarize_counterfactual(
    benchmark: str,
    condition: str,
    qid: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError(f"{benchmark}/{condition}/{qid}: expected greedy N=1")
    record = records[0]
    answer = canonical_answer(record, benchmark)
    return {
        "benchmark": benchmark,
        "question_id": qid,
        "condition": condition,
        "answer": answer,
        "is_valid": int(answer is not None),
        "is_correct": int(record["is_correct"]) if answer is not None else None,
        "shuffle_src_qid": record.get("shuffle_src_qid"),
    }


def invalid_rows_for_condition(
    benchmark: str,
    condition: str,
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    sample_total = sum(len(records) for records in groups.values())
    sample_invalid = sum(
        canonical_answer(record, benchmark) is None for records in groups.values() for record in records
    )
    question_total = len(groups)
    question_any = sum(
        any(canonical_answer(record, benchmark) is None for record in records) for records in groups.values()
    )
    question_all = sum(
        all(canonical_answer(record, benchmark) is None for record in records) for records in groups.values()
    )
    base = {
        "benchmark": benchmark,
        "benchmark_kind": BENCHMARK_KIND[benchmark],
        "condition": condition,
        "samples_per_question": EXPECTED_SAMPLES[condition],
    }
    return [
        {**base, "metric": "sample_invalid", "unit": "sample", **proportion_fields(sample_invalid, sample_total)},
        {**base, "metric": "question_any_invalid", "unit": "question", **proportion_fields(question_any, question_total)},
        {**base, "metric": "question_all_invalid", "unit": "question", **proportion_fields(question_all, question_total)},
    ]


def entropy_bin(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def fixed_histogram(values: list[float], bins: int = 10) -> list[tuple[float, float, int]]:
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int(value * bins)))
        counts[index] += 1
    return [(idx / bins, (idx + 1) / bins, count) for idx, count in enumerate(counts)]


def x_bin(value: float) -> str:
    if value <= 0.5:
        return "00_le_0.500"
    if value <= 0.625:
        return "01_0.500_0.625"
    if value <= 0.75:
        return "02_0.625_0.750"
    if value <= 0.875:
        return "03_0.750_0.875"
    if value < 1.0 - 1e-12:
        return "04_0.875_lt_1.000"
    return "05_eq_1.000"


def aggregate_invalid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    for scope, allowed in (
        ("MCQ_MICRO", {key for key, kind, _ in BENCHMARKS if kind == "closed"}),
        ("OPEN_MICRO", {key for key, kind, _ in BENCHMARKS if kind == "open"}),
        ("ALL_MICRO", {key for key, _, _ in BENCHMARKS}),
    ):
        for condition in CONDITIONS:
            for metric in ("sample_invalid", "question_any_invalid", "question_all_invalid"):
                selected = [
                    row for row in rows
                    if row["benchmark"] in allowed and row["condition"] == condition and row["metric"] == metric
                ]
                successes = sum(int(row["successes"]) for row in selected)
                total = sum(int(row["n"]) for row in selected)
                output.append(
                    {
                        "benchmark": scope,
                        "benchmark_kind": "aggregate",
                        "condition": condition,
                        "samples_per_question": EXPECTED_SAMPLES[condition],
                        "metric": metric,
                        "unit": selected[0]["unit"],
                        **proportion_fields(successes, total),
                    }
                )
    return output


def make_entropy_outputs(question_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = []
    for benchmark, kind, _ in BENCHMARKS:
        if kind == "closed":
            scopes.append((benchmark, [row for row in question_rows if row["benchmark"] == benchmark]))
    scopes.extend(
        [
            ("MCQ_MICRO", [row for row in question_rows if row["benchmark_kind"] == "closed"]),
        ]
    )
    for scope, selected_all in scopes:
        selected = [row for row in selected_all if row["valid_normalized_entropy"] is not None]
        values = [float(row["valid_normalized_entropy"]) for row in selected]
        q25 = quantile(values, 0.25)
        q33 = quantile(values, 1.0 / 3.0)
        q50 = quantile(values, 0.5)
        q67 = quantile(values, 2.0 / 3.0)
        q75 = quantile(values, 0.75)
        correct_values = [
            float(row["valid_normalized_entropy"]) for row in selected if row["valid_majority_correct"] == 1
        ]
        wrong_values = [
            float(row["valid_normalized_entropy"]) for row in selected if row["valid_majority_correct"] == 0
        ]
        summaries.append(
            {
                "benchmark": scope,
                "questions_total": len(selected_all),
                "questions_valid": len(selected),
                "questions_all_invalid": len(selected_all) - len(selected),
                "h_mean": safe_mean(values),
                "h_stdev": safe_stdev(values),
                "h_min": min(values) if values else None,
                "h_q25": q25,
                "h_median": q50,
                "h_q75": q75,
                "h_max": max(values) if values else None,
                "h_iqr": q75 - q25 if q25 is not None and q75 is not None else None,
                "h_unique_rounded_6dp": len({round(value, 6) for value in values}),
                "h_correct_mean": safe_mean(correct_values),
                "h_wrong_mean": safe_mean(wrong_values),
                "tertile_q33": q33,
                "tertile_q67": q67,
            }
        )
        for lower, upper, count in fixed_histogram(values):
            low, high = wilson_interval(count, len(values))
            histogram_rows.append(
                {
                    "benchmark": scope,
                    "h_lower_inclusive": lower,
                    "h_upper_inclusive": upper,
                    "count": count,
                    "n": len(values),
                    "proportion": count / len(values) if values else None,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        if q33 is None or q67 is None:
            continue
        for label in ("low", "mid", "high"):
            bucket = [row for row in selected if entropy_bin(float(row["valid_normalized_entropy"]), q33, q67) == label]
            correct = sum(row["valid_majority_correct"] == 1 for row in bucket)
            random_baseline = safe_mean(
                1.0 / ANSWER_SPACE[row["benchmark"]]
                for row in bucket
                if ANSWER_SPACE[row["benchmark"]] is not None
            )
            bin_rows.append(
                {
                    "benchmark": scope,
                    "h_bin": label,
                    "q33": q33,
                    "q67": q67,
                    "h_mean": safe_mean(float(row["valid_normalized_entropy"]) for row in bucket),
                    "random_accuracy": random_baseline,
                    **proportion_fields(correct, len(bucket), prefix="majority_correct_"),
                }
            )
    return summaries, histogram_rows, bin_rows


def scope_question_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    scopes = [(benchmark, [row for row in rows if row["benchmark"] == benchmark]) for benchmark, _, _ in BENCHMARKS]
    scopes.extend(
        [
            ("MCQ_MICRO", [row for row in rows if row["benchmark_kind"] == "closed"]),
            ("OPEN_MICRO", [row for row in rows if row["benchmark_kind"] == "open"]),
            ("ALL_MICRO", list(rows)),
        ]
    )
    return scopes


def make_obs3_outputs(question_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    consensus_rows: list[dict[str, Any]] = []
    standard_rows: list[dict[str, Any]] = []
    for scope, scoped in scope_question_rows(question_rows):
        for condition in ("shuffle", "blank"):
            selected_all = [row for row in scoped if row["condition"] == condition]
            paired = [row for row in selected_all if row["paired_valid"] == 1]
            for label in (
                "00_le_0.500",
                "01_0.500_0.625",
                "02_0.625_0.750",
                "03_0.750_0.875",
                "04_0.875_lt_1.000",
                "05_eq_1.000",
            ):
                bucket = [row for row in paired if row["consensus_bin"] == label]
                invariant = sum(row["image_invariant"] == 1 for row in bucket)
                consensus_rows.append(
                    {
                        "benchmark": scope,
                        "condition": condition,
                        "row_type": "consensus_bin",
                        "consensus_bin": label,
                        "protocol": MIXED_PROTOCOL,
                        **proportion_fields(invariant, len(bucket), prefix="image_invariant_"),
                    }
                )
            perfect = [row for row in paired if abs(float(row["real_consensus_strength"]) - 1.0) < 1e-12]
            perfect_invariant = sum(row["image_invariant"] == 1 for row in perfect)
            consensus_rows.append(
                {
                    "benchmark": scope,
                    "condition": condition,
                    "row_type": "money_x_eq_1",
                    "consensus_bin": "05_eq_1.000",
                    "protocol": MIXED_PROTOCOL,
                    **proportion_fields(perfect_invariant, len(perfect), prefix="image_invariant_"),
                }
            )
            high = [row for row in paired if float(row["real_consensus_strength"]) >= 0.75]
            dangerous = sum(row["image_invariant"] == 1 for row in high)
            consensus_rows.append(
                {
                    "benchmark": scope,
                    "condition": condition,
                    "row_type": "danger_high_consensus_invariant",
                    "consensus_bin": "x_ge_0.75",
                    "protocol": MIXED_PROTOCOL,
                    **proportion_fields(dangerous, len(high), prefix="image_invariant_"),
                }
            )

            real_correct = sum(row["real_majority_correct"] == 1 for row in paired)
            cf_correct = sum(row["counterfactual_correct"] == 1 for row in paired)
            flips = sum(row["image_invariant"] == 0 for row in paired)
            vbr = sum(row["real_majority_correct"] == 1 and row["counterfactual_correct"] == 0 for row in paired)
            vhr = sum(row["real_majority_correct"] == 0 and row["counterfactual_correct"] == 1 for row in paired)
            row: dict[str, Any] = {
                "benchmark": scope,
                "condition": condition,
                "protocol": MIXED_PROTOCOL,
                "questions_total": len(selected_all),
                "paired_valid_n": len(paired),
                "excluded_real_all_invalid": sum(row["real_valid"] == 0 for row in selected_all),
                "excluded_counterfactual_invalid": sum(row["counterfactual_valid"] == 0 for row in selected_all),
                **proportion_fields(real_correct, len(paired), prefix="real_accuracy_"),
                **proportion_fields(cf_correct, len(paired), prefix="counterfactual_accuracy_"),
                **proportion_fields(flips, len(paired), prefix="image_sensitivity_"),
                **proportion_fields(vbr, len(paired), prefix="vbr_"),
                **proportion_fields(vhr, len(paired), prefix="vhr_"),
            }
            row["descriptive_accuracy_delta"] = (
                real_correct / len(paired) - cf_correct / len(paired) if paired else None
            )
            row["metric_name_warning"] = "Not canonical VRS/BD because decoding protocols differ"
            standard_rows.append(row)
    return consensus_rows, standard_rows


def fmt_rate(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(output)


def svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    width, height = 1100, 600
    left, right, top, bottom = 90, 40, 70, 130
    chart_w, chart_h = width - left - right, height - top - bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-9)
    bar_w = chart_w / max(1, len(values)) * 0.7
    gap = chart_w / max(1, len(values))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="black"/>',
        f'<text x="22" y="{top+chart_h/2}" transform="rotate(-90 22 {top+chart_h/2})" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + chart_h * (1.0 - tick / 5)
        elements.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        elements.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        bar_h = chart_h * value / max_value
        y = top + chart_h - bar_h
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#3274a1"/>')
        elements.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        elements.append(f'<text x="{x+bar_w/2:.1f}" y="{top+chart_h+18:.1f}" transform="rotate(35 {x+bar_w/2:.1f} {top+chart_h+18:.1f})" text-anchor="start" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def make_sampling_diversity_rows(real_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = [
        (benchmark, [row for row in real_question_rows if row["benchmark"] == benchmark])
        for benchmark, _, _ in BENCHMARKS
    ]
    scopes.extend(
        [
            ("MCQ_MICRO", [row for row in real_question_rows if row["benchmark_kind"] == "closed"]),
            ("OPEN_MICRO", [row for row in real_question_rows if row["benchmark_kind"] == "open"]),
            ("ALL_MICRO", list(real_question_rows)),
        ]
    )
    for scope, selected in scopes:
        raw_diverse = sum(int(row["raw_response_support"]) > 1 for row in selected)
        parsed_diverse = sum(int(row["parsed_all_support"]) > 1 for row in selected)
        valid_diverse = sum(int(row["valid_support"]) > 1 for row in selected)
        rows.append(
            {
                "benchmark": scope,
                "questions": len(selected),
                **proportion_fields(raw_diverse, len(selected), prefix="raw_support_gt1_"),
                **proportion_fields(parsed_diverse, len(selected), prefix="parsed_support_gt1_"),
                **proportion_fields(valid_diverse, len(selected), prefix="valid_support_gt1_"),
                "raw_support_mean": safe_mean(float(row["raw_response_support"]) for row in selected),
                "raw_support_max": max(int(row["raw_response_support"]) for row in selected),
                "sampling_gate_threshold": MIN_RAW_DIVERSE_GROUP_RATE,
                "sampling_gate": "pass" if raw_diverse / len(selected) >= MIN_RAW_DIVERSE_GROUP_RATE
                else "fail_effectively_identical_sampling",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    dump_root = experiment_root / "dumps"
    output_dir = (args.output_dir or experiment_root / "analysis_stage_b").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_config = read_json(experiment_root / "sampling_config.json")
    if sampling_config["protocols"]["real"]["n"] != 32:
        raise ValueError("Obs2/Obs3 require real N=32")
    if sampling_config["protocols"]["shuffle"]["n"] != 1 or sampling_config["protocols"]["blank"]["n"] != 1:
        raise ValueError("This script describes the current counterfactual greedy N=1 protocol")

    audit_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    real_question_rows: list[dict[str, Any]] = []
    obs3_question_rows: list[dict[str, Any]] = []

    for benchmark, kind, _ in BENCHMARKS:
        expected_qids = [str(value) for value in read_json(experiment_root / f"subset_{benchmark}.json")]
        loaded: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for condition in CONDITIONS:
            groups, audit = load_condition(dump_root, benchmark, condition, expected_qids)
            loaded[condition] = groups
            audit_rows.append(audit)
            invalid_rows.extend(invalid_rows_for_condition(benchmark, condition, groups))

        for qid in expected_qids:
            real_summary = summarize_real_group(benchmark, qid, loaded["real"][qid])
            real_question_rows.append(real_summary)
            for condition in ("shuffle", "blank"):
                cf = summarize_counterfactual(benchmark, condition, qid, loaded[condition][qid])
                real_valid = int(real_summary["valid_majority_answer"] is not None)
                paired_valid = int(real_valid == 1 and cf["is_valid"] == 1)
                invariant = (
                    int(real_summary["valid_majority_answer"] == cf["answer"])
                    if paired_valid
                    else None
                )
                strength = real_summary["valid_consensus_strength"]
                obs3_question_rows.append(
                    {
                        "benchmark": benchmark,
                        "benchmark_kind": kind,
                        "question_id": qid,
                        "condition": condition,
                        "protocol": MIXED_PROTOCOL,
                        "real_n_total": real_summary["n_total"],
                        "real_n_valid": real_summary["n_valid"],
                        "real_invalid_ratio": real_summary["invalid_ratio"],
                        "real_valid": real_valid,
                        "real_majority_answer": real_summary["valid_majority_answer"],
                        "real_majority_correct": real_summary["valid_majority_correct"],
                        "real_majority_count": real_summary["valid_majority_count"],
                        "real_consensus_strength": strength,
                        "real_normalized_entropy": real_summary["valid_normalized_entropy"],
                        "consensus_bin": x_bin(float(strength)) if strength is not None else None,
                        "counterfactual_answer": cf["answer"],
                        "counterfactual_valid": cf["is_valid"],
                        "counterfactual_correct": cf["is_correct"],
                        "paired_valid": paired_valid,
                        "image_invariant": invariant,
                        "shuffle_src_qid": cf["shuffle_src_qid"],
                    }
                )
        del loaded

    invalid_rows = aggregate_invalid(invalid_rows)
    sampling_diversity_rows = make_sampling_diversity_rows(real_question_rows)
    pooled_sampling = next(row for row in sampling_diversity_rows if row["benchmark"] == "ALL_MICRO")
    sampling_gate_pass = pooled_sampling["sampling_gate"] == "pass"
    entropy_summary, entropy_histogram, entropy_bins = make_entropy_outputs(real_question_rows)
    obs3_consensus, obs3_standard = make_obs3_outputs(obs3_question_rows)
    inferential_status = "ready" if sampling_gate_pass else "blocked_duplicate_validation_sampling"
    gated_outputs = (entropy_summary, entropy_histogram, entropy_bins, obs3_consensus, obs3_standard, obs3_question_rows)
    for rows in gated_outputs:
        for row in rows:
            row["analysis_status"] = inferential_status

    write_json(
        output_dir / "integrity_audit.json",
        {
            "status": "complete",
            "experiment_root": str(experiment_root),
            "sampling_config": sampling_config,
            "protocol_warning": "Obs3 is mixed-protocol descriptive analysis, not canonical same-protocol VRS/BD.",
            "datasets": audit_rows,
        },
    )
    write_csv(output_dir / "obs0_invalid.csv", invalid_rows)
    write_csv(output_dir / "sampling_diversity_audit.csv", sampling_diversity_rows)
    write_csv(output_dir / "obs2_entropy_questions.csv", [row for row in real_question_rows if row["benchmark_kind"] == "closed"])
    write_csv(output_dir / "obs2_entropy_summary.csv", entropy_summary)
    write_csv(output_dir / "obs2_entropy_histogram.csv", entropy_histogram)
    write_csv(output_dir / "obs2_entropy.csv", entropy_bins)
    write_csv(output_dir / "obs3_question_level.csv", obs3_question_rows)
    write_csv(output_dir / "obs3_consensus_grounding.csv", obs3_consensus)
    write_csv(output_dir / "obs3_standard_metrics.csv", obs3_standard)

    pooled_hist = [row for row in entropy_histogram if row["benchmark"] == "MCQ_MICRO"]
    svg_bar_chart(
        output_dir / "obs2_entropy.svg",
        "Valid-only answer entropy, MCQ pooled",
        [f"{row['h_lower_inclusive']:.1f}-{row['h_upper_inclusive']:.1f}" for row in pooled_hist],
        [float(row["proportion"] or 0.0) for row in pooled_hist],
        "Question proportion",
    )
    pooled_obs3 = [
        row for row in obs3_consensus
        if row["benchmark"] == "ALL_MICRO" and row["condition"] == "shuffle" and row["row_type"] == "consensus_bin"
    ]
    svg_bar_chart(
        output_dir / "obs3_consensus_grounding.svg",
        "Mixed protocol: real majority@32 vs shuffled-image greedy",
        [row["consensus_bin"] for row in pooled_obs3],
        [float(row["image_invariant_rate"] or 0.0) for row in pooled_obs3],
        "Image-invariant rate",
    )

    obs0_headline = [
        row for row in invalid_rows
        if row["benchmark"] in {key for key, _, _ in BENCHMARKS} and row["metric"] == "question_all_invalid"
    ]
    obs0_md = [
        "# Obs0: Invalid outputs",
        "",
        "Invalid is reported, not repaired. Obs1/Obs2/Obs3 use valid-only answers.",
        "The headline uses question-level all-invalid rates (n=250 per dataset). Sample-level rows remain in CSV, "
        "but repeated real-image samples are not independent and must not be used for confidence intervals.",
        "",
        markdown_table(
            ["Benchmark", "Condition", "All-invalid questions", "n", "95% Wilson CI"],
            [
                [
                    row["benchmark"],
                    row["condition"],
                    fmt_rate(row["rate"]),
                    row["n"],
                    f"[{fmt_rate(row['ci95_low'])}, {fmt_rate(row['ci95_high'])}]",
                ]
                for row in obs0_headline
            ],
        ),
        "",
    ]
    (output_dir / "obs0_invalid.md").write_text("\n".join(obs0_md), encoding="utf-8")

    entropy_pooled = next(row for row in entropy_summary if row["benchmark"] == "MCQ_MICRO")
    if sampling_gate_pass:
        entropy_status = (
            f"**Status: READY.** {pooled_sampling['raw_support_gt1_successes']}/"
            f"{pooled_sampling['questions']} groups contain more than one unique raw response."
        )
    else:
        entropy_status = (
            f"**Status: BLOCKED by the sampling-diversity gate.** "
            f"{pooled_sampling['questions'] - pooled_sampling['raw_support_gt1_successes']}/"
            f"{pooled_sampling['questions']} groups have one unique raw response across 32 samples."
        )
    entropy_md = [
        "# Obs2: Valid-only consensus entropy",
        "",
        entropy_status,
        "",
        "H is recomputed from valid parsed answers using the repository's `_normalized_entropy` implementation.",
        "",
        markdown_table(
            ["Scope", "valid n", "all-invalid", "mean H", "IQR", "min", "max"],
            [
                [
                    row["benchmark"], row["questions_valid"], row["questions_all_invalid"],
                    f"{row['h_mean']:.4f}" if row["h_mean"] is not None else "NA",
                    f"{row['h_iqr']:.4f}" if row["h_iqr"] is not None else "NA",
                    f"{row['h_min']:.4f}" if row["h_min"] is not None else "NA",
                    f"{row['h_max']:.4f}" if row["h_max"] is not None else "NA",
                ]
                for row in entropy_summary
            ],
        ),
        "",
        f"Pooled MCQ empirical tertile cutpoints: q33={entropy_pooled['tertile_q33']:.6f}, q67={entropy_pooled['tertile_q67']:.6f}.",
        "",
        markdown_table(
            ["H bin", "accuracy", "n", "95% Wilson CI", "random baseline"],
            [
                [
                    row["h_bin"], fmt_rate(row["majority_correct_rate"]), row["majority_correct_n"],
                    f"[{fmt_rate(row['majority_correct_ci95_low'])}, {fmt_rate(row['majority_correct_ci95_high'])}]",
                    fmt_rate(row["random_accuracy"]),
                ]
                for row in entropy_bins if row["benchmark"] == "MCQ_MICRO"
            ],
        ),
        "",
    ]
    (output_dir / "obs2_entropy.md").write_text("\n".join(entropy_md), encoding="utf-8")

    standard_headline = [row for row in obs3_standard if row["benchmark"] not in {"MCQ_MICRO", "OPEN_MICRO", "ALL_MICRO"}]
    if sampling_gate_pass:
        obs3_status = (
            f"**Status: READY for mixed-protocol descriptive analysis.** "
            f"{pooled_sampling['raw_support_gt1_successes']}/{pooled_sampling['questions']} "
            "real-image groups contain more than one unique raw response."
        )
    else:
        obs3_status = (
            f"**Status: BLOCKED by the sampling-diversity gate.** In "
            f"{pooled_sampling['questions'] - pooled_sampling['raw_support_gt1_successes']}/"
            f"{pooled_sampling['questions']} groups, majority@32 is effectively one repeated draw."
        )
    obs3_md = [
        "# Obs3: Consensus versus counterfactual invariance",
        "",
        obs3_status,
        "",
        "**Protocol warning:** real uses sampled valid-majority@32; shuffle/blank use greedy@1. "
        "These are descriptive invariance and accuracy-delta statistics, not canonical same-decoding VRS/BD.",
        "",
        markdown_table(
            ["Benchmark", "CF", "paired valid n", "real acc", "CF acc", "delta", "flip rate"],
            [
                [
                    row["benchmark"], row["condition"], row["paired_valid_n"],
                    fmt_rate(row["real_accuracy_rate"]), fmt_rate(row["counterfactual_accuracy_rate"]),
                    fmt_rate(row["descriptive_accuracy_delta"]), fmt_rate(row["image_sensitivity_rate"]),
                ]
                for row in standard_headline
            ],
        ),
        "",
    ]
    (output_dir / "obs3_consensus_grounding.md").write_text("\n".join(obs3_md), encoding="utf-8")

    write_json(
        output_dir / "analysis_manifest.json",
        {
            "status": "ready_available_analyses" if sampling_gate_pass else "partial_blocked",
            "experiment_root": str(experiment_root),
            "completed": (
                ["integrity_audit", "obs0_invalid", "sampling_diversity_audit", "obs2_entropy", "obs3_consensus_grounding"]
                if sampling_gate_pass
                else ["integrity_audit", "obs0_invalid", "sampling_diversity_audit"]
            ),
            "deferred": (
                {"obs1_fragmentation": "Requires the separate DeBERTa-Large-MNLI semantic clustering pass."}
                if sampling_gate_pass
                else {
                    "obs1_fragmentation": "Blocked by effectively identical sampling.",
                    "obs2_entropy": "Blocked by effectively identical sampling.",
                    "obs3_consensus_grounding": "Blocked by effectively identical sampling.",
                }
            ),
            "headline_policy": "Obs0 reports invalid; provisional Obs2/Obs3 tables are valid-only.",
            "obs3_protocol": MIXED_PROTOCOL,
            "sampling_diversity_gate": pooled_sampling,
            "diagnosed_generation_cause": None if sampling_gate_pass else "Repeated requests inherited the same request-level seed.",
            "plot_format": "SVG (matplotlib is not installed in the ttrv environment)",
        },
    )
    print(json.dumps({"status": "ready" if sampling_gate_pass else "partial_blocked", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
