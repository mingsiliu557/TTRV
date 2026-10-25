#!/usr/bin/env python3
"""Analyze same-protocol N=32 real/shuffle/blank counterfactual rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scipy import stats

from verl.utils.reward_score.ttrl.direct_answer import UNKNOWN_LABEL, normalize_direct_answer


BENCHMARKS = (
    ("mme20", "closed"),
    ("ai2d20", "closed"),
    ("seed20", "closed"),
    ("mathvista20_hq", "closed"),
    ("mathverse20_fixed", "closed"),
    ("realworldqa20_hq", "closed"),
    ("visualsimpleqa", "open"),
    ("ocrbench_v1", "open"),
    ("aokvqa_da_val", "open"),
)
CONDITIONS = ("real", "shuffle", "blank")
Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_answer(record: dict[str, Any], kind: str) -> str | None:
    if not bool(record.get("is_valid")):
        return None
    answer = str(record.get("parsed_answer") or "").strip()
    if not answer:
        return None
    if kind == "open":
        answer = normalize_direct_answer(answer)
        return None if not answer or answer == UNKNOWN_LABEL else answer
    return answer.upper()


def load_condition(
    root: Path, benchmark: str, condition: str, kind: str, expected_qids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = root / "dumps" / benchmark / f"{condition}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("benchmark")) != benchmark or str(row.get("condition")) != condition:
                raise ValueError(f"{path}:{line_number}: benchmark/condition mismatch")
            qid = str(row["question_id"])
            # Prompts contain thousands of image-context tokens. Keep only the
            # fields needed by this post-processing pass so 216k records remain
            # comfortably bounded in memory.
            groups[qid].append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "canonical_answer": canonical_answer(row, kind),
                    "is_correct": int(row.get("is_correct") or 0),
                }
            )
    if set(groups) != expected_qids:
        raise ValueError(
            f"{benchmark}/{condition}: qid mismatch, missing={sorted(expected_qids-set(groups))[:3]}, "
            f"extra={sorted(set(groups)-expected_qids)[:3]}"
        )
    for qid, records in groups.items():
        records.sort(key=lambda row: int(row["sample_idx"]))
        indices = [int(row["sample_idx"]) for row in records]
        if indices != list(range(32)):
            raise ValueError(f"{benchmark}/{condition}/{qid}: expected sample_idx 0..31")
    return groups


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if row["canonical_answer"] is not None]
    counter = Counter(str(row["canonical_answer"]) for row in valid)
    if not counter:
        return {
            "n_total": 32,
            "n_valid": 0,
            "n_invalid": 32,
            "invalid_ratio": 1.0,
            "counter": {},
            "majority_answer": None,
            "majority_count": 0,
            "consensus_valid": None,
            "consensus_all32": 0.0,
            "majority_tie": None,
            "majority_correct": None,
        }
    answer, count = counter.most_common(1)[0]
    majority_records = [row for row in valid if row["canonical_answer"] == answer]
    max_count = max(counter.values())
    return {
        "n_total": 32,
        "n_valid": len(valid),
        "n_invalid": 32 - len(valid),
        "invalid_ratio": (32 - len(valid)) / 32,
        "counter": dict(counter),
        "majority_answer": answer,
        "majority_count": count,
        "consensus_valid": count / len(valid),
        "consensus_all32": count / 32,
        "majority_tie": int(sum(value == max_count for value in counter.values()) > 1),
        "majority_correct": int(any(bool(row.get("is_correct")) for row in majority_records)),
    }


def x_bin(value: float) -> str:
    if value <= 0.625:
        return "00_le_0.625"
    if value <= 0.75:
        return "01_0.625_0.750"
    if value <= 0.875:
        return "02_0.750_0.875"
    if value < 1.0 - 1e-12:
        return "03_0.875_lt_1.000"
    return "04_eq_1.000"


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    probability = successes / total
    denominator = 1.0 + Z95 * Z95 / total
    center = (probability + Z95 * Z95 / (2.0 * total)) / denominator
    margin = Z95 * math.sqrt(
        probability * (1.0 - probability) / total + Z95 * Z95 / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def proportion_fields(successes: int, total: int, prefix: str) -> dict[str, Any]:
    low, high = wilson(successes, total)
    return {
        f"{prefix}successes": successes,
        f"{prefix}n": total,
        f"{prefix}rate": successes / total if total else None,
        f"{prefix}ci95_low": low,
        f"{prefix}ci95_high": high,
    }


def accuracy(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [int(row[key]) for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


def scope_rows(question_rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    result = []
    for benchmark, _ in BENCHMARKS:
        result.append((benchmark, [row for row in question_rows if row["benchmark"] == benchmark]))
    result.extend(
        [
            ("MCQ_MICRO", [row for row in question_rows if row["benchmark_kind"] == "closed"]),
            ("OPEN_MICRO", [row for row in question_rows if row["benchmark_kind"] == "open"]),
            ("ALL_MICRO", list(question_rows)),
        ]
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def standard_metrics(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scope, rows in scope_rows(question_rows):
        for condition in ("shuffle", "blank"):
            valid_real = [row for row in rows if row["real_valid"]]
            valid_cf = [row for row in rows if row[f"{condition}_valid"]]
            paired = [
                row for row in rows if row["real_valid"] and row[f"{condition}_valid"]
            ]
            invariant = sum(row[f"{condition}_invariant"] == 1 for row in paired)
            vbr = sum(
                row["real_majority_correct"] == 1
                and row[f"{condition}_majority_correct"] == 0
                for row in paired
            )
            vhr = sum(
                row["real_majority_correct"] == 0
                and row[f"{condition}_majority_correct"] == 1
                for row in paired
            )
            x1_all = [row for row in rows if row["real_consensus_valid"] == 1.0]
            x1_paired = [row for row in paired if row["real_consensus_valid"] == 1.0]
            x1_invariant = sum(row[f"{condition}_invariant"] == 1 for row in x1_paired)
            strict32_all = [
                row for row in rows
                if row["real_n_valid"] == 32 and row["real_majority_count"] == 32
            ]
            strict32_paired = [
                row for row in strict32_all if row[f"{condition}_valid"]
            ]
            strict32_invariant = sum(
                row[f"{condition}_invariant"] == 1 for row in strict32_paired
            )
            paired16 = [
                row for row in paired
                if row["real_n_valid"] >= 16 and row[f"{condition}_n_valid"] >= 16
            ]
            output.append(
                {
                    "benchmark": scope,
                    "condition": condition,
                    "questions_total": len(rows),
                    "real_valid_n": len(valid_real),
                    "counterfactual_valid_n": len(valid_cf),
                    "paired_valid_n": len(paired),
                    "real_accuracy_all_valid": accuracy(valid_real, "real_majority_correct"),
                    "counterfactual_accuracy_all_valid": accuracy(
                        valid_cf, f"{condition}_majority_correct"
                    ),
                    "real_accuracy_paired": accuracy(paired, "real_majority_correct"),
                    "counterfactual_accuracy_paired": accuracy(
                        paired, f"{condition}_majority_correct"
                    ),
                    "accuracy_drop_paired": (
                        accuracy(paired, "real_majority_correct")
                        - accuracy(paired, f"{condition}_majority_correct")
                        if paired
                        else None
                    ),
                    **proportion_fields(invariant, len(paired), "invariant_"),
                    **proportion_fields(len(paired) - invariant, len(paired), "flip_"),
                    **proportion_fields(vbr, len(paired), "vbr_"),
                    **proportion_fields(vhr, len(paired), "vhr_"),
                    "x_eq_1_real_valid_n": len(x1_all),
                    "x_eq_1_paired_valid_n": len(x1_paired),
                    **proportion_fields(x1_invariant, len(x1_paired), "x_eq_1_invariant_"),
                    "x_eq_1_invariant_conservative_rate": (
                        x1_invariant / len(x1_all) if x1_all else None
                    ),
                    "strict_32_of_32_real_n": len(strict32_all),
                    "strict_32_of_32_paired_n": len(strict32_paired),
                    **proportion_fields(
                        strict32_invariant, len(strict32_paired), "strict_32_of_32_invariant_"
                    ),
                    "paired_n_valid_ge_16": len(paired16),
                    "flip_rate_n_valid_ge_16": (
                        sum(row[f"{condition}_invariant"] == 0 for row in paired16)
                        / len(paired16)
                        if paired16
                        else None
                    ),
                    "accuracy_drop_n_valid_ge_16": (
                        accuracy(paired16, "real_majority_correct")
                        - accuracy(paired16, f"{condition}_majority_correct")
                        if paired16
                        else None
                    ),
                }
            )
    return output


def consensus_bins(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    labels = [
        "00_le_0.625",
        "01_0.625_0.750",
        "02_0.750_0.875",
        "03_0.875_lt_1.000",
        "04_eq_1.000",
    ]
    for scope, rows in scope_rows(question_rows):
        for condition in ("shuffle", "blank"):
            paired = [
                row for row in rows if row["real_valid"] and row[f"{condition}_valid"]
            ]
            for label in labels:
                bucket = [row for row in paired if row["real_consensus_bin"] == label]
                invariant = sum(row[f"{condition}_invariant"] == 1 for row in bucket)
                output.append(
                    {
                        "benchmark": scope,
                        "condition": condition,
                        "consensus_bin": label,
                        "consensus_mean": (
                            sum(float(row["real_consensus_valid"]) for row in bucket) / len(bucket)
                            if bucket
                            else None
                        ),
                        **proportion_fields(invariant, len(bucket), "invariant_"),
                        "real_accuracy": accuracy(bucket, "real_majority_correct"),
                        "counterfactual_accuracy": accuracy(
                            bucket, f"{condition}_majority_correct"
                        ),
                    }
                )
    return output


def association_stats(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scope, rows in scope_rows(question_rows):
        for condition in ("shuffle", "blank"):
            paired = [
                row for row in rows if row["real_valid"] and row[f"{condition}_valid"]
            ]
            x = [float(row["real_consensus_valid"]) for row in paired]
            y = [int(row[f"{condition}_invariant"]) for row in paired]
            if len(set(x)) > 1 and len(set(y)) > 1:
                spearman = stats.spearmanr(x, y)
                point = stats.pointbiserialr(y, x)
            else:
                spearman = point = None

            benchmark_means = {}
            for benchmark in {str(row["benchmark"]) for row in paired}:
                values = [
                    float(row["real_consensus_valid"])
                    for row in paired if row["benchmark"] == benchmark
                ]
                benchmark_means[benchmark] = sum(values) / len(values)
            centered = [
                float(row["real_consensus_valid"]) - benchmark_means[str(row["benchmark"])]
                for row in paired
            ]
            centered_point = (
                stats.pointbiserialr(y, centered)
                if len(set(centered)) > 1 and len(set(y)) > 1
                else None
            )
            output.append(
                {
                    "benchmark": scope,
                    "condition": condition,
                    "paired_valid_n": len(paired),
                    "spearman_rho": spearman.statistic if spearman else None,
                    "spearman_p": spearman.pvalue if spearman else None,
                    "point_biserial_r": point.statistic if point else None,
                    "point_biserial_p": point.pvalue if point else None,
                    "dataset_centered_point_biserial_r": (
                        centered_point.statistic if centered_point else None
                    ),
                    "dataset_centered_point_biserial_p": (
                        centered_point.pvalue if centered_point else None
                    ),
                }
            )
    return output


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling = read_json(root / "sampling_config.json")
    for condition in CONDITIONS:
        protocol = sampling["protocols"][condition]
        if int(protocol["n"]) != 32:
            raise ValueError(f"{condition}: expected n=32, got {protocol['n']}")

    question_rows = []
    integrity = []
    for benchmark, kind in BENCHMARKS:
        expected_qids = {str(value) for value in read_json(root / f"subset_{benchmark}.json")}
        loaded = {
            condition: load_condition(root, benchmark, condition, kind, expected_qids)
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            integrity.append(
                {
                    "benchmark": benchmark,
                    "condition": condition,
                    "questions": len(loaded[condition]),
                    "records": sum(len(value) for value in loaded[condition].values()),
                    "samples_per_question": 32,
                }
            )
        for qid in sorted(expected_qids):
            summaries = {
                condition: summarize_group(loaded[condition][qid]) for condition in CONDITIONS
            }
            real = summaries["real"]
            row: dict[str, Any] = {
                "benchmark": benchmark,
                "benchmark_kind": kind,
                "question_id": qid,
            }
            for condition in CONDITIONS:
                summary = summaries[condition]
                for key, value in summary.items():
                    row[f"{condition}_{key}"] = value
                row[f"{condition}_valid"] = int(summary["majority_answer"] is not None)
            row["real_consensus_bin"] = (
                x_bin(float(real["consensus_valid"]))
                if real["consensus_valid"] is not None
                else None
            )
            for condition in ("shuffle", "blank"):
                cf = summaries[condition]
                paired = real["majority_answer"] is not None and cf["majority_answer"] is not None
                row[f"{condition}_paired_valid"] = int(paired)
                row[f"{condition}_invariant"] = (
                    int(real["majority_answer"] == cf["majority_answer"]) if paired else None
                )
                row[f"{condition}_flip"] = (
                    1 - int(row[f"{condition}_invariant"]) if paired else None
                )
            question_rows.append(row)

    metrics = standard_metrics(question_rows)
    bins = consensus_bins(question_rows)
    associations = association_stats(question_rows)
    write_csv(output_dir / "obs3_question_level.csv", question_rows)
    write_csv(output_dir / "obs3_standard_metrics.csv", metrics)
    write_csv(output_dir / "obs3_consensus_bins.csv", bins)
    write_csv(output_dir / "obs3_association.csv", associations)
    write_csv(output_dir / "integrity.csv", integrity)
    summary = {
        "experiment_root": str(root),
        "protocol": sampling,
        "question_rows": len(question_rows),
        "standard_metrics": metrics,
        "associations": associations,
    }
    (output_dir / "obs3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "questions": len(question_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
