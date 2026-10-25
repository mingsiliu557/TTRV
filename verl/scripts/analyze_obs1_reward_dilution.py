#!/usr/bin/env python3
"""Audit exact-string frequency reward dilution under semantic clustering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import analyze_diagnostic_obs1_qwen_clusters as obs1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--nli-judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nli-veto-max-entailment-prob", type=float, default=0.25)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_pair_judgments(path: Path):
    grouped: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        key = (str(row["case_id"]), int(row["left_index"]), int(row["right_index"]))
        direction = int(row["direction"])
        if direction in grouped[key]:
            raise ValueError(f"Duplicate pair judgment: {key}/{direction}")
        grouped[key][direction] = row
    return grouped


def load_nli_judgments(path: Path):
    judgments = {}
    for row in read_jsonl(path):
        judgments[(str(row["premise"]), str(row["hypothesis"]))] = row
    return judgments


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fragmented = [row for row in rows if row["fragmented"]]
    eligible = [row for row in fragmented if row["dilution_defined"]]
    ratios = [float(row["retention_ratio"]) for row in eligible]
    semantic = [float(row["semantic_candidate_reward_x"]) for row in eligible]
    frequency = [float(row["frequency_candidate_reward_y"]) for row in eligible]
    semantic_mass = sum(float(row["semantic_total_reward_mass"]) for row in eligible)
    frequency_mass = sum(float(row["frequency_total_reward_mass"]) for row in eligible)
    return {
        "questions_total": len(rows),
        "questions_with_valid_rollouts": sum(int(row["n_valid"]) > 0 for row in rows),
        "fragmented_questions": len(fragmented),
        "fragmented_with_correct_cluster": len(eligible),
        "fragmented_without_observed_correct_rollout": len(fragmented) - len(eligible),
        "fragmented_multiple_correct_clusters": sum(
            int(row["correct_semantic_cluster_count"]) > 1 for row in eligible
        ),
        "semantic_reward_x_macro_pct": pct(mean(semantic)),
        "frequency_reward_y_macro_pct": pct(mean(frequency)),
        "absolute_reward_drop_macro_points": (
            pct(mean(semantic)) - pct(mean(frequency)) if eligible else None
        ),
        "ratio_of_macro_rewards_pct": (
            100.0 * mean(frequency) / mean(semantic) if eligible else None
        ),
        "retention_ratio_macro_pct": pct(mean(ratios)),
        "dilution_ratio_macro_pct": pct(mean([1.0 - value for value in ratios])),
        "retention_ratio_median_pct": pct(statistics.median(ratios)) if ratios else None,
        "retention_ratio_p25_pct": pct(quantile(ratios, 0.25)),
        "retention_ratio_p75_pct": pct(quantile(ratios, 0.75)),
        "total_reward_mass_retention_pct": (
            100.0 * frequency_mass / semantic_mass if semantic_mass else None
        ),
        "dilution_at_least_25pct_count": sum((1.0 - value) >= 0.25 for value in ratios),
        "dilution_at_least_50pct_count": sum((1.0 - value) >= 0.50 for value in ratios),
        "dilution_at_least_75pct_count": sum((1.0 - value) >= 0.75 for value in ratios),
        "dilution_at_least_90pct_count": sum((1.0 - value) >= 0.90 for value in ratios),
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def format_number(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    questions = obs1.load_questions(args.stage_root)
    cases = list(questions.values())
    pair_judgments = load_pair_judgments(args.pair_root / "pair_judge_outputs.jsonl")
    nli_judgments = load_nli_judgments(args.nli_judgments)
    obs1.annotate_nli_veto(cases, pair_judgments, nli_judgments)
    cluster_args = argparse.Namespace(
        pair_protocol="semantic_v2",
        nli_veto=True,
        nli_veto_max_entailment_prob=args.nli_veto_max_entailment_prob,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        answers = list(case["answers"])
        counter: Counter[str] = case["counter"]
        if answers:
            labels, pair_count, parse_rate, order_agreement = (
                obs1.connected_component_pair_labels(
                    cluster_args, str(case["case_id"]), len(answers), pair_judgments
                )
            )
        else:
            labels, pair_count, parse_rate, order_agreement = [], 0, 1.0, 1.0

        members_by_label: dict[int, list[int]] = defaultdict(list)
        for answer_index, label in enumerate(labels):
            members_by_label[label].append(answer_index)

        semantic_clusters = []
        correct_clusters = []
        for label, indices in sorted(members_by_label.items()):
            variants = [
                {
                    "answer": answers[index],
                    "count": int(counter[answers[index]]),
                    "evaluator_correct": bool(case["answer_correct"].get(answers[index], 0)),
                }
                for index in indices
            ]
            cluster = {
                "cluster_id": label,
                "mass": sum(item["count"] for item in variants),
                "contains_evaluator_correct": any(
                    item["evaluator_correct"] for item in variants
                ),
                "variants": variants,
            }
            semantic_clusters.append(cluster)
            if cluster["contains_evaluator_correct"]:
                correct_clusters.append(cluster)

        correct_clusters.sort(key=lambda item: (-int(item["mass"]), int(item["cluster_id"])))
        selected = correct_clusters[0] if correct_clusters else None
        n_valid = int(case["n_valid"])
        fragmented = len(answers) > len(semantic_clusters)
        dilution_defined = bool(fragmented and selected is not None and n_valid > 0)

        semantic_reward = None
        frequency_reward = None
        retention = None
        semantic_total = None
        frequency_total = None
        max_variant_reward = None
        max_variant_retention = None
        selected_counts: list[int] = []
        selected_answers: list[str] = []
        selected_mass = None
        if selected is not None and n_valid > 0:
            selected_counts = [int(item["count"]) for item in selected["variants"]]
            selected_answers = [str(item["answer"]) for item in selected["variants"]]
            selected_mass = int(selected["mass"])
            semantic_reward = selected_mass / n_valid
            frequency_reward = sum(value * value for value in selected_counts) / (
                n_valid * selected_mass
            )
            retention = frequency_reward / semantic_reward
            semantic_total = selected_mass * selected_mass / n_valid
            frequency_total = sum(value * value for value in selected_counts) / n_valid
            max_variant_reward = max(selected_counts) / n_valid
            max_variant_retention = max(selected_counts) / selected_mass

        rows.append(
            {
                "benchmark": case["benchmark"],
                "question_id": case["question_id"],
                "question": case["question"],
                "n_total": 32,
                "n_valid": n_valid,
                "n_invalid": 32 - n_valid,
                "string_cluster_count": len(answers),
                "semantic_cluster_count": len(semantic_clusters),
                "fragmentation_ratio": (
                    len(answers) / len(semantic_clusters) if semantic_clusters else None
                ),
                "fragmented": fragmented,
                "pair_count": pair_count,
                "pair_parse_rate": parse_rate,
                "pair_order_agreement": order_agreement,
                "correct_semantic_cluster_count": len(correct_clusters),
                "dilution_defined": dilution_defined,
                "selected_correct_cluster_mass": selected_mass,
                "selected_correct_cluster_variant_count": len(selected_answers),
                "selected_correct_cluster_answers": selected_answers,
                "selected_correct_cluster_counts": selected_counts,
                "semantic_candidate_reward_x": semantic_reward,
                "frequency_candidate_reward_y": frequency_reward,
                "absolute_reward_drop_x_minus_y": (
                    semantic_reward - frequency_reward
                    if semantic_reward is not None and frequency_reward is not None
                    else None
                ),
                "retention_ratio": retention,
                "dilution_ratio": 1.0 - retention if retention is not None else None,
                "semantic_total_reward_mass": semantic_total,
                "frequency_total_reward_mass": frequency_total,
                "max_string_variant_reward": max_variant_reward,
                "max_string_variant_retention": max_variant_retention,
                "exact_answer_counter": dict(counter),
                "semantic_clusters": semantic_clusters,
            }
        )

    rows.sort(key=lambda row: (str(row["benchmark"]), str(row["question_id"])))
    fragmented_rows = [row for row in rows if row["fragmented"]]
    eligible_rows = [row for row in fragmented_rows if row["dilution_defined"]]
    summaries = {
        "overall": summarize(rows),
        "by_dataset": {
            benchmark: summarize([row for row in rows if row["benchmark"] == benchmark])
            for benchmark in sorted({str(row["benchmark"]) for row in rows})
        },
        "definitions": {
            "N": "number of valid rollouts for the question",
            "n_j": "count of exact-string variant j in the selected correct semantic cluster",
            "m": "sum_j n_j, the selected correct semantic cluster mass",
            "X": "m / N",
            "Y": "sum_j(n_j^2) / (N * m)",
            "retention": "Y / X = sum_j(n_j^2) / m^2",
            "dilution": "1 - retention",
        },
        "selection_rule": (
            "A semantic cluster is correct if any member is evaluator-correct. If multiple "
            "correct clusters exist, select the largest mass, then the lowest cluster id."
        ),
    }

    write_jsonl(args.output_dir / "reward_dilution_all_750_questions.jsonl", rows)
    write_csv(args.output_dir / "reward_dilution_all_750_questions.csv", rows)
    write_jsonl(args.output_dir / "reward_dilution_fragmented_260.jsonl", fragmented_rows)
    write_csv(args.output_dir / "reward_dilution_fragmented_260.csv", fragmented_rows)
    write_jsonl(args.output_dir / "reward_dilution_eligible_187.jsonl", eligible_rows)
    write_csv(args.output_dir / "reward_dilution_eligible_187.csv", eligible_rows)
    (args.output_dir / "reward_dilution_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    columns = [
        "fragmented_questions",
        "fragmented_with_correct_cluster",
        "fragmented_without_observed_correct_rollout",
        "semantic_reward_x_macro_pct",
        "frequency_reward_y_macro_pct",
        "retention_ratio_macro_pct",
        "dilution_ratio_macro_pct",
        "ratio_of_macro_rewards_pct",
        "total_reward_mass_retention_pct",
    ]
    report = [
        "# Obs1 Correct-Semantic Reward Dilution",
        "",
        "## Definitions",
        "",
        "For a selected correct semantic cluster with exact-string counts "
        "`n_1, ..., n_k`, let `m = sum(n_j)` and let `N` be the valid-rollout count:",
        "",
        "- Semantic per-candidate reward: `X = m / N`.",
        "- Exact-frequency per-candidate mean reward: `Y = sum(n_j^2) / (N * m)`.",
        "- Retention: `Y / X = sum(n_j^2) / m^2`.",
        "- Dilution: `1 - Y / X`.",
        "",
        "The total cluster reward masses are `m^2/N` under semantic frequency and "
        "`sum(n_j^2)/N` under exact frequency; their ratio is the same `Y/X`.",
        "",
        "Questions with no evaluator-correct rollout have no observed correct semantic "
        "cluster, so their dilution ratio is undefined rather than zero.",
        "",
        "## Summary",
        "",
        "| Dataset | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name, summary in [("overall", summaries["overall"]), *summaries["by_dataset"].items()]:
        report.append(
            "| " + name + " | " + " | ".join(format_number(summary[key]) for key in columns) + " |"
        )
    report.extend(
        [
            "",
            "## Files",
            "",
            "- `reward_dilution_all_750_questions.{csv,jsonl}`: every sampled question.",
            "- `reward_dilution_fragmented_260.{csv,jsonl}`: all fragmented questions, including undefined cases.",
            "- `reward_dilution_eligible_187.{csv,jsonl}`: fragmented questions with an observed correct cluster.",
            "- `reward_dilution_summary.json`: machine-readable definitions and aggregate statistics.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
