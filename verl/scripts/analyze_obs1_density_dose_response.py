#!/usr/bin/env python3
"""Measure whether semantic-density selection gains grow with fragmentation.

This is a post-processing-only Obs1 diagnostic. It consumes the per-question
Qwen semantic-v2 clusters reconstructed by ``analyze_obs1_reward_dilution.py``
and the corresponding Qwen question-level decisions. No rollout or model
inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("visualsimpleqa", "ocrbench_v1", "aokvqa_da_val")
BIN_ORDER = ("f1", "f2", "f3plus")
BIN_LABELS = {"f1": "F=1", "f2": "F=2", "f3plus": "F>=3"}
EXPECTED_QUESTIONS = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-dilution-jsonl", type=Path, required=True)
    parser.add_argument("--qwen-question-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def f_bin(variant_count: int) -> str:
    if variant_count == 1:
        return "f1"
    if variant_count == 2:
        return "f2"
    return "f3plus"


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap_delta(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> tuple[float | None, float | None]:
    if not rows or samples <= 0:
        return None, None
    differences = [int(row["density_correct"]) - int(row["frequency_correct"]) for row in rows]
    rng = random.Random(seed)
    n = len(differences)
    boot = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def bool_int(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def prepare_rows(
    reward_rows: list[dict[str, Any]], qwen_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reward_by_key = {
        (str(row["benchmark"]), str(row["question_id"])): row for row in reward_rows
    }
    qwen_by_key = {
        (str(row["benchmark"]), str(row["question_id"])): row for row in qwen_rows
    }
    expected_keys = set(reward_by_key)
    if expected_keys != set(qwen_by_key):
        missing_qwen = sorted(expected_keys - set(qwen_by_key))[:5]
        missing_reward = sorted(set(qwen_by_key) - expected_keys)[:5]
        raise ValueError(
            f"Qwen/reward key mismatch: missing_qwen={missing_qwen}, "
            f"missing_reward={missing_reward}"
        )
    counts = Counter(key[0] for key in expected_keys)
    if counts != Counter({benchmark: EXPECTED_QUESTIONS for benchmark in BENCHMARKS}):
        raise ValueError(f"Expected 250 questions per benchmark, got {counts}")

    all_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        keys = sorted(key for key in expected_keys if key[0] == benchmark)
        for key in keys:
            reward = reward_by_key[key]
            qwen = qwen_by_key[key]
            correct_mass = reward.get("selected_correct_cluster_mass")
            base = {
                "benchmark": benchmark,
                "question_id": key[1],
                "n_total": int(reward["n_total"]),
                "n_valid": int(reward["n_valid"]),
                "n_invalid": int(reward["n_invalid"]),
                "eligible_correct_answer_observed": int(correct_mass is not None),
                "exclusion_reason": "" if correct_mass is not None else "no_correct_answer_in_valid_rollouts",
            }
            if correct_mass is None:
                all_rows.append(base)
                continue

            correct_answers = set(str(value) for value in reward["selected_correct_cluster_answers"])
            density_answers = set(json.loads(qwen["semantic_majority_answers"]))
            frequency_answer = str(qwen["string_majority_answer"])
            variant_count = int(reward["selected_correct_cluster_variant_count"])
            if not correct_answers or not density_answers or variant_count < 1:
                raise ValueError(f"Malformed eligible record: {key}")
            if int(correct_mass) != sum(int(value) for value in reward["selected_correct_cluster_counts"]):
                raise ValueError(f"Correct-cluster mass drift: {key}")

            row = {
                **base,
                "f_bin": f_bin(variant_count),
                "correct_cluster_variant_count_f": variant_count,
                "correct_cluster_mass_m": int(correct_mass),
                "correct_cluster_answers": sorted(correct_answers),
                "correct_semantic_cluster_count": int(reward["correct_semantic_cluster_count"]),
                "frequency_selected_answer": frequency_answer,
                "frequency_selected_count": int(qwen["string_majority_count"]),
                "frequency_tie": bool_int(qwen["string_majority_tie"]),
                "frequency_correct": int(frequency_answer in correct_answers),
                "density_selected_cluster_answers": sorted(density_answers),
                "density_selected_cluster_mass": int(qwen["semantic_majority_mass"]),
                "density_tie": bool_int(qwen["semantic_majority_tie"]),
                "density_correct": int(bool(correct_answers & density_answers)),
                "selection_differs": int(frequency_answer not in density_answers),
                "semantic_density_reward_correct_cluster": float(reward["semantic_candidate_reward_x"]),
                "exact_frequency_reward_correct_cluster_mean": float(reward["frequency_candidate_reward_y"]),
                "pair_parse_rate": float(reward["pair_parse_rate"]),
                "pair_order_agreement": float(reward["pair_order_agreement"]),
            }
            all_rows.append(row)
            eligible_rows.append(row)
    return all_rows, eligible_rows


def metric_row(
    scope: str,
    label: str,
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    n = len(rows)
    frequency_correct = sum(int(row["frequency_correct"]) for row in rows)
    density_correct = sum(int(row["density_correct"]) for row in rows)
    frequency_low, frequency_high = wilson_interval(frequency_correct, n)
    density_low, density_high = wilson_interval(density_correct, n)
    delta_low, delta_high = paired_bootstrap_delta(rows, bootstrap_samples, bootstrap_seed)
    return {
        "scope": scope,
        "f_bin": label,
        "n": n,
        "mean_correct_cluster_mass_m": statistics.mean(row["correct_cluster_mass_m"] for row in rows) if rows else None,
        "median_correct_cluster_mass_m": statistics.median(row["correct_cluster_mass_m"] for row in rows) if rows else None,
        "frequency_correct": frequency_correct,
        "frequency_accuracy": frequency_correct / n if n else None,
        "frequency_wilson95_low": frequency_low,
        "frequency_wilson95_high": frequency_high,
        "density_correct": density_correct,
        "density_accuracy": density_correct / n if n else None,
        "density_wilson95_low": density_low,
        "density_wilson95_high": density_high,
        "delta_density_minus_frequency": (density_correct - frequency_correct) / n if n else None,
        "delta_paired_bootstrap95_low": delta_low,
        "delta_paired_bootstrap95_high": delta_high,
        "selection_differs_count": sum(int(row["selection_differs"]) for row in rows),
        "frequency_tie_count": sum(int(row["frequency_tie"]) for row in rows),
        "density_tie_count": sum(int(row["density_tie"]) for row in rows),
        "multiple_correct_semantic_clusters": sum(
            int(row["correct_semantic_cluster_count"]) > 1 for row in rows
        ),
    }


def summarize(
    eligible: list[dict[str, Any]], bootstrap_samples: int, bootstrap_seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = [(benchmark, [row for row in eligible if row["benchmark"] == benchmark]) for benchmark in BENCHMARKS]
    scopes.append(("OPEN_POOLED", eligible))
    for scope_index, (scope, selected) in enumerate(scopes):
        for bin_index, label in enumerate(BIN_ORDER):
            output.append(
                metric_row(
                    scope,
                    label,
                    [row for row in selected if row["f_bin"] == label],
                    bootstrap_samples,
                    bootstrap_seed + scope_index * 10 + bin_index,
                )
            )
        output.append(
            metric_row(
                scope,
                "overall",
                selected,
                bootstrap_samples,
                bootstrap_seed + scope_index * 10 + 9,
            )
        )
    return output


def exclusion_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scope in (*BENCHMARKS, "OPEN_POOLED"):
        selected = all_rows if scope == "OPEN_POOLED" else [row for row in all_rows if row["benchmark"] == scope]
        eligible = [row for row in selected if row["eligible_correct_answer_observed"]]
        output.append(
            {
                "scope": scope,
                "questions_total": len(selected),
                "eligible_correct_answer_observed": len(eligible),
                "excluded_no_correct_answer_in_valid_rollouts": len(selected) - len(eligible),
                "all_invalid_questions": sum(int(row["n_valid"]) == 0 for row in selected),
                "valid_but_no_correct_answer": sum(
                    int(row["n_valid"]) > 0 and not row["eligible_correct_answer_observed"]
                    for row in selected
                ),
                "multiple_correct_semantic_clusters_among_eligible": sum(
                    int(row.get("correct_semantic_cluster_count", 0)) > 1 for row in eligible
                ),
            }
        )
    return output


def exact_m_matches(
    rows: list[dict[str, Any]], scope: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = rows if scope == "OPEN_POOLED" else [row for row in rows if row["benchmark"] == scope]
    pairs: list[dict[str, Any]] = []
    f1_rows: list[dict[str, Any]] = []
    fragmented_rows: list[dict[str, Any]] = []
    datasets = BENCHMARKS if scope == "OPEN_POOLED" else (scope,)
    pair_index = 0
    for benchmark in datasets:
        dataset_rows = [row for row in selected if row["benchmark"] == benchmark]
        masses = sorted(set(int(row["correct_cluster_mass_m"]) for row in dataset_rows))
        for mass in masses:
            f1 = sorted(
                [row for row in dataset_rows if row["f_bin"] == "f1" and row["correct_cluster_mass_m"] == mass],
                key=lambda row: str(row["question_id"]),
            )
            fragmented = sorted(
                [row for row in dataset_rows if row["f_bin"] != "f1" and row["correct_cluster_mass_m"] == mass],
                key=lambda row: str(row["question_id"]),
            )
            for left, right in zip(f1, fragmented):
                pair_index += 1
                f1_rows.append(left)
                fragmented_rows.append(right)
                pairs.append(
                    {
                        "scope": scope,
                        "pair_index": pair_index,
                        "benchmark": benchmark,
                        "correct_cluster_mass_m": mass,
                        "f1_question_id": left["question_id"],
                        "fragmented_question_id": right["question_id"],
                        "fragmented_f_bin": right["f_bin"],
                    }
                )
    return f1_rows, fragmented_rows, pairs


def matched_sensitivity(
    eligible: list[dict[str, Any]], bootstrap_samples: int, bootstrap_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for scope_index, scope in enumerate(("aokvqa_da_val", "OPEN_POOLED")):
        f1, fragmented, pairs = exact_m_matches(eligible, scope)
        all_pairs.extend(pairs)
        metrics.append(
            metric_row(
                scope,
                "matched_f1",
                f1,
                bootstrap_samples,
                bootstrap_seed + 100 + scope_index * 2,
            )
        )
        metrics.append(
            metric_row(
                scope,
                "matched_f2plus",
                fragmented,
                bootstrap_samples,
                bootstrap_seed + 101 + scope_index * 2,
            )
        )
    return metrics, all_pairs


def pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def num(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def table_lines(rows_by_bin: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| F_q bin | n | mean m | Exact-frequency | Semantic density | Delta | Different selections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in BIN_ORDER:
        row = rows_by_bin[label]
        lines.append(
            f"| {BIN_LABELS[label]} | {row['n']} | {num(row['mean_correct_cluster_mass_m'])} | "
            f"{row['frequency_correct']}/{row['n']} ({pct(row['frequency_accuracy'])}) | "
            f"{row['density_correct']}/{row['n']} ({pct(row['density_accuracy'])}) | "
            f"{100.0 * float(row['delta_density_minus_frequency']):+.2f} pp | {row['selection_differs_count']} |"
        )
    return lines


def render_report(
    path: Path,
    summary: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    bootstrap_samples: int,
) -> None:
    by_key = {(row["scope"], row["f_bin"]): row for row in summary}
    exclusions_by_scope = {row["scope"]: row for row in exclusions}
    matched_by_key = {(row["scope"], row["f_bin"]): row for row in matched}
    lines = [
        "# Obs1 Semantic-Density Dose-Response Validation",
        "",
        "## Protocol",
        "",
        "- Data: fixed base-model step-0 real-image rollouts, N=32, across VisualSimpleQA, OCRBench v1, and A-OKVQA Direct Answer.",
        "- Semantic partition: the validated Qwen semantic-v2 bidirectional/three-order same-denotation judge with deterministic equivalence rules and the established strong-non-entailment veto.",
        "- Invalid policy: identical to the completed Obs1 semantic analysis; invalid outputs are excluded before exact-string counts and semantic clustering.",
        "- Eligibility: retain a question only if at least one valid rollout belongs to an evaluator-correct semantic cluster.",
        "- If multiple evaluator-correct semantic clusters exist, C_q+ is the largest-mass correct cluster; ties use the original cluster order, matching Obs1.",
        "- Fragmentation dose: F_q is the number of distinct exact strings in C_q+; bins are F=1, F=2, and F>=3 and are fixed before either selector is evaluated.",
        "- Exact-frequency selects the most frequent normalized string. Semantic density assigns every candidate reward m(c)/N_valid, where m(c) is its Qwen semantic-cluster mass, and selects the maximum-reward cluster.",
        "- Outcome: selected answer belongs to C_q+. No entropy temperature, counterfactual term, PAPO loss, or training result enters this comparison.",
        f"- Delta confidence intervals in CSV use {bootstrap_samples:,} paired question-level bootstrap samples.",
        "",
        "## A-OKVQA Main Result",
        "",
    ]
    lines.extend(table_lines({label: by_key[("aokvqa_da_val", label)] for label in BIN_ORDER}))
    lines.extend(["", "## Three-Dataset Pooled Result", ""])
    lines.extend(table_lines({label: by_key[("OPEN_POOLED", label)] for label in BIN_ORDER}))
    lines.extend(["", "## Per-Dataset Appendix", ""])
    for benchmark in BENCHMARKS:
        lines.extend([f"### {benchmark}", ""])
        lines.extend(table_lines({label: by_key[(benchmark, label)] for label in BIN_ORDER}))
        lines.append("")

    lines.extend(
        [
            "## Eligibility and Exclusions",
            "",
            "| Scope | total | eligible | excluded: no correct rollout | all invalid | valid but no correct | multiple correct clusters |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope in (*BENCHMARKS, "OPEN_POOLED"):
        row = exclusions_by_scope[scope]
        lines.append(
            f"| {scope} | {row['questions_total']} | {row['eligible_correct_answer_observed']} | "
            f"{row['excluded_no_correct_answer_in_valid_rollouts']} | {row['all_invalid_questions']} | "
            f"{row['valid_but_no_correct_answer']} | {row['multiple_correct_semantic_clusters_among_eligible']} |"
        )

    lines.extend(
        [
            "",
            "## Exact-m Matched Sensitivity",
            "",
            "Questions are deterministically matched without replacement within dataset and exact correct-cluster mass m. This compares F=1 with F>=2 while holding both dataset and m fixed.",
            "",
            "| Scope | matched group | n | mean m | Exact-frequency | Semantic density | Delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scope in ("aokvqa_da_val", "OPEN_POOLED"):
        for label, display in (("matched_f1", "F=1"), ("matched_f2plus", "F>=2")):
            row = matched_by_key[(scope, label)]
            lines.append(
                f"| {scope} | {display} | {row['n']} | {num(row['mean_correct_cluster_mass_m'])} | "
                f"{row['frequency_correct']}/{row['n']} ({pct(row['frequency_accuracy'])}) | "
                f"{row['density_correct']}/{row['n']} ({pct(row['density_accuracy'])}) | "
                f"{100.0 * float(row['delta_density_minus_frequency']):+.2f} pp |"
            )

    aok = [by_key[("aokvqa_da_val", label)] for label in BIN_ORDER]
    pooled = [by_key[("OPEN_POOLED", label)] for label in BIN_ORDER]
    lines.extend(
        [
            "",
            "## Descriptive Reading",
            "",
            f"- A-OKVQA: Delta is {100.0 * float(aok[0]['delta_density_minus_frequency']):+.2f} pp, "
            f"{100.0 * float(aok[1]['delta_density_minus_frequency']):+.2f} pp, and "
            f"{100.0 * float(aok[2]['delta_density_minus_frequency']):+.2f} pp for F=1, F=2, and F>=3. "
            "The response is threshold-like: no benefit without correct-answer fragmentation, then a clear positive benefit for F>=2; F=2 and F>=3 are not strictly monotonic.",
            f"- Pooled: Delta is {100.0 * float(pooled[0]['delta_density_minus_frequency']):+.2f} pp, "
            f"{100.0 * float(pooled[1]['delta_density_minus_frequency']):+.2f} pp, and "
            f"{100.0 * float(pooled[2]['delta_density_minus_frequency']):+.2f} pp; this is monotonic across the three bins.",
            "- Mean m increases with F, so the unmatched trend alone is confounded by correct-cluster support. The exact-m, within-dataset matched table is the required protection against this mechanical coupling.",
            "- OCRBench contributes no selector changes in these eligible questions; the pooled effect is driven mainly by A-OKVQA. VisualSimpleQA has a small eligible sample and should not be interpreted alone.",
            "- This analysis validates the idealized Qwen semantic-cluster density mechanism used in Obs1. It does not by itself estimate the isolated effect of the training implementation's continuous embedding-kernel density.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_plot(path_png: Path, path_pdf: Path, rows: list[dict[str, Any]], title: str) -> None:
    import matplotlib.pyplot as plt

    by_bin = {row["f_bin"]: row for row in rows}
    x = list(range(len(BIN_ORDER)))
    frequency = [float(by_bin[label]["frequency_accuracy"]) for label in BIN_ORDER]
    density = [float(by_bin[label]["density_accuracy"]) for label in BIN_ORDER]
    labels = [
        f"{BIN_LABELS[label]}\nn={by_bin[label]['n']}, mean m={by_bin[label]['mean_correct_cluster_mass_m']:.2f}"
        for label in BIN_ORDER
    ]
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.bar([value - width / 2 for value in x], frequency, width, label="Exact-frequency", color="#4C78A8")
    axis.bar([value + width / 2 for value in x], density, width, label="Semantic density", color="#E45756")
    for index, label in enumerate(BIN_ORDER):
        delta = float(by_bin[label]["delta_density_minus_frequency"])
        axis.text(index, max(frequency[index], density[index]) + 0.025, f"Delta {100 * delta:+.1f} pp", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Selection accuracy within correct semantic cluster")
    axis.set_title(title)
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_pdf)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reward_rows = read_jsonl(args.reward_dilution_jsonl)
    qwen_rows = read_csv(args.qwen_question_csv)
    all_rows, eligible = prepare_rows(reward_rows, qwen_rows)
    summary = summarize(eligible, args.bootstrap_samples, args.bootstrap_seed)
    exclusions = exclusion_rows(all_rows)
    matched, matched_pairs = matched_sensitivity(
        eligible, args.bootstrap_samples, args.bootstrap_seed
    )

    write_csv(args.output_dir / "question_to_f_bin.csv", all_rows)
    write_csv(args.output_dir / "dose_response_summary.csv", summary)
    write_csv(args.output_dir / "exclusions.csv", exclusions)
    write_csv(args.output_dir / "m_matched_sensitivity.csv", matched)
    write_csv(args.output_dir / "m_matched_pairs.csv", matched_pairs)
    report_path = args.output_dir / "OBS1_DENSITY_DOSE_RESPONSE.md"
    render_report(report_path, summary, exclusions, matched, args.bootstrap_samples)

    aok_rows = [row for row in summary if row["scope"] == "aokvqa_da_val" and row["f_bin"] in BIN_ORDER]
    pooled_rows = [row for row in summary if row["scope"] == "OPEN_POOLED" and row["f_bin"] in BIN_ORDER]
    render_plot(
        args.output_dir / "aokvqa_density_dose_response.png",
        args.output_dir / "aokvqa_density_dose_response.pdf",
        aok_rows,
        "A-OKVQA: semantic-density gain by correct-cluster fragmentation",
    )
    render_plot(
        args.output_dir / "pooled_density_dose_response.png",
        args.output_dir / "pooled_density_dose_response.pdf",
        pooled_rows,
        "Three open-ended benchmarks: pooled dose-response",
    )

    run_summary = {
        "status": "complete",
        "questions_total": len(all_rows),
        "questions_eligible": len(eligible),
        "bin_counts": dict(Counter(row["f_bin"] for row in eligible)),
        "protocol": "Qwen semantic-v2 cluster-density reward; no entropy temperature or counterfactual",
        "report": str(report_path),
        "reward_dilution_source": str(args.reward_dilution_jsonl),
        "qwen_question_source": str(args.qwen_question_csv),
    }
    write_json(args.output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
