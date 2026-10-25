#!/usr/bin/env python3
"""Obs1 semantic-fragmentation analysis for diagnostic Stage A dumps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForSequenceClassification, AutoTokenizer


VERL_ROOT = Path(__file__).resolve().parents[1]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score.ttrl.direct_answer import (
    UNKNOWN_LABEL,
    coerce_answer_list,
    normalize_direct_answer,
)


BENCHMARKS = (
    "mme20",
    "ai2d20",
    "seed20",
    "mathvista20_hq",
    "mathverse20_fixed",
    "realworldqa20_hq",
    "visualsimpleqa",
    "ocrbench_v1",
    "aokvqa_da_val",
)
OPEN_BENCHMARKS = {"visualsimpleqa", "ocrbench_v1", "aokvqa_da_val"}
Z_95 = 1.959963984540054


SHORT_ANSWER_CONTROLS = (
    ("2", "two", True, "number_alias"),
    ("4", "four", True, "number_alias"),
    ("0", "zero", True, "number_alias"),
    ("1,000", "1000", True, "numeric_format"),
    ("50 percent", "50%", True, "numeric_format"),
    ("first", "1st", True, "ordinal_alias"),
    ("five dollars", "$5", True, "currency_alias"),
    ("april 4 2018", "4 april 2018", True, "date_format"),
    ("dog", "a dog", True, "article_variant"),
    ("red", "the color red", True, "description_variant"),
    ("bicycle", "bike", True, "lexical_alias"),
    ("television", "tv", True, "abbreviation"),
    ("united states", "usa", True, "abbreviation"),
    ("new york city", "nyc", True, "abbreviation"),
    ("yes", "yeah", True, "affirmation_alias"),
    ("no", "nope", True, "negation_alias"),
    ("2", "3", False, "number_contrast"),
    ("yes", "no", False, "binary_contrast"),
    ("red", "blue", False, "color_contrast"),
    ("dog", "cat", False, "object_contrast"),
    ("left", "right", False, "direction_contrast"),
    ("indoor", "outdoor", False, "scene_contrast"),
    ("2018", "2019", False, "date_contrast"),
    ("50%", "60%", False, "numeric_contrast"),
    ("morning", "evening", False, "time_contrast"),
    ("car", "bus", False, "object_contrast"),
    ("true", "false", False, "binary_contrast"),
    ("up", "down", False, "direction_contrast"),
    ("man", "person", False, "one_way_entailment"),
    ("poodle", "dog", False, "one_way_entailment"),
    ("two dogs", "dogs", False, "one_way_entailment"),
    ("red car", "car", False, "one_way_entailment"),
)


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    z2 = Z_95 * Z_95
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    spread = Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def proportion(successes: int, total: int, prefix: str) -> dict[str, Any]:
    low, high = wilson(successes, total)
    return {
        f"{prefix}_successes": successes,
        f"{prefix}_n": total,
        f"{prefix}_rate": successes / total if total else None,
        f"{prefix}_ci95_low": low,
        f"{prefix}_ci95_high": high,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def canonical(value: Any) -> str | None:
    answer = normalize_direct_answer(value)
    return None if answer == UNKNOWN_LABEL else answer


def load_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            groups[str(raw["question_id"])].append(
                {
                    "sample_idx": int(raw["sample_idx"]),
                    "parsed_answer": str(raw.get("parsed_answer") or ""),
                    "is_valid": bool(raw.get("is_valid")),
                    "is_correct": int(raw.get("is_correct") or 0),
                    "ground_truth": raw.get("ground_truth"),
                    "official_answers": raw.get("official_answers"),
                    "raw_text": str(raw.get("raw_text") or ""),
                }
            )
    for records in groups.values():
        records.sort(key=lambda row: row["sample_idx"])
        if [row["sample_idx"] for row in records] != list(range(32)):
            raise ValueError("Expected exactly sample_idx=0..31 for every question")
    return groups


def build_question_data(experiment_root: Path) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    questions: dict[str, dict[str, Any]] = {}
    pairs: set[tuple[str, str]] = set()
    for benchmark in BENCHMARKS:
        groups = load_groups(experiment_root / "dumps" / benchmark / "real.jsonl")
        if len(groups) != 250:
            raise ValueError(f"{benchmark}: expected 250 questions, got {len(groups)}")
        for qid, records in groups.items():
            answers: list[str] = []
            answer_correct: dict[str, int] = {}
            for record in records:
                answer = canonical(record["parsed_answer"]) if record["is_valid"] else None
                if answer is None:
                    continue
                if answer not in answer_correct:
                    answers.append(answer)
                    answer_correct[answer] = 0
                answer_correct[answer] = max(answer_correct[answer], record["is_correct"])
            counter = Counter(
                canonical(record["parsed_answer"])
                for record in records
                if record["is_valid"] and canonical(record["parsed_answer"]) is not None
            )
            references: list[str] = []
            source_refs = records[0].get("official_answers")
            if not source_refs:
                source_refs = coerce_answer_list(records[0].get("ground_truth"))
            for value in coerce_answer_list(source_refs):
                reference = canonical(value)
                if reference is not None and reference not in references:
                    references.append(reference)
            if benchmark in OPEN_BENCHMARKS:
                for left_idx, left in enumerate(answers):
                    for right in answers[left_idx + 1 :]:
                        if left != right:
                            pairs.add((left, right))
                            pairs.add((right, left))
                    for reference in references:
                        if left != reference:
                            pairs.add((left, reference))
                            pairs.add((reference, left))
            questions[f"{benchmark}\t{qid}"] = {
                "benchmark": benchmark,
                "question_id": qid,
                "counter": counter,
                "answers": answers,
                "answer_correct": answer_correct,
                "references": references,
                "n_valid": sum(counter.values()),
            }
    return questions, pairs


def is_short_answer(value: str) -> bool:
    value = str(value).strip()
    return bool(value) and len(value) <= 64 and len(value.split()) <= 8


def build_short_answer_pilot_cases(
    questions: dict[str, dict[str, Any]],
    max_actual_per_class: int = 20,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "source": "control",
            "benchmark": "control",
            "question_id": "",
            "left": left,
            "right": right,
            "expected_equivalent": bool(expected),
            "category": category,
        }
        for left, right, expected, category in SHORT_ANSWER_CONTROLS
    ]
    seen: set[tuple[str, str, str, bool]] = set()
    counts: Counter[tuple[str, bool]] = Counter()

    def append_actual(
        benchmark: str,
        question_id: str,
        left: str,
        right: str,
        expected: bool,
        category: str,
    ) -> None:
        if left == right or not is_short_answer(left) or not is_short_answer(right):
            return
        count_key = (benchmark, expected)
        if counts[count_key] >= max_actual_per_class:
            return
        dedupe_key = (benchmark, min(left, right), max(left, right), expected)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        counts[count_key] += 1
        cases.append(
            {
                "source": "actual_rollout",
                "benchmark": benchmark,
                "question_id": question_id,
                "left": left,
                "right": right,
                "expected_equivalent": expected,
                "category": category,
            }
        )

    for key in sorted(questions):
        question = questions[key]
        benchmark = str(question["benchmark"])
        if benchmark not in OPEN_BENCHMARKS:
            continue
        answer_correct = question["answer_correct"]
        correct = sorted(answer for answer, value in answer_correct.items() if value and is_short_answer(answer))
        incorrect = sorted(answer for answer, value in answer_correct.items() if not value and is_short_answer(answer))
        references = sorted(reference for reference in question["references"] if is_short_answer(reference))

        for left_idx, left in enumerate(correct):
            for right in correct[left_idx + 1 :]:
                append_actual(benchmark, question["question_id"], left, right, True, "two_evaluator_correct")
            for reference in references:
                append_actual(benchmark, question["question_id"], left, reference, True, "correct_to_reference")
            for right in incorrect:
                append_actual(benchmark, question["question_id"], left, right, False, "correct_vs_incorrect")
        for reference in references:
            for right in incorrect:
                append_actual(benchmark, question["question_id"], reference, right, False, "reference_vs_incorrect")

    return cases


def pilot_pairs(cases: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for case in cases:
        left = str(case["left"])
        right = str(case["right"])
        pairs.add((left, right))
        pairs.add((right, left))
    return pairs


def distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("Obs1 NLI pass requires a CUDA worker")
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def format_nli_text(value: str, input_style: str) -> str:
    value = str(value).strip()
    if input_style == "raw":
        return value
    if input_style == "answer_sentence":
        return f"The answer is {value.rstrip('.')}."
    raise ValueError(f"Unsupported NLI input style: {input_style}")


def rank_output_path(output_path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")


def infer_pairs(
    pairs: set[tuple[str, str]],
    model_path: Path,
    batch_size: int,
    max_length: int,
    output_path: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    input_style: str,
) -> dict[tuple[str, str], tuple[bool, float, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval().to(device)
    label2id = {str(key).upper(): int(value) for key, value in model.config.label2id.items()}
    entailment_id = label2id.get("ENTAILMENT")
    if entailment_id is None:
        raise ValueError(f"Missing ENTAILMENT label: {model.config.label2id}")

    ordered_all = sorted(pairs)
    ordered = ordered_all[rank::world_size]
    predictions: dict[tuple[str, str], tuple[bool, float, int]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_output_path = rank_output_path(output_path, rank, world_size)
    temporary = local_output_path.with_suffix(local_output_path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset : offset + batch_size]
            model_premises = [format_nli_text(pair[0], input_style) for pair in batch]
            model_hypotheses = [format_nli_text(pair[1], input_style) for pair in batch]
            encoded = tokenizer(
                model_premises,
                model_hypotheses,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device=device, non_blocking=True) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            probabilities = torch.softmax(logits, dim=-1)
            labels = logits.argmax(dim=-1)
            for pair, label, probability in zip(batch, labels.tolist(), probabilities[:, entailment_id].tolist()):
                result = (label == entailment_id, float(probability), int(label))
                predictions[pair] = result
                handle.write(
                    json.dumps(
                        {
                            "premise": pair[0],
                            "hypothesis": pair[1],
                            "model_premise": format_nli_text(pair[0], input_style),
                            "model_hypothesis": format_nli_text(pair[1], input_style),
                            "input_style": input_style,
                            "predicted_label_id": result[2],
                            "entailment_probability": result[1],
                            "is_entailment": result[0],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if offset == 0 or (offset // batch_size + 1) % 50 == 0:
                print(
                    f"[nli rank={rank}] {min(offset + len(batch), len(ordered))}/{len(ordered)} "
                    f"(global={len(ordered_all)})",
                    flush=True,
                )
    temporary.replace(local_output_path)
    return predictions


def merge_pair_outputs(
    output_path: Path,
    world_size: int,
) -> dict[tuple[str, str], tuple[bool, float, int]]:
    predictions: dict[tuple[str, str], tuple[bool, float, int]] = {}
    rows: list[dict[str, Any]] = []
    paths = [rank_output_path(output_path, rank, world_size) for rank in range(world_size)]
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pair = (str(row["premise"]), str(row["hypothesis"]))
                predictions[pair] = (
                    bool(row["is_entailment"]),
                    float(row["entailment_probability"]),
                    int(row["predicted_label_id"]),
                )
                rows.append(row)
    if len(predictions) != len(rows):
        raise ValueError("Duplicate directional NLI pairs found while merging distributed shards")
    rows.sort(key=lambda row: (str(row["premise"]), str(row["hypothesis"])))
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    if world_size > 1:
        for path in paths:
            path.unlink()
    return predictions


def evaluate_short_answer_pilot(
    cases: list[dict[str, Any]],
    predictions: dict[tuple[str, str], tuple[bool, float, int]],
    input_style: str,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        left = str(case["left"])
        right = str(case["right"])
        forward = predictions[(left, right)]
        reverse = predictions[(right, left)]
        bidirectional = bool(forward[0] and reverse[0])
        expected = bool(case["expected_equivalent"])
        rows.append(
            {
                **case,
                "input_style": input_style,
                "forward_entailment": bool(forward[0]),
                "reverse_entailment": bool(reverse[0]),
                "forward_entailment_probability": float(forward[1]),
                "reverse_entailment_probability": float(reverse[1]),
                "bidirectional_entailment": bidirectional,
                "directional_disagreement": int(bool(forward[0]) != bool(reverse[0])),
                "prediction_correct": int(bidirectional == expected),
            }
        )

    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        positives = [row for row in selected if row["expected_equivalent"]]
        negatives = [row for row in selected if not row["expected_equivalent"]]
        correct = sum(int(row["prediction_correct"]) for row in selected)
        true_positive = sum(int(row["bidirectional_entailment"]) for row in positives)
        true_negative = sum(int(not row["bidirectional_entailment"]) for row in negatives)
        return {
            "n": len(selected),
            **proportion(correct, len(selected), "accuracy"),
            **proportion(true_positive, len(positives), "equivalent_recall"),
            **proportion(true_negative, len(negatives), "non_equivalent_specificity"),
            **proportion(
                sum(int(row["directional_disagreement"]) for row in selected),
                len(selected),
                "directional_disagreement",
            ),
        }

    summary: dict[str, Any] = {
        "input_style": input_style,
        "control": metrics([row for row in rows if row["source"] == "control"]),
        "actual_rollout": metrics([row for row in rows if row["source"] == "actual_rollout"]),
        "actual_by_benchmark": {
            benchmark: metrics(
                [
                    row
                    for row in rows
                    if row["source"] == "actual_rollout" and row["benchmark"] == benchmark
                ]
            )
            for benchmark in sorted(OPEN_BENCHMARKS)
        },
    }
    control = summary["control"]
    summary["gate"] = {
        "accuracy_min": 0.80,
        "equivalent_recall_min": 0.75,
        "non_equivalent_specificity_min": 0.90,
        "passed": bool(
            (control["accuracy_rate"] or 0.0) >= 0.80
            and (control["equivalent_recall_rate"] or 0.0) >= 0.75
            and (control["non_equivalent_specificity_rate"] or 0.0) >= 0.90
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pilot_cases.csv", rows)
    write_jsonl(output_dir / "pilot_cases.jsonl", rows)
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Obs1 short-answer bidirectional-entailment pilot",
        "",
        f"- Input style: {input_style}",
        f"- Control accuracy: {fmt_rate(control['accuracy_rate'])} "
        f"({control['accuracy_successes']}/{control['accuracy_n']})",
        f"- Equivalent recall: {fmt_rate(control['equivalent_recall_rate'])} "
        f"({control['equivalent_recall_successes']}/{control['equivalent_recall_n']})",
        f"- Non-equivalent specificity: {fmt_rate(control['non_equivalent_specificity_rate'])} "
        f"({control['non_equivalent_specificity_successes']}/{control['non_equivalent_specificity_n']})",
        f"- Directional disagreement: {fmt_rate(control['directional_disagreement_rate'])}",
        f"- Gate passed: {summary['gate']['passed']}",
        "",
        "The full 32-rollout semantic clustering pass must not start unless one fixed input style passes this gate.",
    ]
    (output_dir / "pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def analyze_questions(
    questions: dict[str, dict[str, Any]],
    predictions: dict[tuple[str, str], tuple[bool, float, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def entails(left: str, right: str, benchmark: str) -> bool:
        if left == right:
            return True
        if benchmark not in OPEN_BENCHMARKS:
            return False
        return predictions[(left, right)][0]

    def equivalent(left: str, right: str, benchmark: str) -> bool:
        return entails(left, right, benchmark) and entails(right, left, benchmark)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for question in questions.values():
        benchmark = question["benchmark"]
        counter: Counter[str] = question["counter"]
        if not counter:
            rows.append(
                {
                    "benchmark": question["benchmark"],
                    "question_id": question["question_id"],
                    "n_total": 32,
                    "n_valid": 0,
                    "all_invalid": 1,
                }
            )
            continue

        answers = question["answers"]
        clusters: list[list[str]] = []
        for answer in answers:
            for cluster in clusters:
                if equivalent(answer, cluster[0], benchmark):
                    cluster.append(answer)
                    break
            else:
                clusters.append([answer])
        masses = [sum(counter[answer] for answer in cluster) for cluster in clusters]
        semantic_max = max(masses)
        semantic_tie = int(sum(mass == semantic_max for mass in masses) > 1)
        semantic_cluster = clusters[masses.index(semantic_max)]

        string_answer, string_count = counter.most_common(1)[0]
        string_tie = int(sum(count == string_count for count in counter.values()) > 1)
        disagreement = int(string_answer not in semantic_cluster)
        evaluator_correct = question["answer_correct"]
        string_eval_correct = int(evaluator_correct.get(string_answer, 0) > 0)
        semantic_eval_correct = int(any(evaluator_correct.get(answer, 0) > 0 for answer in semantic_cluster))
        references = question["references"]
        nli_correct = {
            answer: int(any(equivalent(answer, reference, benchmark) for reference in references))
            for answer in answers
        }
        string_nli_correct = nli_correct[string_answer]
        semantic_nli_correct = int(any(nli_correct[answer] for answer in semantic_cluster))
        fatal_eval = int(disagreement and not string_eval_correct and semantic_eval_correct)
        fatal_nli = int(disagreement and not string_nli_correct and semantic_nli_correct)
        row = {
            "benchmark": question["benchmark"],
            "question_id": question["question_id"],
            "n_total": 32,
            "n_valid": question["n_valid"],
            "all_invalid": 0,
            "string_cluster_count": len(counter),
            "semantic_cluster_count": len(clusters),
            "fragmentation_ratio": len(counter) / len(clusters),
            "string_majority_answer": string_answer,
            "string_majority_count": string_count,
            "string_majority_tie": string_tie,
            "semantic_majority_answers": json.dumps(semantic_cluster, ensure_ascii=False),
            "semantic_majority_mass": semantic_max,
            "semantic_majority_tie": semantic_tie,
            "disagreement": disagreement,
            "string_eval_correct": string_eval_correct,
            "semantic_eval_correct": semantic_eval_correct,
            "string_nli_correct": string_nli_correct,
            "semantic_nli_correct": semantic_nli_correct,
            "fatal_eval": fatal_eval,
            "fatal_nli": fatal_nli,
        }
        rows.append(row)
        if disagreement or fatal_eval or fatal_nli:
            cases.append(
                {
                    **row,
                    "string_counter": dict(counter),
                    "semantic_clusters": [
                        {"answers": cluster, "mass": mass}
                        for cluster, mass in zip(clusters, masses)
                    ],
                    "official_answers": references,
                }
            )
    return rows, cases


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = [(benchmark, [row for row in rows if row["benchmark"] == benchmark]) for benchmark in BENCHMARKS]
    scopes.append(
        (
            "OPEN_MICRO",
            [row for row in rows if row["benchmark"] in OPEN_BENCHMARKS],
        )
    )
    for name, selected_all in scopes:
        selected = [row for row in selected_all if not row["all_invalid"]]
        string_wrong_eval = [row for row in selected if not row["string_eval_correct"]]
        string_wrong_nli = [row for row in selected if not row["string_nli_correct"]]
        fragmentation = [float(row["fragmentation_ratio"]) for row in selected]
        row: dict[str, Any] = {
            "benchmark": name,
            "questions_total": len(selected_all),
            "questions_valid": len(selected),
            "questions_all_invalid": len(selected_all) - len(selected),
            "string_clusters_mean": statistics.fmean(float(item["string_cluster_count"]) for item in selected),
            "semantic_clusters_mean": statistics.fmean(float(item["semantic_cluster_count"]) for item in selected),
            "fragmentation_mean": statistics.fmean(fragmentation),
            "fragmentation_median": statistics.median(fragmentation),
            **proportion(sum(item["disagreement"] for item in selected), len(selected), "disagreement"),
            **proportion(sum(item["string_eval_correct"] for item in selected), len(selected), "string_eval_accuracy"),
            **proportion(sum(item["semantic_eval_correct"] for item in selected), len(selected), "semantic_eval_accuracy"),
            **proportion(sum(item["string_nli_correct"] for item in selected), len(selected), "string_nli_accuracy"),
            **proportion(sum(item["semantic_nli_correct"] for item in selected), len(selected), "semantic_nli_accuracy"),
            **proportion(sum(item["fatal_eval"] for item in string_wrong_eval), len(string_wrong_eval), "fatal_eval_given_string_wrong"),
            **proportion(sum(item["fatal_nli"] for item in string_wrong_nli), len(string_wrong_nli), "fatal_nli_given_string_wrong"),
            **proportion(sum(item["string_majority_tie"] for item in selected), len(selected), "string_tie"),
            **proportion(sum(item["semantic_majority_tie"] for item in selected), len(selected), "semantic_tie"),
        }
        output.append(row)
    return output


def fmt_rate(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def write_report(path: Path, summary: list[dict[str, Any]], model_path: Path) -> None:
    lines = [
        "# Obs1: Normalized-string fragmentation",
        "",
        f"Model: `{model_path}`. Two answers are merged only when DeBERTa predicts entailment in both directions.",
        "Greedy clustering uses first-appearance order. Invalid outputs are excluded and reported separately.",
        "Evaluator accuracy is the non-circular headline; NLI-reference accuracy is a secondary diagnostic.",
        "",
        "| Benchmark | valid n | all invalid | F mean | disagree | string eval acc | semantic eval acc | fatal / string-wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {benchmark} | {questions_valid} | {questions_all_invalid} | {fragmentation_mean:.3f} | {disagreement} | {string_acc} | {semantic_acc} | {fatal} ({fatal_s}/{fatal_n}) |".format(
                benchmark=row["benchmark"],
                questions_valid=row["questions_valid"],
                questions_all_invalid=row["questions_all_invalid"],
                fragmentation_mean=row["fragmentation_mean"],
                disagreement=fmt_rate(row["disagreement_rate"]),
                string_acc=fmt_rate(row["string_eval_accuracy_rate"]),
                semantic_acc=fmt_rate(row["semantic_eval_accuracy_rate"]),
                fatal=fmt_rate(row["fatal_eval_given_string_wrong_rate"]),
                fatal_s=row["fatal_eval_given_string_wrong_successes"],
                fatal_n=row["fatal_eval_given_string_wrong_n"],
            )
        )
    lines.extend(
        [
            "",
            "`fatal` means the evaluator marks the semantic-majority cluster correct, while the normalized-string majority is wrong and belongs to another cluster.",
            "All proportion confidence intervals are stored in `obs1_fragmentation.csv` as 95% Wilson intervals.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--nli-input-style", choices=("raw", "answer_sentence"), default="raw")
    parser.add_argument("--max-actual-pilot-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--reuse-pair-predictions", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = distributed_context()
    experiment_root = args.experiment_root.resolve()
    output_dir = (args.output_dir or experiment_root / "analysis_stage_b").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    questions, pairs = build_question_data(experiment_root)
    pilot_cases: list[dict[str, Any]] = []
    if args.mode == "pilot":
        pilot_cases = build_short_answer_pilot_cases(
            questions,
            max_actual_per_class=args.max_actual_pilot_per_class,
        )
        pairs = pilot_pairs(pilot_cases)
    if rank == 0:
        print(
            f"[obs1] mode={args.mode} input_style={args.nli_input_style} "
            f"questions={len(questions)} directional_pairs={len(pairs)} world_size={world_size}",
            flush=True,
        )

    pair_output = output_dir / ("pilot_nli_pairs.jsonl" if args.mode == "pilot" else "obs1_nli_pairs.jsonl")
    if args.reuse_pair_predictions:
        if world_size != 1:
            raise ValueError("--reuse-pair-predictions must run as a single process")
        if not pair_output.is_file():
            raise FileNotFoundError(f"Missing cached NLI predictions: {pair_output}")
        predictions = merge_pair_outputs(pair_output, world_size=1)
        if set(predictions) != pairs:
            missing = len(pairs.difference(predictions))
            extra = len(set(predictions).difference(pairs))
            raise ValueError(f"Cached NLI pair mismatch: missing={missing}, extra={extra}")
    else:
        infer_pairs(
            pairs=pairs,
            model_path=args.model.resolve(),
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_path=pair_output,
            rank=rank,
            world_size=world_size,
            device=device,
            input_style=args.nli_input_style,
        )
        if world_size > 1:
            dist.barrier(device_ids=[device.index])
        if rank != 0:
            if dist.is_initialized():
                dist.destroy_process_group()
            return 0
        predictions = merge_pair_outputs(pair_output, world_size)
    if dist.is_initialized():
        dist.destroy_process_group()
    if args.mode == "pilot":
        summary = evaluate_short_answer_pilot(
            pilot_cases,
            predictions,
            input_style=args.nli_input_style,
            output_dir=output_dir,
        )
        print(json.dumps({"status": "complete", "output_dir": str(output_dir), **summary["gate"]}))
        return 0

    question_rows, cases = analyze_questions(questions, predictions)
    summary = summarize(question_rows)
    write_csv(output_dir / "obs1_question_level.csv", question_rows)
    write_csv(output_dir / "obs1_fragmentation.csv", summary)
    write_jsonl(output_dir / "obs1_cases.jsonl", cases)
    write_report(output_dir / "obs1_fragmentation.md", summary, args.model.resolve())
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
