#!/usr/bin/env python3
"""Normalize and validate raw selection groups into diagnostic Stage A dumps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

VERL_ROOT = Path(__file__).resolve().parents[1]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import _extract_answers, _normalize_choice_answer


CONDITIONS = ("real", "shuffle", "blank")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def _flat_record_to_group(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": record.get("data_source"),
        "ground_truth": record.get("ground_truth"),
        "prompt": record.get("prompt"),
        "extra_info": record.get("extra_info"),
        "samples": [
            {
                "response_raw": record.get("response_raw"),
                "response_token_len": record.get("response_token_len"),
            }
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _task_from_data_source(data_source: str) -> str:
    if data_source == "GPQA-TTT":
        return "gpqa"
    if data_source in {"VQA-DA-TTT", "VisualSimpleQA-TTT", "AOKVQA-DA-TTT"}:
        return "vqa_da"
    if data_source in {"OCR-TTT", "OCRBench-TTT"}:
        return "ocr"
    raise ValueError(f"Unsupported data_source={data_source!r}")


def _references(extra: dict[str, Any], ground_truth: Any) -> list[str]:
    for key in ("official_answers", "direct_answers", "answers"):
        value = extra.get(key)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if value is not None and str(value).strip():
            return [str(value)]
    return [str(ground_truth)]


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _normalize_group(
    group: dict[str, Any],
    *,
    benchmark: str,
    parser_mode: str,
    choice_labels: str,
    expected_samples: int,
    max_tokens: int,
) -> tuple[str, str, int, list[dict[str, Any]]]:
    extra = group.get("extra_info")
    if not isinstance(extra, dict):
        raise ValueError("Group is missing dict extra_info")
    if extra.get("diagnostic_benchmark") != benchmark:
        raise ValueError(f"Unexpected benchmark metadata: {extra.get('diagnostic_benchmark')!r}")

    condition = str(extra.get("diagnostic_condition") or "")
    if condition not in CONDITIONS:
        raise ValueError(f"Invalid condition={condition!r}")
    question_id = str(extra.get("diagnostic_question_id") or "")
    if not question_id:
        raise ValueError("Missing diagnostic_question_id")
    subset_position = int(extra.get("diagnostic_subset_position"))

    samples = group.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(
            f"{benchmark}/{condition}/{question_id}: expected {expected_samples} samples, "
            f"got {len(samples) if isinstance(samples, list) else 'non-list'}"
        )

    raw_texts = [str(sample.get("response_raw") or "") for sample in samples]
    ground_truth = group.get("ground_truth")
    data_source = str(group.get("data_source") or "")
    task = _task_from_data_source(data_source)
    parsed_answers = _extract_answers(
        task,
        raw_texts,
        ground_truth,
        extra_info=[extra] * expected_samples,
        answer_parse_mode=parser_mode,
        choice_labels=choice_labels,
    )

    if task == "gpqa":
        normalized_ground_truth = _normalize_choice_answer(ground_truth, choice_labels=choice_labels)
        exact = [
            float(_normalize_choice_answer(answer, choice_labels=choice_labels) == normalized_ground_truth)
            for answer in parsed_answers
        ]
        validity = [
            _normalize_choice_answer(answer, choice_labels=choice_labels) != "unknown"
            for answer in parsed_answers
        ]
        verifier_rewards = list(exact)
    else:
        verifier_rewards, verifier_info = auto_verify(
            task,
            raw_texts,
            [ground_truth] * expected_samples,
            extra_info=[extra] * expected_samples,
        )
        exact = [float(value) for value in verifier_info["exact_acc"]]
        validity = [str(answer).strip().lower() not in {"", "unknown"} for answer in parsed_answers]

    target_image = str(extra.get("diagnostic_target_image_path") or "")
    effective_image = str(extra.get("diagnostic_effective_image_path") or "")
    shuffle_src_qid = extra.get("diagnostic_shuffle_src_qid")
    references = _references(extra, ground_truth)
    prompt = group.get("prompt")
    normalized: list[dict[str, Any]] = []
    for sample_index, (sample, raw_text, parsed, is_exact, verifier_reward, is_valid) in enumerate(
        zip(samples, raw_texts, parsed_answers, exact, verifier_rewards, validity)
    ):
        response_token_len = int(sample.get("response_token_len") or 0)
        normalized.append(
            {
                "benchmark": benchmark,
                "question_id": question_id,
                "condition": condition,
                "sample_idx": sample_index,
                "raw_text": raw_text,
                "parsed_answer": str(parsed),
                "is_valid": bool(is_valid),
                "is_correct": int(is_exact > 0.0),
                "verifier_reward": float(verifier_reward),
                "shuffle_src_qid": str(shuffle_src_qid) if shuffle_src_qid is not None else None,
                "prompt": prompt,
                "ground_truth": str(ground_truth),
                "official_answers": references,
                "target_image_path": target_image,
                "effective_image_path": effective_image,
                "response_token_len": response_token_len,
                "hit_max_tokens": bool(response_token_len >= max_tokens),
                "parser_mode": parser_mode,
                "answer_choice_labels": choice_labels,
                "data_source": data_source,
                "subset_position": subset_position,
            }
        )
    return condition, question_id, subset_position, normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--parser-mode", required=True)
    parser.add_argument("--choice-labels", required=True)
    parser.add_argument("--expected-questions", type=int, default=250)
    parser.add_argument("--expected-samples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--input-format", choices=("groups", "flat"), default="groups")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--integrity-name", default="integrity.json")
    args = parser.parse_args()

    selected_conditions = tuple(args.conditions)
    if not args.integrity_name.endswith(".json") or Path(args.integrity_name).name != args.integrity_name:
        raise ValueError("integrity-name must be a JSON basename")

    records_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qids_by_condition: dict[str, set[str]] = defaultdict(set)
    group_counts = Counter()
    condition_group_keys: set[tuple[str, str]] = set()

    source_records = _read_jsonl(args.groups)
    if args.input_format == "flat":
        source_records = (_flat_record_to_group(record) for record in source_records)

    for group in source_records:
        condition, question_id, _, records = _normalize_group(
            group,
            benchmark=args.benchmark,
            parser_mode=args.parser_mode,
            choice_labels=args.choice_labels,
            expected_samples=args.expected_samples,
            max_tokens=args.max_tokens,
        )
        if condition not in selected_conditions:
            raise ValueError(f"Unexpected condition={condition!r}; expected {selected_conditions}")
        key = (condition, question_id)
        if key in condition_group_keys:
            raise ValueError(f"Duplicate condition/question group: {key}")
        condition_group_keys.add(key)
        qids_by_condition[condition].add(question_id)
        records_by_condition[condition].extend(records)
        group_counts[condition] += 1

    for condition in selected_conditions:
        if group_counts[condition] != args.expected_questions:
            raise ValueError(
                f"{args.benchmark}/{condition}: expected {args.expected_questions} groups, "
                f"got {group_counts[condition]}"
            )
    reference_qids = qids_by_condition[selected_conditions[0]]
    for condition in selected_conditions[1:]:
        if qids_by_condition[condition] != reference_qids:
            raise ValueError(f"{args.benchmark}: question set differs for {condition}")

    dump_root = args.output_root / "dumps" / args.benchmark
    expected_records = args.expected_questions * args.expected_samples
    condition_summaries = {}
    for condition in selected_conditions:
        records = records_by_condition[condition]
        if len(records) != expected_records:
            raise ValueError(f"{args.benchmark}/{condition}: expected {expected_records} records, got {len(records)}")
        records.sort(key=lambda item: (item["subset_position"], item["sample_idx"]))
        _atomic_jsonl(dump_root / f"{condition}.jsonl", records)
        condition_summaries[condition] = {
            "groups": group_counts[condition],
            "records": len(records),
            "empty_raw_text": sum(not item["raw_text"] for item in records),
            "invalid_predictions": sum(not bool(item["is_valid"]) for item in records),
            "hit_max_tokens": sum(bool(item["hit_max_tokens"]) for item in records),
        }

    integrity = {
        "benchmark": args.benchmark,
        "status": "complete",
        "expected_questions_per_condition": args.expected_questions,
        "expected_samples_per_question": args.expected_samples,
        "unique_question_ids": len(reference_qids),
        "conditions": condition_summaries,
        "input_format": args.input_format,
        "source_groups": str(args.groups),
    }
    _write_json(dump_root / args.integrity_name, integrity)
    print(json.dumps(integrity, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
