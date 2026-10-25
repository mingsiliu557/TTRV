#!/usr/bin/env python3
"""Build compact Obs2 TTS case folders from the completed Selection@32 run."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path


REPO_ROOT = Path("/workspace/TTRV")
EXPERIMENT_ROOT = Path(
    "/data/ttrv/experiments/step0_reward_selection32_20260711_020518"
)
OUTPUT_ROOT = REPO_ROOT / "obv2_cases"
FIXED_BIN_MAP = (
    REPO_ROOT
    / "outputs/diagnostic_obs2_fixed_bin_selection_20260723/obs2_fixed_pooled_bin_map.csv"
)

POOLED_Q33 = 0.4656663294644227
POOLED_Q67 = 0.7540931063268035
EXPECTED_N = 32

# Reuse completed TTS@32 outputs only. These non-math cases all exceed the
# previous fixed pooled high-entropy threshold; TTRV has a strict valid-but-
# wrong frequency majority and PROVE selects the correct answer.
CASE_SPECS = (
    ("ai2d20", "1653", "01_ai2d_1653"),
    ("realworldqa20_hq", "627", "02_realworldqa_627"),
    ("realworldqa20_hq", "25", "03_realworldqa_25"),
    ("realworldqa20_hq", "348", "04_realworldqa_348"),
    ("realworldqa20_hq", "3", "05_realworldqa_3"),
    ("realworldqa20_hq", "116", "06_realworldqa_116"),
)


def normalized_entropy(counter: dict[str, int]) -> float:
    counts = [int(value) for value in counter.values() if int(value) > 0]
    if len(counts) <= 1:
        return 0.0
    total = sum(counts)
    probabilities = [value / total for value in counts]
    return -sum(value * math.log(value) for value in probabilities) / math.log(len(counts))


def clean_prompt(prompt: str) -> str:
    text = re.sub(r"<IMG_CONTEXT>|<[^>]+>", " ", str(prompt or ""))
    text = " ".join(text.split())
    text = re.sub(r"^user\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+assistant$", "", text, flags=re.IGNORECASE)
    return text.strip()


def option_pairs(text: str, style: str) -> dict[str, str]:
    if style == "dot":
        pattern = r"([A-F])\.\s*(.*?)(?=\s+[A-F]\.\s|$)"
    elif style == "colon":
        pattern = r"([A-F]):\s*(.*?)(?=\s+[A-F]:\s*|$)"
    elif style == "paren":
        pattern = r"\(([A-F])\)\s*(.*?)(?=\s+\([A-F]\)\s|$)"
    else:
        raise ValueError(style)
    return {label: value.strip().rstrip(".") for label, value in re.findall(pattern, text)}


def extract_question_and_options(prompt: str) -> tuple[str, dict[str, str]]:
    text = clean_prompt(prompt)

    option_marker = re.search(r"\s+Options(?:\s+are)?:\s*", text, flags=re.IGNORECASE)
    if option_marker is not None:
        prefix = text[: option_marker.start()]
        option_block = text[option_marker.end() :]
        option_block = re.split(
            r"\s+Please answer(?: directly)?\b",
            option_block,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        options = option_pairs(option_block, "dot") or option_pairs(option_block, "colon")
        question = re.split(
            r"\s*Choose the correct answer from the options below.*",
            prefix,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
    elif "Choices:" in text:
        prefix, option_block = text.split("Choices:", 1)
        options = option_pairs(option_block, "paren") or option_pairs(option_block, "colon")
        question = prefix.rsplit("Question:", 1)[-1].strip()
    else:
        match = re.search(r"\sA[\.:]\s*", text)
        if match is None:
            raise ValueError(f"Could not locate options in prompt: {text[:200]}")
        question = text[: match.start()].strip()
        option_block = text[match.start() + 1 :]
        option_block = re.split(
            r"\s+Please answer(?: directly)?\b",
            option_block,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        options = option_pairs(option_block, "dot") or option_pairs(option_block, "colon")

    question = re.sub(r"^Hint:.*?Question:\s*", "", question, flags=re.IGNORECASE)
    if not question or len(options) < 2:
        raise ValueError(f"Question/options parse failed: question={question!r}, options={options!r}")
    return question, options


def answer_text(options: dict[str, str], label: str) -> str:
    if label not in options:
        raise ValueError(f"Selected label {label!r} absent from options {options!r}")
    return options[label]


def load_fixed_bin_map() -> dict[tuple[str, str], dict]:
    import csv

    result = {}
    with FIXED_BIN_MAP.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["benchmark"]), str(row["question_id"]))
            result[key] = {
                "entropy": float(row["step0_all_output_entropy"]),
                "bin": str(row["h_bin"]),
            }
    return result


def answer_level_reward_gap(record: dict) -> tuple[float, dict[str, float]]:
    label_scores: dict[str, float] = {}
    for sample in record.get("samples") or []:
        label = str(sample.get("prediction"))
        score = float(
            sample.get(
                "final_selection_reward",
                sample.get("selection_reward", sample.get("reward", 0.0)),
            )
        )
        label_scores[label] = max(label_scores.get(label, float("-inf")), score)
    ordered = sorted(label_scores.values(), reverse=True)
    if len(ordered) < 2:
        raise RuntimeError("Need at least two answer labels to compute a reward gap")
    return ordered[0] - ordered[1], label_scores


def load_selected_records() -> dict[tuple[str, str], dict]:
    wanted_by_dataset: dict[str, set[str]] = {}
    for dataset, question_id, _ in CASE_SPECS:
        wanted_by_dataset.setdefault(dataset, set()).add(question_id)

    selected: dict[tuple[str, str], dict] = {}
    for dataset, ids in wanted_by_dataset.items():
        source = EXPERIMENT_ROOT / dataset / "selection_groups.jsonl"
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                extra = record.get("extra_info") or {}
                question_id = str(extra.get("id"))
                if question_id in ids:
                    key = (dataset, question_id)
                    if key in selected:
                        raise RuntimeError(f"Duplicate case key: {key}")
                    selected[key] = record

    expected = {(dataset, question_id) for dataset, question_id, _ in CASE_SPECS}
    if set(selected) != expected:
        raise RuntimeError(f"Missing records: {sorted(expected - set(selected))}")
    return selected


def validate_record(dataset: str, question_id: str, record: dict) -> float:
    counter = {str(key): int(value) for key, value in record["prediction_counter"].items()}
    if sum(counter.values()) != EXPECTED_N or len(record.get("samples") or []) != EXPECTED_N:
        raise RuntimeError(f"{dataset}/{question_id}: expected exactly 32 rollouts")

    existing_entropy = normalized_entropy(counter)
    if not existing_entropy > POOLED_Q67:
        raise RuntimeError(
            f"{dataset}/{question_id}: does not exceed the previous fixed high-entropy threshold"
        )
    if not bool(record.get("reward_selection_correct")):
        raise RuntimeError(f"{dataset}/{question_id}: ours is not correct")
    if bool(record.get("majority_correct")):
        raise RuntimeError(f"{dataset}/{question_id}: TTRV majority is not wrong")

    majority = str(record.get("majority_prediction"))
    ours = str(record.get("reward_selection_prediction"))
    invalid_labels = {"", "None", "invalid", "unknown"}
    if majority in invalid_labels or ours in invalid_labels:
        raise RuntimeError(f"{dataset}/{question_id}: invalid selected label")

    majority_count = int(record.get("majority_count", 0))
    other_peak = max((count for label, count in counter.items() if label != majority), default=-1)
    if majority_count <= other_peak:
        raise RuntimeError(f"{dataset}/{question_id}: TTRV selection is not a strict majority")
    return existing_entropy


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite existing directory: {OUTPUT_ROOT}")

    records = load_selected_records()
    OUTPUT_ROOT.mkdir(parents=True)

    for dataset, question_id, folder_name in CASE_SPECS:
        record = records[(dataset, question_id)]
        existing_entropy = validate_record(dataset, question_id, record)
        question, options = extract_question_and_options(record["prompt"])

        extra = record.get("extra_info") or {}
        source_image = Path(str(extra["image_path"]))
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        ground_truth = str(record["ground_truth"])
        ttrv_label = str(record["majority_prediction"])
        ours_label = str(record["reward_selection_prediction"])
        counter = {str(key): int(value) for key, value in record["prediction_counter"].items()}

        case_dir = OUTPUT_ROOT / folder_name
        case_dir.mkdir()
        image_name = "original_image" + source_image.suffix.lower()
        shutil.copy2(source_image, case_dir / image_name)

        metadata = {
            "case_id": folder_name,
            "dataset": dataset,
            "dataset_question_id": question_id,
            "experiment_index": str(record["index"]),
            "setting": {
                "name": "TTS Selection@32",
                "sample_count": EXPECTED_N,
                "training_update": False,
                "sampling_temperature": 1.0,
                "sampling_top_p": 0.95,
            },
            "original_image": image_name,
            "question": question,
            "options": options,
            "ground_truth": {
                "label": ground_truth,
                "text": answer_text(options, ground_truth),
            },
            "ttrv_selection": {
                "method": "frequency_majority@32",
                "label": ttrv_label,
                "text": answer_text(options, ttrv_label),
                "correct": False,
                "count": int(record["majority_count"]),
                "ratio": float(record["majority_ratio"]),
                "strict_majority": True,
            },
            "ours_selection": {
                "method": "PROVE_reward_selection@32",
                "label": ours_label,
                "text": answer_text(options, ours_label),
                "correct": True,
                "sample_index": int(record["reward_selection_sample_index"]),
                "score": float(record["reward_selection_score"]),
                "stored_sample_level_margin": float(record["reward_selection_margin"]),
                "top_score_tie_count": int(record["reward_selection_tie_count"]),
            },
            "entropy": {
                "value": existing_entropy,
                "bin": "high",
                "scope": "all_32_parser_outputs_including_unknown",
                "formula": "-sum_i(p_i * ln(p_i)) / ln(K_observed)",
                "source": "existing_TTS_Selection@32_prediction_counter; no rerun",
                "selection_basis": "previous_fixed_pooled_high_entropy_threshold",
                "fixed_previous_bin_thresholds": {
                    "low_max_q33": POOLED_Q33,
                    "mid_max_q67": POOLED_Q67,
                    "high_condition": "H > mid_max_q67",
                },
            },
            "rollout_distribution": counter,
            "invalid_or_unknown_count": int(record["invalid_count"]),
            "provenance": {
                "selection_groups_jsonl": str(EXPERIMENT_ROOT / dataset / "selection_groups.jsonl"),
                "source_image_path": str(source_image),
                "high_entropy_definition_source": str(
                    REPO_ROOT / "verl/scripts/analyze_obs2_all_output_majority.py"
                ),
                "previous_fixed_bin_map": str(FIXED_BIN_MAP),
            },
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"output_root": str(OUTPUT_ROOT), "cases": len(CASE_SPECS)}, indent=2))


if __name__ == "__main__":
    main()
