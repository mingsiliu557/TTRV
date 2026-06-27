#!/usr/bin/env python3
"""Recompute density-only prototype rewards from rollout JSONL with response embeddings.

This diagnostic intentionally reuses the current feature-center representation:
`samples[*].response_embedding`, optionally written by TTRLRewardManager when
`TTRL_LOG_RESPONSE_EMBEDDINGS=1` is set. It does not implement a separate VLM
hidden-state extractor.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "verl"))

from verl.utils.reward_score.ttrl.ttt_metrics import (  # noqa: E402
    _density_peak_rewards,
    _normalize_choice_answer,
    _parse_choice_labels,
)

STYLE_CONFIG = {
    "density_peak_hard": ("hard", "fixed"),
    "density_peak_soft": ("soft", "fixed"),
    "density_peak_answer_entropy": ("hard", "answer_entropy"),
    "density_peak_density_entropy": ("hard", "density_entropy"),
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def data_source_to_task(data_source: Optional[str]) -> str:
    if data_source in {"MATH-TTT", "AIME-TTT", "AMC-TTT", "data/AIME-TTT"}:
        return "math"
    if data_source in {"GPQA-TTT", None, ""}:
        return "gpqa"
    if data_source == "bbox":
        return "bbox"
    return "gpqa"


def sample_embedding(sample: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("response_embedding", "embedding", "response_embeddings"):
        value = sample.get(key)
        if value is not None:
            return value
    return None


def extract_embeddings(record: Dict[str, Any]) -> Optional[List[List[float]]]:
    if "response_embeddings" in record:
        return record["response_embeddings"]
    embeddings = []
    for sample in record.get("samples", []):
        value = sample_embedding(sample)
        if value is None:
            return None
        embeddings.append(value)
    return embeddings


def summarize_style(details_list: List[Dict[str, Any]]) -> Dict[str, float]:
    if not details_list:
        return {"groups": 0}
    keys = [
        "density_peak_label_accuracy",
        "original_majority_accuracy",
        "arithmetic_centroid_label_accuracy",
        "density_valid_ratio",
        "density_peak_mass",
        "density_answer_entropy",
        "density_density_entropy",
        "density_temperature",
        "density_sim_mean",
        "density_sim_std",
        "density_vs_majority_agreement",
        "density_vs_centroid_agreement",
        "corr_adv_density_freq",
        "mean_abs_diff_adv_density_freq",
    ]
    out: Dict[str, float] = {"groups": float(len(details_list))}
    for key in keys:
        values = [float(d.get(key, 0.0)) for d in details_list]
        out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--styles", nargs="+", default=list(STYLE_CONFIG))
    parser.add_argument("--choice-labels", default=None)
    parser.add_argument("--t0", type=float, default=0.2)
    parser.add_argument("--t-min", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=0.8)
    args = parser.parse_args()

    choice_labels = _parse_choice_labels(args.choice_labels)
    unknown_styles = [style for style in args.styles if style not in STYLE_CONFIG]
    if unknown_styles:
        raise ValueError(f"Unknown density styles: {unknown_styles}")

    output_records = []
    style_details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped_missing_embeddings = 0
    total_groups = 0

    for record in read_jsonl(args.input_jsonl):
        total_groups += 1
        samples = record.get("samples", [])
        embeddings = extract_embeddings(record)
        if embeddings is None:
            skipped_missing_embeddings += 1
            continue
        answers = [sample.get("prediction", "unknown") for sample in samples]
        solutions = [sample.get("response_raw", "") for sample in samples]
        ground_truth = record.get("ground_truth")
        task = data_source_to_task(record.get("data_source"))
        extra_info = record.get("extra_info")

        diagnostics: Dict[str, Any] = {}
        for style in args.styles:
            reward_mode, temperature_mode = STYLE_CONFIG[style]
            rewards, details = _density_peak_rewards(
                task=task,
                model_answers=answers,
                response_embeddings=embeddings,
                ground_truth=ground_truth,
                extra_info=extra_info,
                choice_labels=choice_labels,
                reward_mode=reward_mode,
                temperature_mode=temperature_mode,
                t0=args.t0,
                t_min=args.t_min,
                t_max=args.t_max,
            )
            style_details[style].append(details)
            diagnostics[style] = {
                "rewards": [float(x) for x in rewards],
                "density_peak_label": details["density_peak_label"],
                "density_peak_label_correct": bool(details["density_peak_label_accuracy"]),
                "density_peak_sample_index": details["density_peak_sample_index"],
                "density_peak_mass": details["density_peak_mass"],
                "density_valid_counter": details["density_valid_counter"],
                "density_valid_ratio": details["density_valid_ratio"],
                "density_temperature": details["density_temperature"],
                "density_temperature_mode": details["density_temperature_mode"],
                "density_reward_mode": details["density_reward_mode"],
                "density_answer_entropy": details["density_answer_entropy"],
                "density_density_entropy": details["density_density_entropy"],
                "density_sim_mean": details["density_sim_mean"],
                "density_sim_std": details["density_sim_std"],
                "original_majority": details["original_majority"],
                "original_majority_correct": bool(details["original_majority_accuracy"]),
                "arithmetic_centroid_label": details["arithmetic_centroid_label"],
                "arithmetic_centroid_label_correct": bool(details["arithmetic_centroid_label_accuracy"]),
                "density_vs_majority_agreement": details["density_vs_majority_agreement"],
                "density_vs_centroid_agreement": details["density_vs_centroid_agreement"],
                "corr_adv_density_freq": details["corr_adv_density_freq"],
                "mean_abs_diff_adv_density_freq": details["mean_abs_diff_adv_density_freq"],
            }

        updated = dict(record)
        updated["density_diagnostics"] = diagnostics
        updated["choice_labels"] = list(choice_labels)
        updated["ground_truth_normalized"] = _normalize_choice_answer(ground_truth, choice_labels=choice_labels)
        output_records.append(updated)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "total_groups": total_groups,
        "processed_groups": len(output_records),
        "skipped_missing_embeddings": skipped_missing_embeddings,
        "styles": {style: summarize_style(details) for style, details in style_details.items()},
    }

    write_jsonl(args.output_jsonl, output_records)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
