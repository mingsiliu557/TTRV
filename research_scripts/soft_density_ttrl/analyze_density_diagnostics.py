#!/usr/bin/env python3
"""Bucket-level summary for density-only prototype diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "verl"))

from verl.utils.reward_score.ttrl.ttt_metrics import _normalize_choice_answer, _parse_choice_labels  # noqa: E402


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def bucket_names(record: Dict[str, Any], style_payload: Dict[str, Any], choice_labels) -> List[str]:
    counter = {k: int(v) for k, v in style_payload.get("density_valid_counter", {}).items()}
    valid_total = sum(counter.values())
    buckets = ["all"]
    if len(counter) == 2:
        buckets.append("binary")
    if len(counter) >= 3:
        buckets.append("3plus_answers")
    if counter and max(counter.values()) == 1:
        buckets.append("all_unique")
    if float(style_payload.get("density_answer_entropy", 0.0)) >= 0.75:
        buckets.append("high_entropy")

    gt = _normalize_choice_answer(record.get("ground_truth"), choice_labels=choice_labels)
    majority = style_payload.get("original_majority", "unknown")
    majority_count = counter.get(majority, 0)
    if gt in counter and gt != majority:
        buckets.append("minority_correct")
    if valid_total > 0 and majority != gt and (majority_count / valid_total) >= 0.75:
        buckets.append("high_confidence_wrong")
    return buckets


def mean_bool(values: List[Any]) -> float:
    return float(np.mean([1.0 if value else 0.0 for value in values])) if values else 0.0


def summarize(records: List[Dict[str, Any]], choice_labels) -> Dict[str, Any]:
    by_style: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for style, payload in record.get("density_diagnostics", {}).items():
            for bucket in bucket_names(record, payload, choice_labels):
                by_style[style][bucket].append(payload)

    out: Dict[str, Any] = {}
    for style, buckets in by_style.items():
        out[style] = {}
        for bucket, payloads in buckets.items():
            out[style][bucket] = {
                "groups": len(payloads),
                "density_peak_label_acc": mean_bool([p.get("density_peak_label_correct") for p in payloads]),
                "majority_label_acc": mean_bool([p.get("original_majority_correct") for p in payloads]),
                "arithmetic_centroid_label_acc": mean_bool([p.get("arithmetic_centroid_label_correct") for p in payloads]),
                "density_vs_majority_agreement": float(np.mean([float(p.get("density_vs_majority_agreement", 0.0)) for p in payloads])) if payloads else 0.0,
                "density_vs_centroid_agreement": float(np.mean([float(p.get("density_vs_centroid_agreement", 0.0)) for p in payloads])) if payloads else 0.0,
                "peak_mass": float(np.mean([float(p.get("density_peak_mass", 0.0)) for p in payloads])) if payloads else 0.0,
                "temperature": float(np.mean([float(p.get("density_temperature", 0.0)) for p in payloads])) if payloads else 0.0,
                "corr_adv_density_freq": float(np.mean([float(p.get("corr_adv_density_freq", 0.0)) for p in payloads])) if payloads else 0.0,
                "mean_abs_diff_adv_density_freq": float(np.mean([float(p.get("mean_abs_diff_adv_density_freq", 0.0)) for p in payloads])) if payloads else 0.0,
            }
    return out


def to_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# Density Prototype Diagnostics", ""]
    for style, buckets in summary.items():
        lines.extend([f"## {style}", ""])
        lines.append("| bucket | groups | density acc | majority acc | centroid acc | peak mass | corr adv(freq) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for bucket, metrics in buckets.items():
            lines.append(
                "| {bucket} | {groups} | {density:.4f} | {majority:.4f} | {centroid:.4f} | {peak:.4f} | {corr:.4f} |".format(
                    bucket=bucket,
                    groups=metrics["groups"],
                    density=metrics["density_peak_label_acc"],
                    majority=metrics["majority_label_acc"],
                    centroid=metrics["arithmetic_centroid_label_acc"],
                    peak=metrics["peak_mass"],
                    corr=metrics["corr_adv_density_freq"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--choice-labels", default=None)
    args = parser.parse_args()

    choice_labels = _parse_choice_labels(args.choice_labels)
    records = list(read_jsonl(args.diagnostics_jsonl))
    summary = summarize(records, choice_labels)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
