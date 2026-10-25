#!/usr/bin/env python3
"""Replay the main density selection reward on fixed Stage-A N=32 outputs.

This command never generates responses. It reconstructs the evidence-query
embeddings for the already saved responses, applies the repository density
reward, and compares its selected candidate with all-output majority@32 under
the fixed step-0 pooled entropy bins.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VERL_ROOT = REPO_ROOT / "verl"
for value in (REPO_ROOT, VERL_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from attention_visualization.attention_viz import _prepare_inputs
from verl.trainer.ppo.ray_trainer import (
    _DENSITY_CANONICAL_EVIDENCE_DEFAULT_TEMPLATE,
    _DENSITY_EVIDENCE_DEFAULT_TEMPLATE,
    _canonicalize_density_evidence_text,
)
from verl.utils.reward_score.ttrl.ttt_metrics import (
    _density_peak_rewards,
    _normalize_choice_answer,
    _normalized_entropy,
    _parse_choice_labels,
)
from verl.utils.tokenizer import hf_processor


POOLED_Q33 = 0.4656663294644227
POOLED_Q67 = 0.7540931063268035
EXPECTED_BIN_COUNTS = {"low": 513, "mid": 487, "high": 500}
EXPECTED_QUESTIONS_PER_DATASET = 250
EXPECTED_SAMPLES = 32
BENCHMARK_ORDER = (
    "mme20",
    "ai2d20",
    "seed20",
    "mathvista20_hq",
    "mathverse20_fixed",
    "realworldqa20_hq",
)
DATASET_CONFIG = {
    "mme20": {
        "labels": "AB",
        "scope": "canonical_evidence_query_mean_pool",
        "temperature_mode": "answer_entropy",
        "t0": 0.20,
        "t_min": 0.10,
        "t_max": 0.30,
    },
    "ai2d20": {
        "labels": "A-D",
        "scope": "evidence_query_mean_pool",
        "temperature_mode": "answer_entropy",
        "t0": 0.20,
        "t_min": 0.10,
        "t_max": 0.30,
    },
    "seed20": {
        "labels": "A-D",
        "scope": "canonical_evidence_query_mean_pool",
        "temperature_mode": "answer_entropy",
        "t0": 0.20,
        "t_min": 0.10,
        "t_max": 0.30,
    },
    "mathvista20_hq": {
        "labels": "A-F",
        "scope": "canonical_evidence_query_mean_pool",
        "temperature_mode": "answer_entropy",
        "t0": 0.20,
        "t_min": 0.10,
        "t_max": 0.30,
    },
    "mathverse20_fixed": {
        "labels": "A-F",
        "scope": "evidence_query_mean_pool",
        "temperature_mode": "fixed",
        "t0": 0.20,
        "t_min": 0.20,
        "t_max": 0.20,
    },
    "realworldqa20_hq": {
        "labels": "A-D",
        "scope": "canonical_evidence_query_mean_pool",
        "temperature_mode": "fixed",
        "t0": 0.20,
        "t_min": 0.20,
        "t_max": 0.20,
    },
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fixed_bin(entropy: float) -> str:
    if entropy <= POOLED_Q33:
        return "low"
    if entropy <= POOLED_Q67:
        return "mid"
    return "high"


def load_bin_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (str(raw["benchmark"]), str(raw["question_id"]))
            if key in result:
                raise ValueError(f"Duplicate fixed-bin key: {key}")
            entropy = float(raw["step0_all_output_entropy"])
            label = str(raw["h_bin"])
            if label != fixed_bin(entropy):
                raise ValueError(f"Fixed-bin mismatch for {key}: {label} vs {fixed_bin(entropy)}")
            result[key] = {"entropy": entropy, "h_bin": label}
    counts = Counter(value["h_bin"] for value in result.values())
    if len(result) != 1500 or dict(counts) != EXPECTED_BIN_COUNTS:
        raise ValueError(f"Invalid fixed-bin map: n={len(result)} counts={dict(counts)}")
    return result


def load_manifest(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            benchmark = str(row.get("benchmark") or row.get("key") or "")
            parquet = str(row.get("real_parquet") or "")
            if benchmark in DATASET_CONFIG and parquet:
                result[benchmark] = Path(parquet)
    missing = set(BENCHMARK_ORDER) - set(result)
    if missing:
        raise ValueError(f"Manifest is missing real parquet paths for {sorted(missing)}")
    return result


def load_parquet_rows(path: Path) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    result: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        extra = row.get("extra_info")
        if not isinstance(extra, dict):
            continue
        qid = str(extra.get("diagnostic_question_id") or "")
        if not qid:
            continue
        if qid in result:
            raise ValueError(f"Duplicate diagnostic_question_id={qid} in {path}")
        result[qid] = row
    return result


def load_dump_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(path):
        grouped[str(record["question_id"])].append(record)
    for qid, records in grouped.items():
        records.sort(key=lambda item: int(item["sample_idx"]))
        if [int(item["sample_idx"]) for item in records] != list(range(EXPECTED_SAMPLES)):
            raise ValueError(f"{path}/{qid}: expected sample_idx 0..31")
    if len(grouped) != EXPECTED_QUESTIONS_PER_DATASET:
        raise ValueError(f"{path}: expected 250 groups, got {len(grouped)}")
    return grouped


def load_model(model_path: Path, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    processor = hf_processor(str(model_path), trust_remote_code=True)
    if processor is None:
        raise RuntimeError(f"No multimodal processor for {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.eval().to(device)
    return model, tokenizer, processor


def normalized_response_text(tokenizer: Any, raw_text: str) -> str:
    token_ids = tokenizer.encode(str(raw_text), add_special_tokens=False)[:1024]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return text if text else "unknown"


def evidence_query(scope: str, prompt_text: str, response_text: str) -> str:
    if scope == "canonical_evidence_query_mean_pool":
        return _canonicalize_density_evidence_text(
            prompt_text,
            response_text,
            _DENSITY_CANONICAL_EVIDENCE_DEFAULT_TEMPLATE,
        )
    if scope == "evidence_query_mean_pool":
        return _DENSITY_EVIDENCE_DEFAULT_TEMPLATE.format(
            response=response_text,
            answer=response_text,
            candidate=response_text,
            option_label="unknown",
            option_text="",
        )
    raise ValueError(f"Unsupported embedding scope: {scope}")


def embed_queries(
    model: Any,
    tokenizer: Any,
    prepared: dict[str, Any],
    queries: list[str],
    device: str,
    micro_batch_size: int,
) -> dict[str, np.ndarray]:
    import torch

    if not queries:
        return {}
    input_ids = prepared["input_ids"].to(device)
    attention_mask = prepared.get("attention_mask", torch.ones_like(input_ids)).to(device)
    pixel_values = prepared["pixel_values"].to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        visual = model.extract_feature(pixel_values)
    visual_flat = visual.reshape(-1, visual.shape[-1])
    expected_visual = int(input_ids.eq(model.img_context_token_id).sum().item())
    if expected_visual != int(visual_flat.shape[0]):
        raise ValueError(
            f"Visual token mismatch: prompt={expected_visual} encoder={visual_flat.shape[0]}"
        )

    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(queries), micro_batch_size):
        chunk = queries[start : start + micro_batch_size]
        encoded = tokenizer(
            chunk,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        query_ids = encoded["input_ids"].to(device)
        query_mask = encoded["attention_mask"].to(device)
        batch_size = query_ids.shape[0]
        prompt_ids = input_ids.expand(batch_size, -1)
        prompt_mask = attention_mask.expand(batch_size, -1)
        full_ids = torch.cat([prompt_ids, query_ids], dim=1)
        full_mask = torch.cat([prompt_mask, query_mask], dim=1)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inputs_embeds = model.language_model.get_input_embeddings()(full_ids).clone()
            selected = full_ids.eq(model.img_context_token_id)
            tiled_visual = visual_flat.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, visual_flat.shape[-1])
            if int(selected.sum().item()) != int(tiled_visual.shape[0]):
                raise ValueError("Repeated visual-token count mismatch")
            inputs_embeds[selected] = tiled_visual.to(inputs_embeds.dtype)
            output = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = output.hidden_states[-1][:, -query_ids.shape[1] :, :].float()
            mask = query_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        for query, vector in zip(chunk, pooled.detach().cpu().numpy()):
            embeddings[query] = vector.astype(np.float32)
    return embeddings


def process_question(
    benchmark: str,
    qid: str,
    records: list[dict[str, Any]],
    row: dict[str, Any],
    fixed: dict[str, Any],
    model: Any,
    tokenizer: Any,
    processor: Any,
    device: str,
    micro_batch_size: int,
) -> dict[str, Any]:
    config = DATASET_CONFIG[benchmark]
    labels = _parse_choice_labels(config["labels"])
    parsed = [str(record["parsed_answer"]) for record in records]
    counter = Counter(parsed)
    entropy = _normalized_entropy(counter, len(parsed))
    if not math.isclose(entropy, float(fixed["entropy"]), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{benchmark}/{qid}: entropy drift {entropy} vs {fixed['entropy']}")

    prepared, raw_prompt = _prepare_inputs(processor, row)
    response_texts = [normalized_response_text(tokenizer, record["raw_text"]) for record in records]
    queries = [evidence_query(config["scope"], raw_prompt, response) for response in response_texts]
    canonical_answers = [
        _normalize_choice_answer(answer, choice_labels=config["labels"])
        for answer in parsed
    ]
    valid_queries = sorted(
        {
            query
            for query, answer in zip(queries, canonical_answers)
            if answer in labels
        }
    )
    query_embeddings = embed_queries(
        model,
        tokenizer,
        prepared,
        valid_queries,
        device,
        micro_batch_size,
    )
    hidden_size = int(model.language_model.config.hidden_size)
    response_embeddings = [
        query_embeddings.get(query, np.zeros((hidden_size,), dtype=np.float32))
        for query in queries
    ]
    rewards, details = _density_peak_rewards(
        "gpqa",
        parsed,
        response_embeddings,
        records[0]["ground_truth"],
        choice_labels=config["labels"],
        reward_mode="cluster_soft",
        temperature_mode=config["temperature_mode"],
        t0=config["t0"],
        t_min=config["t_min"],
        t_max=config["t_max"],
        embedding_scope=config["scope"],
    )

    majority_answer, majority_count = counter.most_common(1)[0]
    majority_index = parsed.index(majority_answer)
    reward_index = int(np.argmax(np.asarray(rewards, dtype=np.float32)))
    max_reward = float(rewards[reward_index])
    reward_ties = sum(abs(float(value) - max_reward) <= 1e-8 for value in rewards)
    return {
        "benchmark": benchmark,
        "question_id": qid,
        "step0_all_output_entropy": entropy,
        "h_bin": fixed["h_bin"],
        "pooled_q33": POOLED_Q33,
        "pooled_q67": POOLED_Q67,
        "ground_truth": records[0]["ground_truth"],
        "majority_answer": majority_answer,
        "majority_count": int(majority_count),
        "majority_sample_index": int(majority_index),
        "majority_correct": bool(records[majority_index]["is_correct"]),
        "majority_is_invalid": not bool(records[majority_index]["is_valid"]),
        "ours_answer": parsed[reward_index],
        "ours_sample_index": reward_index,
        "ours_correct": bool(records[reward_index]["is_correct"]),
        "ours_is_invalid": not bool(records[reward_index]["is_valid"]),
        "ours_selection_score": max_reward,
        "ours_selection_tie_count": int(reward_ties),
        "selection_answer_differs": parsed[reward_index] != majority_answer,
        "reward_variant": "density_plus_repository_temperature_no_counterfactual",
        "counterfactual_available": False,
        "counterfactual_omission_reason": "Stage-A dump has no original/counterfactual token log-prob",
        "density_embedding_scope": config["scope"],
        "density_temperature_mode": config["temperature_mode"],
        "density_temperature": float(details["density_temperature"]),
        "density_answer_entropy_valid_only": float(details["density_answer_entropy"]),
        "density_valid_ratio": float(details["density_valid_ratio"]),
        "density_peak_label": details["density_peak_label"],
        "density_cluster_masses": details["density_cluster_masses"],
        "parsed_counter_all_outputs": dict(counter),
        "candidate_rewards": [float(value) for value in rewards],
    }


def summarize(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scopes = [(benchmark, [row for row in records if row["benchmark"] == benchmark]) for benchmark in BENCHMARK_ORDER]
    scopes.append(("MCQ_MICRO", records))
    rows: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    for scope, selected in scopes:
        for label in ("low", "mid", "high"):
            bucket = [row for row in selected if row["h_bin"] == label]
            majority_correct = sum(bool(row["majority_correct"]) for row in bucket)
            ours_correct = sum(bool(row["ours_correct"]) for row in bucket)
            rows.append(
                {
                    "benchmark": scope,
                    "h_bin": label,
                    "n": len(bucket),
                    "majority_correct": majority_correct,
                    "majority_at_32": majority_correct / len(bucket) if bucket else None,
                    "ours_correct": ours_correct,
                    "ours_selection_at_32": ours_correct / len(bucket) if bucket else None,
                    "delta_ours_minus_majority": (ours_correct - majority_correct) / len(bucket) if bucket else None,
                    "selection_answer_differs_count": sum(bool(row["selection_answer_differs"]) for row in bucket),
                    "majority_invalid_count": sum(bool(row["majority_is_invalid"]) for row in bucket),
                    "ours_invalid_count": sum(bool(row["ours_is_invalid"]) for row in bucket),
                }
            )
        majority_correct = sum(bool(row["majority_correct"]) for row in selected)
        ours_correct = sum(bool(row["ours_correct"]) for row in selected)
        totals.append(
            {
                "benchmark": scope,
                "n": len(selected),
                "majority_correct": majority_correct,
                "majority_at_32": majority_correct / len(selected),
                "ours_correct": ours_correct,
                "ours_selection_at_32": ours_correct / len(selected),
                "delta_ours_minus_majority": (ours_correct - majority_correct) / len(selected),
                "selection_answer_differs_count": sum(bool(row["selection_answer_differs"]) for row in selected),
                "majority_invalid_count": sum(bool(row["majority_is_invalid"]) for row in selected),
                "ours_invalid_count": sum(bool(row["ours_is_invalid"]) for row in selected),
            }
        )
    return rows, totals


def pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def write_report(path: Path, bins: list[dict[str, Any]], totals: list[dict[str, Any]]) -> None:
    by_key = {(row["benchmark"], row["h_bin"]): row for row in bins}
    lines = [
        "# Fixed Step-0 Entropy Bins: Majority@32 vs Ours-Selection",
        "",
        "- Bins are fixed by step-0 all-output entropy: Low <= 0.4656663295,",
        "  Mid <= 0.7540931063, High above it.",
        "- Majority counts every parser output; an invalid majority is wrong.",
        "- Ours uses the repository semantic-density and temperature implementation.",
        "- Counterfactual log-prob was not stored in Stage A, so this is density +",
        "  temperature only. No responses were regenerated.",
        "- The repository temperature is inside `exp(sim_ij / T)` before density",
        "  aggregation, so it can change the selected candidate.",
        "",
        "## Pooled Main Table",
        "",
        "| H bin | n | majority@32 | Ours-Selection | Delta | Different answers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("low", "mid", "high"):
        row = by_key[("MCQ_MICRO", label)]
        lines.append(
            f"| {label.title()} | {row['n']} | {row['majority_correct']}/{row['n']} "
            f"({pct(row['majority_at_32'])}) | {row['ours_correct']}/{row['n']} "
            f"({pct(row['ours_selection_at_32'])}) | {pct(row['delta_ours_minus_majority'])} "
            f"| {row['selection_answer_differs_count']} |"
        )
    total = next(row for row in totals if row["benchmark"] == "MCQ_MICRO")
    lines.append(
        f"| **Overall** | **{total['n']}** | **{total['majority_correct']}/{total['n']} "
        f"({pct(total['majority_at_32'])})** | **{total['ours_correct']}/{total['n']} "
        f"({pct(total['ours_selection_at_32'])})** | **{pct(total['delta_ours_minus_majority'])}** "
        f"| **{total['selection_answer_differs_count']}** |"
    )
    lines.extend(["", "## Per-dataset Appendix", ""])
    for benchmark in BENCHMARK_ORDER:
        lines.extend(
            [
                f"### {benchmark}",
                "",
                "| H bin | n | majority@32 | Ours-Selection | Delta | Different answers |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label in ("low", "mid", "high"):
            row = by_key[(benchmark, label)]
            lines.append(
                f"| {label.title()} | {row['n']} | {row['majority_correct']}/{row['n']} "
                f"({pct(row['majority_at_32'])}) | {row['ours_correct']}/{row['n']} "
                f"({pct(row['ours_selection_at_32'])}) | {pct(row['delta_ours_minus_majority'])} "
                f"| {row['selection_answer_differs_count']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bin-map", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--benchmarks", nargs="*", choices=BENCHMARK_ORDER)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    args = parser.parse_args()

    selected_benchmarks = tuple(args.benchmarks or BENCHMARK_ORDER)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_map = load_bin_map(args.bin_map)
    parquet_paths = load_manifest(args.manifest)
    result_path = args.output_dir / "question_selection.jsonl"
    completed = {
        (str(row["benchmark"]), str(row["question_id"]))
        for row in read_jsonl(result_path)
    } if result_path.exists() else set()

    model, tokenizer, processor = load_model(args.base_model, args.device)
    for benchmark in selected_benchmarks:
        groups = load_dump_groups(args.dump_root / benchmark / "real.jsonl")
        parquet_rows = load_parquet_rows(parquet_paths[benchmark])
        qids = sorted(groups, key=lambda value: int(groups[value][0]["subset_position"]))
        if args.limit_per_dataset > 0:
            qids = qids[: args.limit_per_dataset]
        for position, qid in enumerate(qids, start=1):
            key = (benchmark, qid)
            if key in completed:
                continue
            if key not in fixed_map:
                raise KeyError(f"Missing fixed-bin mapping for {key}")
            if qid not in parquet_rows:
                raise KeyError(f"Missing replay parquet row for {key}")
            result = process_question(
                benchmark,
                qid,
                groups[qid],
                parquet_rows[qid],
                fixed_map[key],
                model,
                tokenizer,
                processor,
                args.device,
                args.micro_batch_size,
            )
            append_jsonl(result_path, result)
            completed.add(key)
            if position % 10 == 0 or position == len(qids):
                print(f"[replay] {benchmark} {position}/{len(qids)}", flush=True)

    results = list(read_jsonl(result_path))
    expected = len(selected_benchmarks) * (
        args.limit_per_dataset if args.limit_per_dataset > 0 else EXPECTED_QUESTIONS_PER_DATASET
    )
    selected_results = [row for row in results if row["benchmark"] in selected_benchmarks]
    if len(selected_results) != expected:
        raise ValueError(f"Expected {expected} completed questions, got {len(selected_results)}")

    if set(selected_benchmarks) == set(BENCHMARK_ORDER) and args.limit_per_dataset == 0:
        bins, totals = summarize(selected_results)
        atomic_csv(args.output_dir / "fixed_bin_selection.csv", bins)
        atomic_csv(args.output_dir / "fixed_bin_selection_totals.csv", totals)
        write_report(args.output_dir / "fixed_bin_selection_report.md", bins, totals)
        atomic_json(
            args.output_dir / "run_summary.json",
            {
                "status": "complete",
                "questions": len(selected_results),
                "fixed_bin_counts": dict(Counter(row["h_bin"] for row in selected_results)),
                "counterfactual_available": False,
                "reward_variant": "density_plus_repository_temperature_no_counterfactual",
                "report": str(args.output_dir / "fixed_bin_selection_report.md"),
            },
        )
    else:
        atomic_json(
            args.output_dir / "smoke_summary.json",
            {
                "status": "partial_smoke",
                "benchmarks": selected_benchmarks,
                "questions": len(selected_results),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
