#!/usr/bin/env python3
"""Build a RealWorldQA train split whose base majority vote is correct.

The script has two stages:
1. make-pool: use step0 flat validation outputs to collect likely-correct
   candidates from the existing test split.
2. make-final: use a base majority-scan rollout JSONL to keep only examples
   whose majority prediction matches the ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from datasets import Dataset


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_dataset(path: Path, rows) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "train.json", rows["train"])
    write_json(path / "test.json", rows["test"])
    Dataset.from_list(rows["train"]).to_parquet(str(path / "train.parquet"))
    Dataset.from_list(rows["test"]).to_parquet(str(path / "test.parquet"))


def normalize_answer(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def retag_sample(sample, dataset_name: str, split: str, new_index: int, source_index: str):
    item = deepcopy(sample)
    extra = dict(item.get("extra_info") or {})
    extra["source_dataset"] = extra.get("dataset")
    extra["source_index"] = source_index
    extra["split"] = split
    extra["index"] = f"{dataset_name}-{split}-{new_index}"
    extra["dataset"] = dataset_name
    item["extra_info"] = extra
    return item


def make_pool(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    source_test = load_json(source_dir / "test.json")
    source_train = load_json(source_dir / "train.json")
    by_index = {
        (row.get("extra_info") or {}).get("index"): row
        for row in source_test
    }

    records = list(iter_jsonl(Path(args.validation_flat)))
    if args.validation_pass == "first":
        records = records[: len(source_test)]
    elif args.validation_pass == "last":
        records = records[-len(source_test):]

    selected = []
    seen = set()
    for record in records:
        if args.eval_dump_call is not None and int(record.get("eval_dump_call", 0)) != args.eval_dump_call:
            continue
        if not bool(record.get("correct")):
            continue
        index = record.get("index") or (record.get("extra_info") or {}).get("index")
        if not index or index in seen or index not in by_index:
            continue
        seen.add(index)
        selected.append(index)
        if len(selected) >= args.pool_size:
            break

    if len(selected) < args.pool_size:
        raise RuntimeError(
            f"Only found {len(selected)} correct step0 candidates, need {args.pool_size}."
        )

    dataset_name = output_dir.name
    train_rows = [
        retag_sample(by_index[index], dataset_name, "train", i, index)
        for i, index in enumerate(selected)
    ]
    test_rows = [
        retag_sample(row, dataset_name, "test", i, (row.get("extra_info") or {}).get("index", str(i)))
        for i, row in enumerate(source_test)
    ]
    write_dataset(output_dir, {"train": train_rows, "test": test_rows})
    summary = {
        "source_dir": str(source_dir),
        "validation_flat": str(args.validation_flat),
        "validation_pass": args.validation_pass,
        "eval_dump_call": args.eval_dump_call,
        "selection_rule": "flat validation correct == true, unique test examples",
        "pool_size": len(train_rows),
        "original_train_size": len(source_train),
        "test_size": len(test_rows),
        "selected_source_indices": selected,
    }
    write_json(output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_final(args: argparse.Namespace) -> None:
    pool_dir = Path(args.pool_dir)
    output_dir = Path(args.output_dir)
    pool_train = load_json(pool_dir / "train.json")
    source_test = load_json(Path(args.test_source_dir) / "test.json")
    by_index = {
        (row.get("extra_info") or {}).get("index"): row
        for row in pool_train
    }

    selected_records = []
    selected_source_indices = []
    seen = set()
    for record in iter_jsonl(Path(args.scan_rollouts)):
        index = record.get("index")
        if not index or index in seen or index not in by_index:
            continue
        majority = normalize_answer(record.get("majority_prediction"))
        truth = normalize_answer(record.get("ground_truth"))
        if majority != truth:
            continue
        seen.add(index)
        extra = by_index[index].get("extra_info") or {}
        selected_records.append(
            {
                "pool_index": index,
                "source_index": extra.get("source_index"),
                "ground_truth": truth,
                "majority_prediction": majority,
                "majority_count": int(record.get("majority_count", 0)),
                "majority_ratio": float(record.get("majority_ratio", 0.0)),
                "prediction_counter": record.get("prediction_counter", {}),
                "valid_counter": record.get("valid_counter", {}),
            }
        )
        selected_source_indices.append(extra.get("source_index"))
        if len(selected_records) >= args.final_size:
            break

    if len(selected_records) < args.final_size:
        raise RuntimeError(
            f"Only found {len(selected_records)} majority-correct candidates, need {args.final_size}."
        )

    dataset_name = output_dir.name
    train_rows = [
        retag_sample(by_index[item["pool_index"]], dataset_name, "train", i, item["source_index"])
        for i, item in enumerate(selected_records)
    ]
    test_rows = [
        retag_sample(row, dataset_name, "test", i, (row.get("extra_info") or {}).get("index", str(i)))
        for i, row in enumerate(source_test)
    ]
    write_dataset(output_dir, {"train": train_rows, "test": test_rows})
    for name in ("test_alignment_manifest.json", "train_alignment_manifest.json"):
        src = Path(args.test_source_dir) / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    summary = {
        "pool_dir": str(pool_dir),
        "scan_rollouts": str(args.scan_rollouts),
        "test_source_dir": str(args.test_source_dir),
        "selection_rule": "base 32-sample train rollout majority_prediction == ground_truth",
        "final_size": len(train_rows),
        "test_size": len(test_rows),
        "selected": selected_records,
        "selected_source_indices": selected_source_indices,
    }
    write_json(output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pool = sub.add_parser("make-pool")
    p_pool.add_argument("--source-dir", required=True)
    p_pool.add_argument("--validation-flat", required=True)
    p_pool.add_argument("--output-dir", required=True)
    p_pool.add_argument("--pool-size", type=int, default=80)
    p_pool.add_argument("--validation-pass", choices=("first", "last", "all"), default="first")
    p_pool.add_argument("--eval-dump-call", type=int, default=None)
    p_pool.set_defaults(func=make_pool)

    p_final = sub.add_parser("make-final")
    p_final.add_argument("--pool-dir", required=True)
    p_final.add_argument("--scan-rollouts", required=True)
    p_final.add_argument("--test-source-dir", required=True)
    p_final.add_argument("--output-dir", required=True)
    p_final.add_argument("--final-size", type=int, default=20)
    p_final.set_defaults(func=make_final)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
