#!/usr/bin/env python
import argparse
import copy
import json
import re
from pathlib import Path

import datasets


VARIANTS = ("kv_short",)
ANSWER_ONLY_PATTERNS = (
    r"\s*Please answer directly with only the letter of the correct option and nothing else\.?\s*$",
    r"\s*Please respond with only the corresponding option letter\s*\([^)]*\)\.?\s*$",
    r"\s*Do not include any explanation or extra text\.?\s*$",
)


def _load_rows(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _clean_realworldqa_prompt(content: str) -> str:
    text = str(content or "").strip()
    for pattern in ANSWER_ONLY_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _prompt_for_variant(original_content: str, variant: str) -> str:
    if variant != "kv_short":
        raise ValueError(f"Unsupported RealWorldQA variant: {variant}")
    base = _clean_realworldqa_prompt(original_content)
    instruction = (
        "Answer using exactly two lines.\n"
        "Line 1 gives one short visual clue.\n"
        "Line 2 gives one option letter only.\n\n"
        "Format:\n"
        "Evidence: short visual clue\n"
        "Answer: A\n\n"
        "Do not write anything before or after these two lines."
    )
    return f"{base}\n\n{instruction}"


def _convert_rows(rows: list[dict], variant: str) -> list[dict]:
    converted = []
    for row in rows:
        item = copy.deepcopy(row)
        original = row["prompt"][0]["content"]
        item["prompt"][0]["content"] = _prompt_for_variant(original, variant)
        extra = item.setdefault("extra_info", {})
        extra["reasoning_pilot_variant"] = variant
        extra["reasoning_pilot_adapter"] = "realworldqa"
        extra["reasoning_pilot_original_prompt"] = original
        converted.append(item)
    return converted


def _write_split(rows: list[dict], output_dir: Path, split: str) -> None:
    json_path = output_dir / f"{split}.json"
    parquet_path = output_dir / f"{split}.parquet"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(rows).to_parquet(str(parquet_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RealWorldQA-specific kv_short prompt variant.")
    parser.add_argument("--source-dir", default="/jiigan-hp/ttrv-datasets/verl_data/realworldqa_20")
    parser.add_argument("--output-root", default="/jiigan-hp/ttrv-datasets/verl_data/realworldqa_20_reasoning_kv_short")
    parser.add_argument("--test-limit", type=int, default=-1)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--variants", nargs="+", default=["kv_short"], choices=VARIANTS)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = _load_rows(source_dir / "train.json", args.train_limit)
    test_rows = _load_rows(source_dir / "test.json", args.test_limit)

    summary = {
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "adapter": "realworldqa",
        "variants": {},
        "test_limit": args.test_limit,
        "train_limit": args.train_limit,
    }
    for variant in args.variants:
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        converted_train = _convert_rows(train_rows, variant)
        converted_test = _convert_rows(test_rows, variant)
        _write_split(converted_train, variant_dir, "train")
        _write_split(converted_test, variant_dir, "test")
        summary["variants"][variant] = {
            "dir": str(variant_dir),
            "train_examples": len(converted_train),
            "test_examples": len(converted_test),
            "train_parquet": str(variant_dir / "train.parquet"),
            "test_parquet": str(variant_dir / "test.parquet"),
        }

    summary_path = output_root / "prepare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
