#!/usr/bin/env python
import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import datasets


def _split_question_options(content: str) -> tuple[str, str]:
    patterns = [
        r"\n\s*Options\s*:\s*",
        r"\n\s*Options\s+are\s*:\s*",
        r"\n\s*Choices?\s*:\s*",
        r"(?=\n\s*A[\.\):]\s+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            return content[: match.start()].strip(), content[match.start() :].strip()
    if re.search(r"\bQuestion\s*:\s*", content, flags=re.IGNORECASE):
        return content.strip(), ""
    raise ValueError(f"Prompt does not contain recognizable options: {content[:200]}")


def _clean_question(question: str) -> str:
    question = re.sub(r"^\s*<image>\s*", "", question, flags=re.IGNORECASE).strip()
    question = re.sub(
        r"\s*Please respond with only the corresponding option letter\s*\([^)]*\)\.?",
        "",
        question,
        flags=re.IGNORECASE,
    )
    question = re.sub(
        r"\s*Choose the correct answer from the options below and respond with only the corresponding option letter\s*\([^)]*\)\.?",
        "",
        question,
        flags=re.IGNORECASE,
    )
    question = re.sub(r"\s*Please answer with yes or no\.?", "", question, flags=re.IGNORECASE)
    question = re.sub(
        r"\s*Please directly answer the question and provide the correct option letter.*?extra text\.?",
        "",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    question = re.sub(r"\s*Hint:\s*", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s*Do not include any explanation or extra text\.?", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s*Please answer directly with only the letter.*?nothing else\.?", "", question, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", question).strip()


def _kv_short_prompt(content: str) -> str:
    question_block, options = _split_question_options(content)
    question = _clean_question(question_block)
    instruction = (
        "Answer using exactly two lines.\n"
        "Line 1 gives one short visual clue.\n"
        "Line 2 gives one option letter only.\n\n"
        "Format:\n"
        "Evidence: short visual clue\n"
        "Answer: A\n\n"
        "Do not write anything before or after these two lines."
    )
    if options:
        return f"<image>\n{question}\n\n{instruction}\n\n{options}"
    return f"<image>\n{question}\n\n{instruction}"


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows[:limit] if limit and limit > 0 else rows


def _convert_rows(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item = copy.deepcopy(row)
        original = row["prompt"][0]["content"]
        if variant != "kv_short":
            raise ValueError(f"Unsupported variant: {variant}")
        item["prompt"][0]["content"] = _kv_short_prompt(original)
        extra = item.setdefault("extra_info", {})
        extra["reasoning_pilot_variant"] = variant
        extra["reasoning_pilot_original_prompt"] = original
        converted.append(item)
    return converted


def _write_split(rows: list[dict[str, Any]], output_dir: Path, split: str) -> None:
    with (output_dir / f"{split}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(rows).to_parquet(str(output_dir / f"{split}.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generic Table 2 VQA reasoning prompt variants.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--variant", default="kv_short", choices=["kv_short"])
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    task_name = args.task_name or source_dir.name
    output_dir = Path(args.output_root) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _convert_rows(_load_rows(source_dir / "train.json", args.train_limit), args.variant)
    test_rows = _convert_rows(_load_rows(source_dir / "test.json", args.test_limit), args.variant)
    _write_split(train_rows, output_dir, "train")
    _write_split(test_rows, output_dir, "test")
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "variant": args.variant,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "train_parquet": str(output_dir / "train.parquet"),
        "test_parquet": str(output_dir / "test.parquet"),
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
