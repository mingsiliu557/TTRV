#!/usr/bin/env python
import argparse
import copy
import json
import re
from pathlib import Path

import datasets


VARIANTS = ("p0", "p1", "p2", "p3", "kv_short", "ans_short", "json_short")


def _split_prompt(content: str) -> tuple[str, str]:
    match = re.search(r"\n\s*Options\s*:\s*", content, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Prompt does not contain an Options block: {content[:200]}")
    question = content[: match.start()]
    options = content[match.start() :].strip()
    return question, options


def _clean_question(question: str) -> str:
    question = re.sub(r"^\s*<image>\s*", "", question, flags=re.IGNORECASE).strip()
    question = re.sub(
        r"\s*Please respond with only the corresponding option letter\s*\([^)]*\)\.?",
        "",
        question,
        flags=re.IGNORECASE,
    )
    question = re.sub(r"\s*Please answer with yes or no\.?", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s*Do not include any explanation or extra text\.?", "", question, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", question).strip()


def _prompt_for_variant(original_content: str, variant: str) -> str:
    if variant == "p0":
        return original_content

    question_block, options = _split_prompt(original_content)
    question = _clean_question(question_block)
    if variant == "p1":
        instruction = (
            "Look at the image and answer the question.\n"
            "First write one short visual evidence sentence.\n"
            "Then write the final option letter.\n\n"
            "Use exactly this format:\n"
            "<evidence>...</evidence>\n"
            "<answer>A</answer>\n\n"
            "The answer must be one option letter only."
        )
    elif variant == "p2":
        instruction = (
            "Write only one short visual evidence sentence and one final option letter.\n"
            "Do not include any other text.\n\n"
            "Format:\n"
            "<evidence>...</evidence>\n"
            "<answer>A</answer>\n\n"
            "Rules:\n"
            "- Evidence must describe visible information in the image.\n"
            "- Answer must be one option letter only.\n"
            "- If uncertain, still choose the most likely option."
        )
    elif variant == "p3":
        instruction = (
            "Answer with this exact format:\n"
            "<evidence>short visual clue</evidence>\n"
            "<answer>option letter</answer>"
        )
    elif variant == "kv_short":
        instruction = (
            "Answer using exactly two lines.\n"
            "Line 1 gives one short visual clue.\n"
            "Line 2 gives one option letter only.\n\n"
            "Format:\n"
            "Evidence: short visual clue\n"
            "Answer: A\n\n"
            "Do not write anything before or after these two lines."
        )
    elif variant == "ans_short":
        instruction = (
            "Answer using exactly two short fields.\n"
            "Use one visible clue and one option letter only.\n\n"
            "Format:\n"
            "OBS=short visual clue\n"
            "ANS=A\n\n"
            "Do not write anything before or after these two fields."
        )
    elif variant == "json_short":
        instruction = (
            "Return one compact JSON object only.\n"
            "The evidence must be one short visual clue.\n"
            "The answer must be one option letter only.\n\n"
            'Format: {"evidence":"short visual clue","answer":"A"}'
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return f"<image>\n{question}\n\n{instruction}\n\n{options}"


def _load_rows(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _convert_rows(rows: list[dict], variant: str) -> list[dict]:
    converted = []
    for row in rows:
        item = copy.deepcopy(row)
        item["prompt"][0]["content"] = _prompt_for_variant(row["prompt"][0]["content"], variant)
        extra = item.setdefault("extra_info", {})
        extra["reasoning_pilot_variant"] = variant
        extra["reasoning_pilot_original_prompt"] = row["prompt"][0]["content"]
        converted.append(item)
    return converted


def _write_split(rows: list[dict], output_dir: Path, split: str) -> None:
    json_path = output_dir / f"{split}.json"
    parquet_path = output_dir / f"{split}.parquet"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(rows).to_parquet(str(parquet_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporary MME20 prompt variants for evidence-reasoning pilot.")
    parser.add_argument("--source-dir", default="/jiigan-hp/ttrv-datasets/verl_data/mme_20")
    parser.add_argument("--output-root", default="/jiigan-hp/ttrv-datasets/verl_data/mme_20_reasoning_pilot")
    parser.add_argument("--test-limit", type=int, default=-1)
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = _load_rows(source_dir / "train.json", args.train_limit)
    test_rows = _load_rows(source_dir / "test.json", args.test_limit)

    summary = {
        "source_dir": str(source_dir),
        "output_root": str(output_root),
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
