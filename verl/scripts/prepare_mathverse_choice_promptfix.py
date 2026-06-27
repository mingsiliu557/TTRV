#!/usr/bin/env python
import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import datasets


DEFAULT_LABELS = ("A", "B", "C", "D", "E", "F")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _labels_text(labels: tuple[str, ...]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _option_block(content: str) -> str:
    matches = list(re.finditer(r"\bChoices?\s*:", content, flags=re.IGNORECASE))
    if not matches:
        return ""
    return content[matches[-1].start() :]


def _infer_labels(content: str) -> tuple[str, ...]:
    block = _option_block(content)
    if not block:
        return DEFAULT_LABELS
    labels = []
    for match in re.finditer(r"(?<![A-Z0-9])([A-F])\s*[:.)]", block.upper()):
        label = match.group(1)
        if label not in labels:
            labels.append(label)
    return tuple(labels) if labels else DEFAULT_LABELS


def _replace_instruction(content: str, labels: tuple[str, ...]) -> tuple[str, bool]:
    label_text = _labels_text(labels)
    replacement = (
        f"Please directly answer the question and provide only one option letter "
        f"({label_text}).\nDo not include any explanation or extra text."
    )
    patterns = [
        (
            r"Please directly answer the question and provide the correct option letter,\s*"
            r"e\.g\.,\s*A,\s*B,\s*C,\s*D\.\s*"
            r"Do not include any explanation or extra text\.?",
            replacement,
        ),
        (
            r"Please directly answer the question and provide the correct option letter.*?"
            r"Do not include any explanation or extra text\.?",
            replacement,
        ),
    ]
    for pattern, repl in patterns:
        new_content, n = re.subn(pattern, repl, content, count=1, flags=re.IGNORECASE | re.DOTALL)
        if n:
            return new_content, True
    return content, False


def _convert_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted = []
    label_counts: dict[str, int] = {}
    replaced = 0
    no_option_block = 0
    for row in rows:
        item = copy.deepcopy(row)
        content = item["prompt"][0]["content"]
        labels = _infer_labels(content)
        if not _option_block(content):
            no_option_block += 1
        new_content, did_replace = _replace_instruction(content, labels)
        replaced += int(did_replace)
        item["prompt"][0]["content"] = new_content
        extra = item.setdefault("extra_info", {})
        extra["mathverse_promptfix"] = True
        extra["mathverse_promptfix_choice_labels"] = "".join(labels)
        extra["mathverse_promptfix_replaced_instruction"] = did_replace
        extra["mathverse_promptfix_has_option_block"] = bool(_option_block(content))
        key = "".join(labels)
        label_counts[key] = label_counts.get(key, 0) + 1
        converted.append(item)
    summary = {
        "num_examples": len(rows),
        "instruction_replaced": replaced,
        "no_option_block": no_option_block,
        "choice_label_counts": label_counts,
    }
    return converted, summary


def _write_split(rows: list[dict[str, Any]], output_dir: Path, split: str) -> None:
    with (output_dir / f"{split}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(rows).to_parquet(str(output_dir / f"{split}.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MathVerse prompt copy with dynamic option-letter instruction.")
    parser.add_argument("--source-dir", default="/jiigan-hp/ttrv-datasets/verl_data/mathverse_20_choice_norm")
    parser.add_argument("--output-dir", default="/jiigan-hp/ttrv-datasets/verl_data/mathverse_20_choice_norm_promptfix")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }
    for split in ("train", "test"):
        rows, split_summary = _convert_rows(_load_rows(source_dir / f"{split}.json"))
        _write_split(rows, output_dir, split)
        summary["splits"][split] = {
            **split_summary,
            "json": str(output_dir / f"{split}.json"),
            "parquet": str(output_dir / f"{split}.parquet"),
        }

    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
