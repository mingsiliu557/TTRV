#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any

import datasets


DATASETS = ("ai2d", "mathverse", "mathvista", "mme", "realworldqa", "seed")
OLD_PREFIX = "/home/anirban/kanksha1/"


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _relocate_image_path(image_path: str, data_root: Path) -> str:
    if image_path.startswith(OLD_PREFIX):
        return str(data_root / image_path[len(OLD_PREFIX) :])
    return image_path


def _convert_split(dataset: str, split: str, examples: list[dict[str, Any]], data_root: Path, allow_missing_images: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    for idx, example in enumerate(examples):
        original_image_path = example.get("image_path", "")
        image_path = _relocate_image_path(original_image_path, data_root)
        exists = bool(image_path) and Path(image_path).exists()
        if not exists:
            missing.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "row": idx,
                    "id": example.get("id"),
                    "original_image_path": original_image_path,
                    "resolved_image_path": image_path,
                }
            )
        rows.append(
            {
                "data_source": "GPQA-TTT",
                "prompt": [{"role": "user", "content": example["prompt"]}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": {
                    "split": split,
                    "index": f"{dataset}_20-{split}-{idx}",
                    "dataset": f"{dataset}_20",
                    "id": example.get("id"),
                    "source": example.get("source"),
                    "image_path": image_path,
                    "original_image_path": original_image_path,
                    "image_exists": exists,
                },
                "images": [{"image": image_path}] if exists or allow_missing_images else None,
            }
        )
    return rows, missing


def _prepare_one(dataset: str, template_root: Path, output_root: Path, data_root: Path, allow_missing_images: bool) -> dict[str, Any]:
    source_dir = template_root / f"{dataset}_20"
    output_dir = output_root / f"{dataset}_20"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }
    all_missing = []
    for split in ("train", "test"):
        raw_examples = _load_json(source_dir / f"{split}.json")
        rows, missing = _convert_split(dataset, split, raw_examples, data_root, allow_missing_images)
        all_missing.extend(missing)
        json_path = output_dir / f"{split}.json"
        parquet_path = output_dir / f"{split}.parquet"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        datasets.Dataset.from_list(rows).to_parquet(str(parquet_path))
        summary["splits"][split] = {
            "num_examples": len(rows),
            "missing_images": len(missing),
            "json": str(json_path),
            "parquet": str(parquet_path),
        }
    if all_missing:
        missing_path = output_dir / "missing_images.json"
        with missing_path.open("w", encoding="utf-8") as f:
            json.dump(all_missing, f, ensure_ascii=False, indent=2)
        summary["missing_images_file"] = str(missing_path)
    summary["missing_images_total"] = len(all_missing)
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if all_missing and not allow_missing_images:
        preview = "\n".join(f"- {m['resolved_image_path']}" for m in all_missing[:20])
        raise FileNotFoundError(f"{dataset}: {len(all_missing)} image paths are missing.\n{preview}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare relocated Table 2 VQA 20-shot JSON/parquet files for TTRV/verl.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--template-root", default="verl/data")
    parser.add_argument("--output-root", default="/jiigan-hp/ttrv-datasets/verl_data")
    parser.add_argument("--data-root", default="/jiigan-hp/ttrv-datasets")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--summary-path", default="/jiigan-hp/ttrv-datasets/verl_data/table2_vqa20_prepare_summary.json")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "template_root": args.template_root,
        "output_root": str(output_root),
        "data_root": args.data_root,
        "datasets": {},
        "skipped": {"capture": "no local TTRV template", "crpe": "no local TTRV template"},
    }
    for dataset in args.datasets:
        result = _prepare_one(
            dataset,
            Path(args.template_root),
            output_root,
            Path(args.data_root),
            args.allow_missing_images,
        )
        summary["datasets"][dataset] = result
        print(json.dumps(result, ensure_ascii=False, indent=2))
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[prepare_table2_vqa20_data] summary={summary_path}")


if __name__ == "__main__":
    main()
