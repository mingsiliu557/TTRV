#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/export MME images from Hugging Face into category subfolders.")
    parser.add_argument("--dataset-id", default="MM-Hallu/MME")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-parquet-dir", default=None)
    parser.add_argument("--output-root", default="/jiigan-hp/ttrv-datasets/MME/images")
    parser.add_argument("--cache-dir", default="/jiigan-hp/ttrv-datasets/hf_home/datasets")
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.local_parquet_dir:
        parquet_files = sorted(str(path) for path in Path(args.local_parquet_dir).glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {args.local_parquet_dir}")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train", cache_dir=str(cache_dir))
    else:
        parquet_files = None
        dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=str(cache_dir))
    written = 0
    skipped = 0
    category_counts = {}
    missing_question_id = []

    for row_idx, row in enumerate(dataset):
        rel_path = row.get("question_id")
        if not rel_path:
            missing_question_id.append(row_idx)
            continue
        rel_path = str(rel_path).lstrip("/")
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        category = str(row.get("category") or Path(rel_path).parts[0])
        category_counts[category] = category_counts.get(category, 0) + 1

        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(out_path)
        written += 1

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "local_parquet_dir": args.local_parquet_dir,
        "parquet_files": parquet_files,
        "rows": len(dataset),
        "output_root": str(output_root),
        "cache_dir": str(cache_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_question_id": missing_question_id,
        "category_counts": dict(sorted(category_counts.items())),
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_root.parent / "download_mme_images_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
