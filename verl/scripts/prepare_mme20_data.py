#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import datasets


def _candidate_image_roots(data_root: Path, explicit_root: str | None) -> list[Path]:
    roots = []
    if explicit_root:
        roots.append(Path(explicit_root).expanduser())
    roots.extend(
        [
            data_root / "MME" / "images",
            data_root / "mme" / "MME" / "images",
            data_root / "MME_Benchmark_release_version" / "images",
            data_root / "images",
        ]
    )
    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _mme_relative_path(image_path: str) -> Path:
    marker = "/MME/images/"
    if marker in image_path:
        return Path(image_path.split(marker, 1)[1])
    marker = "MME/images/"
    if marker in image_path:
        return Path(image_path.split(marker, 1)[1])
    return Path(image_path).name


def _resolve_image_path(original_path: str, image_roots: list[Path]) -> tuple[str, bool, str]:
    original = Path(original_path)
    if original.exists():
        rel = _mme_relative_path(original_path)
        return str(original), True, rel.parts[0] if len(rel.parts) > 1 else "unknown"

    rel = _mme_relative_path(original_path)
    for root in image_roots:
        candidate = root / rel
        if candidate.exists():
            return str(candidate), True, rel.parts[0] if len(rel.parts) > 1 else "unknown"

    fallback = image_roots[0] / rel
    return str(fallback), False, rel.parts[0] if len(rel.parts) > 1 else "unknown"


def _convert_split(split: str, examples: list[dict], image_roots: list[Path], allow_missing_images: bool) -> tuple[list[dict], list[dict]]:
    rows = []
    missing = []
    for idx, example in enumerate(examples):
        original_image_path = example.get("image_path", "")
        image_path, image_exists, mme_category = _resolve_image_path(original_image_path, image_roots)
        if not image_exists:
            missing.append(
                {
                    "split": split,
                    "row": idx,
                    "id": example.get("id"),
                    "original_image_path": original_image_path,
                    "resolved_image_path": image_path,
                }
            )

        images = [{"image": image_path}] if image_exists or allow_missing_images else None
        rows.append(
            {
                "data_source": "GPQA-TTT",
                "prompt": [
                    {
                        "role": "user",
                        "content": example["prompt"],
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": {
                    "split": split,
                    "index": f"mme_20-{split}-{idx}",
                    "dataset": "mme_20",
                    "id": example.get("id"),
                    "source": example.get("source"),
                    "image_path": image_path,
                    "original_image_path": original_image_path,
                    "mme_category": mme_category,
                    "image_exists": image_exists,
                },
                "images": images,
            }
        )
    if missing and not allow_missing_images:
        preview = "\n".join(
            f"- {m['split']}[{m['row']}]: {m['resolved_image_path']} <- {m['original_image_path']}"
            for m in missing[:20]
        )
        raise FileNotFoundError(
            "MME images are missing. Set MME_IMAGE_ROOT to the directory that contains "
            "category subfolders such as code_reasoning/, scene/, posters/.\n"
            f"Missing preview:\n{preview}"
        )
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare relocated MME20 parquet files for TTRV/verl.")
    parser.add_argument("--source-dir", default="verl/data/mme_20", help="Directory containing train.json and test.json.")
    parser.add_argument("--output-dir", default="/jiigan-hp/ttrv-datasets/verl_data/mme_20")
    parser.add_argument("--data-root", default="/jiigan-hp/ttrv-datasets")
    parser.add_argument("--mme-image-root", default=os.environ.get("MME_IMAGE_ROOT"))
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    image_roots = _candidate_image_roots(data_root, args.mme_image_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "image_roots_checked": [str(root) for root in image_roots],
        "splits": {},
    }

    all_missing = []
    for split in ("train", "test"):
        with (source_dir / f"{split}.json").open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)
        limit = args.train_limit if split == "train" else args.test_limit
        if limit is not None and limit > 0:
            raw_examples = raw_examples[:limit]
        rows, missing = _convert_split(split, raw_examples, image_roots, args.allow_missing_images)
        all_missing.extend(missing)

        json_path = output_dir / f"{split}.json"
        parquet_path = output_dir / f"{split}.parquet"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        datasets.Dataset.from_list(rows).to_parquet(str(parquet_path))

        category_counts = {}
        for row in rows:
            category = row["extra_info"]["mme_category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        summary["splits"][split] = {
            "num_examples": len(rows),
            "missing_images": len(missing),
            "category_counts": category_counts,
            "json": str(json_path),
            "parquet": str(parquet_path),
        }

    summary["missing_images_total"] = len(all_missing)
    summary_path = output_dir / "prepare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if all_missing:
        print(f"[prepare_mme20_data] warning: {len(all_missing)} missing images; allow_missing_images={args.allow_missing_images}")


if __name__ == "__main__":
    main()
