#!/usr/bin/env python3
"""Create white-image copies of existing verl json/parquet datasets.

The prompt, answer, and metadata are preserved. Image references are redirected
to generated all-white RGB images with the same size as the original image when
the source file is available.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _dump_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _row_image_path(row: dict[str, Any]) -> str | None:
    images = row.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            value = first.get("image")
            if value:
                return str(value)
        elif first:
            return str(first)
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("image_path"):
        return str(extra["image_path"])
    return None


def _blank_path(image_root: Path, task: str, source: str) -> Path:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:24]
    suffix = Path(source).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = ".png"
    return image_root / task / f"{digest}.png"


def _make_white_image(source: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(source)
    missing = not src_path.exists()
    size = (224, 224)
    if not missing:
        try:
            with Image.open(src_path) as img:
                size = tuple(img.size)
        except Exception:
            missing = True
    if not destination.exists():
        Image.new("RGB", size, (255, 255, 255)).save(destination)
    return {"path": str(destination), "size": list(size), "source_missing": bool(missing)}


def _rewrite_row(row: dict[str, Any], blank_image: str, source_image: str | None) -> dict[str, Any]:
    new_row = copy.deepcopy(row)
    extra = new_row.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}
        new_row["extra_info"] = extra
    extra["blank_image_replacement"] = True
    if source_image is not None:
        extra["original_image_path_before_blank"] = source_image
    extra["image_path"] = blank_image

    images = new_row.get("images")
    if isinstance(images, list) and images:
        if isinstance(images[0], dict):
            images[0]["image"] = blank_image
        else:
            images[0] = blank_image
    else:
        new_row["images"] = [{"image": blank_image}]
    return new_row


def process_task(data_root: Path, image_root: Path, task: str, suffix: str) -> dict[str, Any]:
    src_dir = data_root / task
    dst_task = f"{task}{suffix}"
    dst_dir = data_root / dst_task
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    summary: dict[str, Any] = {
        "source_task": task,
        "blank_task": dst_task,
        "source_dir": str(src_dir),
        "output_dir": str(dst_dir),
        "splits": {},
        "images": {},
    }
    image_cache: dict[str, dict[str, Any]] = {}

    for split in ("train", "test"):
        src_json = src_dir / f"{split}.json"
        if not src_json.exists():
            continue
        rows = _load_json(src_json)
        new_rows: list[dict[str, Any]] = []
        missing_image_refs = 0
        for row in rows:
            source = _row_image_path(row)
            if source is None:
                missing_image_refs += 1
                source = f"missing://{task}/{split}/{len(new_rows)}"
            if source not in image_cache:
                dst_image = _blank_path(image_root, task, source)
                image_cache[source] = _make_white_image(source, dst_image)
            new_rows.append(_rewrite_row(row, image_cache[source]["path"], None if source.startswith("missing://") else source))

        _dump_json(dst_dir / f"{split}.json", new_rows)
        pd.DataFrame(new_rows).to_parquet(dst_dir / f"{split}.parquet", index=False)
        summary["splits"][split] = {
            "rows": len(rows),
            "missing_image_refs": missing_image_refs,
        }

    summary["images"] = {
        "unique_sources": len(image_cache),
        "generated_root": str(image_root / task),
        "source_missing": sum(1 for item in image_cache.values() if item["source_missing"]),
    }
    _dump_json(dst_dir / "blank_white_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/ttrv/verl_data"))
    parser.add_argument("--image-root", type=Path, default=Path("/data/ttrv/blank_white_images"))
    parser.add_argument("--suffix", default="_white_blank")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    summaries = [process_task(args.data_root, args.image_root, task, args.suffix) for task in args.tasks]
    if args.summary_json:
        _dump_json(args.summary_json, summaries)
    for item in summaries:
        print(
            f"{item['source_task']} -> {item['blank_task']} "
            f"rows={sum(split['rows'] for split in item['splits'].values())} "
            f"unique_images={item['images']['unique_sources']} "
            f"source_missing={item['images']['source_missing']}"
        )


if __name__ == "__main__":
    main()
