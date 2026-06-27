#!/usr/bin/env python
import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


DATASETS = ("ai2d", "mathverse", "mathvista", "mme", "realworldqa", "seed")

OLD_PREFIX = "/home/anirban/kanksha1/"

HF_DATASETS = {
    "ai2d": ("lmms-lab/ai2d", None, None),
    "mathverse": ("AI4Math/MathVerse", "testmini", None),
    "mathvista": ("AI4Math/MathVista", None, None),
    "realworldqa": ("xai-org/RealworldQA", None, "test"),
}


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _target_path(image_path: str, data_root: Path) -> Path:
    if image_path.startswith(OLD_PREFIX):
        return data_root / image_path[len(OLD_PREFIX) :]
    return Path(image_path)


def _safe_stem(path: str) -> str:
    return Path(path).stem


def _normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<image>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"please respond.*?(?:extra text\.|nothing else\.)", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"do not include.*?(?:extra text\.|nothing else\.)", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"choose the correct answer.*?option letter.*?\.", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"hint:\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"options?\s*(are)?\s*:\s*.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"choices?\s*:\s*.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bquestion\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _template_rows(template_root: Path, dataset: str) -> list[dict[str, Any]]:
    task = f"{dataset}_20"
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for row in _load_json(template_root / task / f"{split}.json"):
            item = dict(row)
            item["_split"] = split
            rows.append(item)
    return rows


def _needed_paths(template_root: Path, data_root: Path, dataset: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in _template_rows(template_root, dataset):
        src = row.get("image_path")
        if not src:
            continue
        paths[src] = _target_path(src, data_root)
    return paths


def _image_from_value(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        path = value.get("path")
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
    path = Path(str(value))
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def _row_image(row: dict[str, Any]) -> Image.Image | None:
    for key in ("image", "decoded_image", "img", "image_file"):
        image = _image_from_value(row.get(key))
        if image is not None:
            return image
    return None


def _image_relative_path(row: dict[str, Any]) -> str | None:
    for key in ("image", "decoded_image", "img", "image_file"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("path"):
            return str(value["path"])
        if isinstance(value, str):
            return value
    return None


def _candidate_targets_from_image_path(dataset: str, row: dict[str, Any], data_root: Path) -> list[Path]:
    rel = _image_relative_path(row)
    if not rel:
        return []
    rel_path = Path(rel)
    if dataset == "mathverse":
        return [data_root / "mathverse" / "images" / rel_path]
    if dataset == "mathvista":
        return [data_root / "mathvista" / "data" / "images" / rel_path.name]
    return []


def _row_text_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("question", "query", "problem", "prompt"):
        if key in row:
            normalized = _normalize_text(row[key])
            if normalized:
                keys.append(normalized)
    return keys


def _disable_image_decode(ds: Any) -> Any:
    from datasets import Image

    def cast_one(split_ds: Any) -> Any:
        for column, feature in getattr(split_ds, "features", {}).items():
            if feature.__class__.__name__ == "Image":
                split_ds = split_ds.cast_column(column, Image(decode=False))
        return split_ds

    if hasattr(ds, "keys") and not isinstance(ds, list):
        for split in list(ds.keys()):
            ds[split] = cast_one(ds[split])
        return ds
    return cast_one(ds)


def _iter_dataset(ds: Any):
    if hasattr(ds, "keys") and not isinstance(ds, list):
        for split in ds.keys():
            yield from ds[split]
    else:
        yield from ds


def _load_hf_dataset(dataset: str, cache_dir: Path) -> Any:
    from datasets import load_dataset

    dataset_id, config, split = HF_DATASETS[dataset]
    kwargs = {"cache_dir": str(cache_dir)}
    if split:
        kwargs["split"] = split
    if config:
        loaded = load_dataset(dataset_id, config, **kwargs)
    else:
        loaded = load_dataset(dataset_id, **kwargs)
    return _disable_image_decode(loaded)


def _save_image(image: Image.Image, target: Path, overwrite: bool) -> bool:
    if target.exists() and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    fmt = "PNG" if suffix == ".png" else "JPEG"
    image.save(target, format=fmt)
    return True


def _download_by_prompt_match(
    dataset: str,
    template_root: Path,
    data_root: Path,
    cache_dir: Path,
    overwrite: bool,
    limit: int,
) -> dict[str, Any]:
    rows = _template_rows(template_root, dataset)
    rows_by_question: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _normalize_text(row.get("prompt", ""))
        if key:
            rows_by_question.setdefault(key, []).append(row)

    needed_targets = {str(path): path for path in _needed_paths(template_root, data_root, dataset).values()}
    hf_rows = _iter_dataset(_load_hf_dataset(dataset, cache_dir))
    saved = 0
    matched = 0
    ambiguous = 0
    inspected = 0
    for hf_row in hf_rows:
        if limit > 0 and inspected >= limit:
            break
        inspected += 1
        unique_targets: dict[str, Path] = {}

        for target in _candidate_targets_from_image_path(dataset, hf_row, data_root):
            if str(target) in needed_targets:
                unique_targets[str(target)] = target

        if not unique_targets:
            candidate_rows: list[dict[str, Any]] = []
            for key in _row_text_keys(hf_row):
                candidate_rows.extend(rows_by_question.get(key, []))
            for row in candidate_rows:
                target = _target_path(row["image_path"], data_root)
                unique_targets[str(target)] = target

        if not unique_targets:
            continue
        if len(unique_targets) > 1:
            ambiguous += 1
        image = _row_image(hf_row)
        if image is None:
            continue
        for target in unique_targets.values():
            matched += 1
            if _save_image(image, target, overwrite):
                saved += 1
    return {
        "dataset": dataset,
        "method": "hf_prompt_match",
        "hf_rows_inspected": inspected,
        "template_rows": len(rows),
        "matched_targets": matched,
        "saved_images": saved,
        "ambiguous_hf_rows": ambiguous,
    }


def _download_realworldqa(template_root: Path, data_root: Path, cache_dir: Path, overwrite: bool) -> dict[str, Any]:
    hf_dataset = _load_hf_dataset("realworldqa", cache_dir)
    rows = _iter_dataset(hf_dataset)
    saved = 0
    hf_rows_count = 0
    for idx, row in enumerate(rows):
        hf_rows_count += 1
        image = _row_image(row)
        if image is None:
            continue
        target = data_root / "realworldqa" / "images" / f"{idx}.png"
        if _save_image(image, target, overwrite):
            saved += 1
    needed = _needed_paths(template_root, data_root, "realworldqa")
    present = sum(1 for path in needed.values() if path.exists())
    return {
        "dataset": "realworldqa",
        "method": "hf_row_index",
        "hf_rows": hf_rows_count,
        "saved_images": saved,
        "needed_unique_images": len(needed),
        "needed_present": present,
    }


def _download_seed(template_root: Path, data_root: Path, cache_dir: Path, overwrite: bool) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    needed = _needed_paths(template_root, data_root, "seed")
    needed_names = {path.name for path in needed.values()}
    zip_path = Path(
        hf_hub_download(
            repo_id="AILab-CVC/SEED-Bench",
            repo_type="dataset",
            filename="SEED-Bench-image.zip",
            cache_dir=str(cache_dir),
        )
    )
    saved = 0
    scanned = 0
    with zipfile.ZipFile(zip_path) as zf:
        by_name = {Path(info.filename).name: info for info in zf.infolist() if not info.is_dir()}
        for original, target in needed.items():
            if target.exists() and not overwrite:
                continue
            info = by_name.get(target.name)
            if info is None:
                continue
            scanned += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            saved += 1
    present = sum(1 for path in needed.values() if path.exists())
    return {
        "dataset": "seed",
        "method": "hf_zip_selective_extract",
        "archive": str(zip_path),
        "needed_unique_images": len(needed_names),
        "archive_matches": scanned,
        "saved_images": saved,
        "needed_present": present,
    }


def _verify(dataset: str, template_root: Path, data_root: Path) -> dict[str, Any]:
    needed = _needed_paths(template_root, data_root, dataset)
    missing = [str(path) for path in needed.values() if not path.exists()]
    return {
        "dataset": dataset,
        "needed_unique_images": len(needed),
        "present": len(needed) - len(missing),
        "missing": len(missing),
        "missing_preview": missing[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Table 2 VQA images needed by local TTRV *_20 templates.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--template-root", default="verl/data")
    parser.add_argument("--data-root", default="/jiigan-hp/ttrv-datasets")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/jiigan-hp/ttrv-datasets/hf_home"))
    parser.add_argument("--summary-path", default="/jiigan-hp/ttrv-datasets/verl_data/table2_vqa20_download_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-hf-rows", type=int, default=-1, help="Debug limit for prompt-matched HF datasets.")
    args = parser.parse_args()

    template_root = Path(args.template_root)
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    summary_path = Path(args.summary_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "template_root": str(template_root),
        "cache_dir": str(cache_dir),
        "datasets": {},
        "skipped": {"capture": "no local TTRV template", "crpe": "no local TTRV template"},
    }
    for dataset in args.datasets:
        if dataset == "mme":
            result = {"dataset": "mme", "method": "existing_local_or_manual", **_verify(dataset, template_root, data_root)}
        elif dataset == "realworldqa":
            result = _download_realworldqa(template_root, data_root, cache_dir, args.overwrite)
        elif dataset == "seed":
            result = _download_seed(template_root, data_root, cache_dir, args.overwrite)
        else:
            result = _download_by_prompt_match(
                dataset,
                template_root,
                data_root,
                cache_dir,
                args.overwrite,
                args.limit_hf_rows,
            )
        result["verify_after"] = _verify(dataset, template_root, data_root)
        summary["datasets"][dataset] = result
        print(json.dumps(result, ensure_ascii=False, indent=2))

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[download_table2_vqa_images] summary={summary_path}")


if __name__ == "__main__":
    main()
