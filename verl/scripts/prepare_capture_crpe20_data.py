#!/usr/bin/env python
import argparse
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import datasets
from PIL import Image


DATA_SOURCE = "GPQA-TTT"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_snapshot(hf_home: Path, repo_slug: str) -> Path:
    root = hf_home / repo_slug / "snapshots"
    snapshots = [p for p in root.iterdir() if p.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under {root}")
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (output_dir / info.filename).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe zip member path: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _write_template(task_dir: Path, split: str, examples: list[dict[str, Any]]) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{split}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    return path


def _to_verl_rows(task: str, split: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, example in enumerate(examples):
        image_path = example["image_path"]
        exists = bool(image_path) and Path(image_path).exists()
        extra_info = {
            "split": split,
            "index": f"{task}-{split}-{idx}",
            "dataset": task,
            "id": example.get("id"),
            "source": example.get("source"),
            "image_path": image_path,
            "original_image_path": example.get("original_image_path", image_path),
            "image_exists": exists,
        }
        for key in (
            "category",
            "object",
            "metric",
            "question_id",
            "raw_image",
            "answer_choice_labels",
            "source_row_index",
        ):
            if key in example:
                extra_info[key] = example[key]
        rows.append(
            {
                "data_source": example.get("data_source", DATA_SOURCE),
                "prompt": [{"role": "user", "content": example["prompt"]}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": extra_info,
                "images": [{"image": image_path}] if exists else None,
            }
        )
    return rows


def _write_verl_split(output_dir: Path, task: str, split: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _to_verl_rows(task, split, examples)
    json_path = output_dir / f"{split}.json"
    parquet_path = output_dir / f"{split}.parquet"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(rows).to_parquet(str(parquet_path))
    missing = sum(1 for row in rows if not row["extra_info"]["image_exists"])
    return {
        "num_examples": len(rows),
        "missing_images": missing,
        "json": str(json_path),
        "parquet": str(parquet_path),
    }


def _sample_train(examples: list[dict[str, Any]], train_size: int, seed: int) -> list[dict[str, Any]]:
    if train_size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(examples)), train_size))
    return [examples[i] for i in selected]


def _capture_prompt(object_name: str) -> str:
    return (
        f"<image>\n"
        f"Count the exact number of {object_name} in the image. "
        f"Assume the pattern of {object_name} continues behind any black box. "
        f"Provide the total number of {object_name} as if the black box were not there.\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}"
    )


def _prepare_capture(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = Path(args.capture_snapshot) if args.capture_snapshot else _latest_snapshot(
        Path(args.hf_home), "datasets--atinp--CAPTURe"
    )
    capture_root = Path(args.data_root) / "CAPTURe"
    _safe_extract_zip(snapshot / "real_dataset.zip", capture_root)
    shutil.copy2(snapshot / "real_metadata.json", capture_root / "real_metadata.json")

    metadata = _load_json(snapshot / "real_metadata.json")
    examples = []
    filtered_large = []
    missing = []
    for source_idx, row in enumerate(metadata):
        image_path = capture_root / "dataset" / row["image_file"]
        if not image_path.exists():
            missing.append(row["image_file"])
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        if max(width, height) > args.max_image_size:
            filtered_large.append({"image_file": row["image_file"], "width": width, "height": height})
            continue
        object_name = str(row["object"]).strip()
        examples.append(
            {
                "id": Path(row["image_file"]).stem,
                "source": "capture_real",
                "data_source": "MATH-TTT",
                "source_row_index": source_idx,
                "prompt": _capture_prompt(object_name),
                "answer": str(row["ground_truth"]),
                "image_path": str(image_path),
                "original_image_path": f"real_dataset.zip:dataset/{row['image_file']}",
                "object": object_name,
                "metric": "smape",
            }
        )

    task = "capture_20"
    train_examples = _sample_train(examples, args.train_size, args.seed)
    task_template_dir = Path(args.template_root) / task
    output_dir = Path(args.output_root) / task
    template_summary = {
        "train": str(_write_template(task_template_dir, "train", train_examples)),
        "test": str(_write_template(task_template_dir, "test", examples)),
    }
    split_summary = {
        "train": _write_verl_split(output_dir, task, "train", train_examples),
        "test": _write_verl_split(output_dir, task, "test", examples),
    }
    summary = {
        "dataset": "capture",
        "task": task,
        "source": "atinp/CAPTURe real split",
        "snapshot": str(snapshot),
        "metadata_rows": len(metadata),
        "prepared_examples": len(examples),
        "filtered_large_images": len(filtered_large),
        "missing_images": len(missing),
        "train_size": len(train_examples),
        "train_seed": args.seed,
        "template": template_summary,
        "splits": split_summary,
        "notes": [
            "TTRV reports CAPTURe used test size 817 after filtering images larger than 1000x1000.",
            "Ground truth is numeric; final TTRV reproduction needs a numeric/boxed parser and sMAPE metric.",
        ],
    }
    if filtered_large:
        summary["filtered_large_preview"] = filtered_large[:20]
    if missing:
        summary["missing_preview"] = missing[:20]
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _normalize_crpe_prompt(text: str) -> str:
    text = text.strip()
    full_instruction = (
        "Answer with the option's letter from the given choices directly "
        "and do not include any explanation or extra text."
    )
    text = re.sub(
        r"Answer with the option's letter from the given choices directly\.?",
        full_instruction,
        text,
        flags=re.IGNORECASE,
    )
    if "do not include any explanation" not in text.lower():
        text = f"{text.rstrip()}\n{full_instruction}"
    return "<image> \n" + text


def _resolve_crpe_image(crpe_root: Path, image_ref: str) -> Path:
    if image_ref.startswith("coco/val2017/"):
        return crpe_root / image_ref
    if image_ref.startswith("abnormal_images/"):
        return crpe_root / image_ref
    return crpe_root / image_ref


def _prepare_crpe(args: argparse.Namespace) -> dict[str, Any]:
    crpe_root = Path(args.data_root) / "CRPE"
    source_jsonl = crpe_root / args.crpe_jsonl
    rows = _load_jsonl(source_jsonl)
    examples = []
    missing = []
    for source_idx, row in enumerate(rows):
        image_path = _resolve_crpe_image(crpe_root, row["image"])
        if not image_path.exists():
            missing.append({"row": source_idx, "image": row["image"], "resolved": str(image_path)})
        examples.append(
            {
                "id": str(row["question_id"]),
                "question_id": row["question_id"],
                "source": Path(args.crpe_jsonl).stem,
                "source_row_index": source_idx,
                "category": row.get("category"),
                "prompt": _normalize_crpe_prompt(row["text"]),
                "answer": str(row["correct_option"]).strip().upper(),
                "image_path": str(image_path),
                "original_image_path": row["image"],
                "raw_image": row["image"],
                "answer_choice_labels": "A-D",
                "metric": "accuracy",
            }
        )

    task = "crpe_20"
    train_examples = _sample_train(examples, args.train_size, args.seed)
    task_template_dir = Path(args.template_root) / task
    output_dir = Path(args.output_root) / task
    template_summary = {
        "train": str(_write_template(task_template_dir, "train", train_examples)),
        "test": str(_write_template(task_template_dir, "test", examples)),
    }
    split_summary = {
        "train": _write_verl_split(output_dir, task, "train", train_examples),
        "test": _write_verl_split(output_dir, task, "test", examples),
    }
    summary = {
        "dataset": "crpe",
        "task": task,
        "source": f"OpenGVLab/CRPE {args.crpe_jsonl}",
        "source_rows": len(rows),
        "prepared_examples": len(examples),
        "missing_images": len(missing),
        "train_size": len(train_examples),
        "train_seed": args.seed,
        "template": template_summary,
        "splits": split_summary,
        "notes": [
            "TTRV Table 2/Appendix Table 8 corresponds to CRPE relation size about 7575; current HF crpe_relation.jsonl has 7576 circular-eval rows.",
            "CRPE relation mixes coco/val2017 images and abnormal_images; both must be present locally.",
        ],
    }
    if missing:
        summary["missing_preview"] = missing[:20]
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if missing:
        raise FileNotFoundError(f"CRPE has {len(missing)} missing images; see {output_dir / 'prepare_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CAPTURe/CRPE 20-shot data for the current TTRV/verl layout.")
    parser.add_argument("--datasets", nargs="+", default=["capture", "crpe"], choices=["capture", "crpe"])
    parser.add_argument("--data-root", default="/jiigan-hp/ttrv-datasets")
    parser.add_argument("--hf-home", default="/jiigan-hp/ttrv-datasets/hf_home")
    parser.add_argument("--template-root", default="verl/data")
    parser.add_argument("--output-root", default="/jiigan-hp/ttrv-datasets/verl_data")
    parser.add_argument("--capture-snapshot", default="")
    parser.add_argument("--crpe-jsonl", default="crpe_relation.jsonl")
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-size", type=int, default=1000)
    parser.add_argument("--summary-path", default="/jiigan-hp/ttrv-datasets/verl_data/capture_crpe20_prepare_summary.json")
    args = parser.parse_args()

    results = {}
    if "capture" in args.datasets:
        results["capture"] = _prepare_capture(args)
        print(json.dumps(results["capture"], ensure_ascii=False, indent=2))
    if "crpe" in args.datasets:
        results["crpe"] = _prepare_crpe(args)
        print(json.dumps(results["crpe"], ensure_ascii=False, indent=2))

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[prepare_capture_crpe20_data] summary={summary_path}")


if __name__ == "__main__":
    main()
