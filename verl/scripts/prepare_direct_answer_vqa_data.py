#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import datasets
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.utils.reward_score.ttrl.direct_answer import normalize_direct_answer


VISUAL_REPO = "WYLing/VisualSimpleQA"
OCRBENCH_REPO = "echo840/OCRBench"
AOKVQA_URL = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
AOKVQA_HF_REPO = "HuggingFaceM4/A-OKVQA"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"


DATASET_SPECS = {
    "visualsimpleqa": {
        "source_task": "visualsimpleqa_da",
        "pool_task": "visualsimpleqa_da_hq_pool80_seed42",
        "data_source": "VisualSimpleQA-TTT",
        "metric": "visualsimpleqa",
    },
    "ocrbench_v1": {
        "source_task": "ocrbench_v1",
        "pool_task": "ocrbench_v1_hq_pool80_seed42",
        "data_source": "OCRBench-TTT",
        "metric": "ocrbench",
    },
    "aokvqa_da_val": {
        "source_task": "aokvqa_da_val",
        "pool_task": "aokvqa_da_val_hq_pool80_seed42",
        "data_source": "AOKVQA-DA-TTT",
        "metric": "aokvqa_direct_answer",
    },
}


def configure_proxy(proxy: str | None) -> None:
    if not proxy:
        return
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_stem(value: Any, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "").strip())
    return text[:120].strip("._") or fallback


def save_image(value: Any, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)

    image = None
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict):
        if value.get("bytes"):
            image = Image.open(BytesIO(value["bytes"]))
        elif value.get("path"):
            src = Path(value["path"])
            if src.exists():
                shutil.copy2(src, output_path)
                return str(output_path)
    elif isinstance(value, (str, os.PathLike)):
        src = Path(value)
        if src.exists():
            shutil.copy2(src, output_path)
            return str(output_path)

    if image is None:
        raise ValueError(f"Unsupported image value for {output_path}: {type(value)!r}")
    image.convert("RGB").save(output_path, quality=95)
    return str(output_path)


def download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(output_path)
    return output_path


def extract_tar_once(path: Path, output_dir: Path) -> None:
    marker = output_dir / ".extract_complete"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:*") as tf:
        tf.extractall(output_dir)
    marker.write_text("ok\n")


def extract_zip_once(path: Path, output_dir: Path) -> None:
    marker = output_dir / ".extract_complete"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(output_dir)
    marker.write_text("ok\n")


def answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text[0] in "[{":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            parsed_answers = answer_list(parsed)
            if parsed_answers:
                return parsed_answers
    return [text]


def primary_answer(answers: list[str]) -> str:
    normalized = [normalize_direct_answer(answer) for answer in answers]
    normalized = [answer for answer in normalized if answer != "unknown"]
    if not normalized:
        return ""
    return Counter(normalized).most_common(1)[0][0]


def prompt_text(question: str, *, ocr: bool = False) -> str:
    suffix = "Answer with the exact text from the image." if ocr else "Answer with a short direct answer."
    return f"<image>\n{question.strip()}\n{suffix}"


def to_verl_rows(task: str, split: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(examples):
        image_path = ex["image_path"]
        exists = bool(image_path) and Path(image_path).exists()
        extra_info = {
            "split": split,
            "index": f"{task}-{split}-{idx}",
            "dataset": task,
            "source_dataset": ex.get("source_dataset"),
            "id": ex.get("id"),
            "question_id": ex.get("question_id"),
            "image_path": image_path,
            "image_exists": exists,
            "metric": ex["metric"],
            "official_answers": ex["official_answers"],
            "raw_answer": ex.get("raw_answer"),
            "category": ex.get("category"),
            "question_type": ex.get("question_type"),
        }
        rows.append(
            {
                "data_source": ex["data_source"],
                "prompt": [{"role": "user", "content": ex["prompt"]}],
                "ability": "vqa",
                "reward_model": {"style": "rule", "ground_truth": ex["answer"]},
                "extra_info": extra_info,
                "images": [{"image": image_path}] if exists else None,
            }
        )
    return rows


def write_task(output_root: Path, task: str, train_examples: list[dict[str, Any]], test_examples: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = to_verl_rows(task, "train", train_examples)
    test_rows = to_verl_rows(task, "test", test_examples)
    write_json(output_dir / "train.json", train_rows)
    write_json(output_dir / "test.json", test_rows)
    datasets.Dataset.from_list(train_rows).to_parquet(str(output_dir / "train.parquet"))
    datasets.Dataset.from_list(test_rows).to_parquet(str(output_dir / "test.parquet"))
    task_summary = {
        **summary,
        "task": task,
        "output_dir": str(output_dir),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "missing_train_images": sum(1 for row in train_rows if not row["extra_info"]["image_exists"]),
        "missing_test_images": sum(1 for row in test_rows if not row["extra_info"]["image_exists"]),
    }
    write_json(output_dir / "prepare_summary.json", task_summary)
    return task_summary


def choose_examples(examples: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return [examples[i] for i in sorted(rng.sample(range(len(examples)), size))]


def materialize_tasks(dataset_key: str, examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_key]
    source_train = choose_examples(examples, args.train_size, args.seed)
    pool_train = choose_examples(examples, args.pool_size, args.seed)
    common = {
        "dataset_key": dataset_key,
        "selection_seed": args.seed,
        "note": "Pool task is for base 32-sample majority scan; final HQ20 task is produced after scan.",
    }
    source_summary = write_task(
        Path(args.output_root),
        spec["source_task"],
        source_train,
        examples,
        {**common, "selection_rule": f"seeded source train sample size {len(source_train)}"},
    )
    pool_summary = write_task(
        Path(args.output_root),
        spec["pool_task"],
        pool_train,
        examples,
        {**common, "selection_rule": f"seeded HQ candidate pool size {len(pool_train)}"},
    )
    return {"source": source_summary, "pool": pool_summary}


def prepare_visualsimpleqa(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = DATASET_SPECS["visualsimpleqa"]
    visual_config = args.visualsimpleqa_config.strip()
    if args.visualsimpleqa_local_parquet:
        ds = datasets.load_dataset(
            "parquet",
            data_files=str(Path(args.visualsimpleqa_local_parquet)),
            split="train",
            cache_dir=args.hf_home,
        )
        config_name = f"local_{Path(args.visualsimpleqa_local_parquet).stem}"
        source_dataset = str(Path(args.visualsimpleqa_local_parquet))
    elif visual_config.lower() in {"", "none", "null"}:
        ds = datasets.load_dataset(
            args.visualsimpleqa_repo,
            split=args.visualsimpleqa_split,
            cache_dir=args.hf_home,
        )
        config_name = "default"
        source_dataset = f"{args.visualsimpleqa_repo}/{config_name}"
    else:
        ds = datasets.load_dataset(
            args.visualsimpleqa_repo,
            visual_config,
            split=args.visualsimpleqa_split,
            cache_dir=args.hf_home,
        )
        config_name = visual_config
        source_dataset = f"{args.visualsimpleqa_repo}/{config_name}"
    image_dir = Path(args.data_root) / "VisualSimpleQA" / safe_stem(args.visualsimpleqa_repo, "repo") / config_name / "images"
    examples = []
    for idx, row in enumerate(ds):
        message = {}
        raw_messages = row.get("messages")
        if raw_messages:
            try:
                parsed_messages = json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages
                if isinstance(parsed_messages, list) and parsed_messages:
                    message = parsed_messages[0] or {}
                elif isinstance(parsed_messages, dict):
                    message = parsed_messages
            except Exception:
                message = {}
        question = (
            row.get("multimodal_question")
            or row.get("question")
            or row.get("prompt")
            or row.get("query")
            or message.get("question")
        )
        answers = answer_list(row.get("answer") or row.get("answers") or message.get("answer") or message.get("answers"))
        image_value = row.get("cropped_image") if args.visualsimpleqa_use_cropped and row.get("cropped_image") is not None else row.get("image")
        if image_value is None:
            media = row.get("media")
            if isinstance(media, (list, tuple)) and media:
                image_value = media[0]
            else:
                image_value = media
        if not question or not answers or image_value is None:
            continue
        image_path = save_image(image_value, image_dir / f"{idx:06d}.jpg")
        examples.append(
            {
                "id": row.get("id") or idx,
                "source_dataset": source_dataset,
                "data_source": spec["data_source"],
                "metric": spec["metric"],
                "prompt": prompt_text(str(question)),
                "answer": primary_answer(answers),
                "official_answers": answers,
                "raw_answer": row.get("answer") or message.get("answer"),
                "image_path": image_path,
                "category": row.get("category") or message.get("category"),
            }
        )
    return examples


def prepare_ocrbench(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = DATASET_SPECS["ocrbench_v1"]
    ds = datasets.load_dataset(OCRBENCH_REPO, split=args.ocrbench_split, cache_dir=args.hf_home)
    image_dir = Path(args.data_root) / "OCRBench_v1" / "images"
    examples = []
    for idx, row in enumerate(ds):
        question = row.get("question")
        answers = answer_list(row.get("answer") or row.get("answers"))
        image_value = row.get("image")
        if not question or not answers or image_value is None:
            continue
        image_name = f"{safe_stem(row.get('dataset'), 'ocrbench')}_{idx:06d}.jpg"
        image_path = save_image(image_value, image_dir / image_name)
        examples.append(
            {
                "id": idx,
                "source_dataset": OCRBENCH_REPO,
                "data_source": spec["data_source"],
                "metric": spec["metric"],
                "prompt": prompt_text(str(question), ocr=True),
                "answer": primary_answer(answers),
                "official_answers": answers,
                "raw_answer": row.get("answer"),
                "image_path": image_path,
                "category": row.get("dataset"),
                "question_type": row.get("question_type"),
            }
        )
    return examples


def ensure_aokvqa_annotations(args: argparse.Namespace) -> Path:
    if args.aokvqa_val_json:
        return Path(args.aokvqa_val_json)
    root = Path(args.data_root) / "A-OKVQA"
    candidates = [
        root / "aokvqa_v1p0_val.json",
        root / "aokvqa_v1p0" / "aokvqa_v1p0_val.json",
        root / "annotations" / "aokvqa_v1p0_val.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    archive = download_file(AOKVQA_URL, root / "aokvqa_v1p0.tar.gz")
    extract_tar_once(archive, root)
    for path in candidates:
        if path.exists():
            return path
    found = list(root.rglob("aokvqa_v1p0_val.json"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Could not find aokvqa_v1p0_val.json under {root}")


def _image_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"})


def ensure_coco_val(args: argparse.Namespace) -> Path:
    candidates = [
        Path(args.coco_val_dir) if args.coco_val_dir else None,
        Path(args.data_root) / "COCO" / "val2017",
        Path(args.data_root) / "coco" / "val2017",
        Path(args.data_root) / "CRPE" / "coco" / "val2017",
    ]
    for path in candidates:
        if path and _image_file_count(path) >= args.min_coco_val_images:
            return path
    root = Path(args.data_root) / "COCO"
    archive = download_file(COCO_VAL_URL, root / "val2017.zip")
    extract_zip_once(archive, root)
    path = root / "val2017"
    if _image_file_count(path) < args.min_coco_val_images:
        raise FileNotFoundError(f"Could not prepare full COCO val2017 under {root}; found {_image_file_count(path)} images")
    return path


def prepare_aokvqa_da_val_from_hf(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = DATASET_SPECS["aokvqa_da_val"]
    if args.aokvqa_local_parquet:
        ds = datasets.load_dataset(
            "parquet",
            data_files=str(Path(args.aokvqa_local_parquet)),
            split="train",
            cache_dir=args.hf_home,
        )
        source_dataset = str(Path(args.aokvqa_local_parquet))
    else:
        ds = datasets.load_dataset(AOKVQA_HF_REPO, split="validation", cache_dir=args.hf_home)
        source_dataset = AOKVQA_HF_REPO
    image_dir = Path(args.data_root) / "A-OKVQA" / "hf_validation_images"
    examples = []
    for idx, row in enumerate(ds):
        if bool(row.get("difficult_direct_answer", False)):
            continue
        question = row.get("question") or row.get("sent") or row.get("text")
        answers = answer_list(
            row.get("direct_answers")
            or row.get("direct_answer")
            or row.get("answers")
            or row.get("answer")
        )
        image_value = row.get("image") or row.get("img")
        if not question or not answers or image_value is None:
            continue
        image_path = save_image(image_value, image_dir / f"{idx:06d}.jpg")
        examples.append(
            {
                "id": row.get("question_id") or idx,
                "question_id": row.get("question_id"),
                "source_dataset": source_dataset,
                "data_source": spec["data_source"],
                "metric": spec["metric"],
                "prompt": prompt_text(str(question)),
                "answer": primary_answer(answers),
                "official_answers": answers,
                "raw_answer": answers,
                "image_path": image_path,
            }
        )
    return examples


def prepare_aokvqa_da_val(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec = DATASET_SPECS["aokvqa_da_val"]
    if args.aokvqa_source == "hf":
        return prepare_aokvqa_da_val_from_hf(args)
    try:
        val_json = ensure_aokvqa_annotations(args)
        coco_val = ensure_coco_val(args)
    except Exception as exc:
        if args.aokvqa_source == "official":
            raise
        print(f"[prepare_aokvqa_da_val] official download failed, using HF fallback: {exc}", flush=True)
        return prepare_aokvqa_da_val_from_hf(args)
    rows = load_json(val_json)
    examples = []
    for idx, row in enumerate(rows):
        if row.get("difficult_direct_answer", False):
            continue
        answers = answer_list(row.get("direct_answers"))
        if not answers:
            continue
        image_id = int(row["image_id"])
        image_path = coco_val / f"{image_id:012d}.jpg"
        if not image_path.exists():
            continue
        examples.append(
            {
                "id": row.get("question_id") or idx,
                "question_id": row.get("question_id"),
                "source_dataset": "allenai/aokvqa",
                "data_source": spec["data_source"],
                "metric": spec["metric"],
                "prompt": prompt_text(str(row["question"])),
                "answer": primary_answer(answers),
                "official_answers": answers,
                "raw_answer": answers,
                "image_path": str(image_path),
            }
        )
    if args.aokvqa_source == "auto" and len(examples) < args.min_aokvqa_examples:
        print(f"[prepare_aokvqa_da_val] official path produced only {len(examples)} examples, using HF fallback", flush=True)
        return prepare_aokvqa_da_val_from_hf(args)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare direct-answer VQA datasets for TTRV/verl.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS), choices=list(DATASET_SPECS))
    parser.add_argument("--data-root", default="/jiigan-hp/ttrv-datasets")
    parser.add_argument("--output-root", default="/jiigan-hp/ttrv-datasets/verl_data")
    parser.add_argument("--hf-home", default="/jiigan-hp/ttrv-datasets/hf_home")
    parser.add_argument("--proxy", default="127.0.0.1:7892")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=80)
    parser.add_argument("--summary-path", default="/jiigan-hp/ttrv-datasets/verl_data/direct_answer_vqa_prepare_summary.json")
    parser.add_argument("--visualsimpleqa-repo", default=VISUAL_REPO)
    parser.add_argument("--visualsimpleqa-config", default="VisualSimpleQA")
    parser.add_argument("--visualsimpleqa-split", default="test")
    parser.add_argument("--visualsimpleqa-use-cropped", action="store_true")
    parser.add_argument("--visualsimpleqa-local-parquet", default="", help="Read VisualSimpleQA from a pre-downloaded local parquet file.")
    parser.add_argument("--ocrbench-split", default="test")
    parser.add_argument("--aokvqa-val-json", default="")
    parser.add_argument("--aokvqa-source", default="auto", choices=["auto", "official", "hf"])
    parser.add_argument("--aokvqa-local-parquet", default="", help="Read A-OKVQA validation from a pre-downloaded local parquet file with embedded images.")
    parser.add_argument("--min-aokvqa-examples", type=int, default=500)
    parser.add_argument("--coco-val-dir", default="")
    parser.add_argument("--min-coco-val-images", type=int, default=4000)
    args = parser.parse_args()

    configure_proxy(args.proxy)
    os.environ.setdefault("HF_HOME", args.hf_home)

    preparers = {
        "visualsimpleqa": prepare_visualsimpleqa,
        "ocrbench_v1": prepare_ocrbench,
        "aokvqa_da_val": prepare_aokvqa_da_val,
    }
    summary = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "hf_home": args.hf_home,
        "datasets": {},
    }
    for key in args.datasets:
        examples = preparers[key](args)
        if not examples:
            raise RuntimeError(f"{key}: no valid examples prepared")
        summary["datasets"][key] = {
            "num_examples": len(examples),
            **materialize_tasks(key, examples, args),
        }
        print(json.dumps({key: summary["datasets"][key]}, ensure_ascii=False, indent=2))
    write_json(Path(args.summary_path), summary)
    print(f"[prepare_direct_answer_vqa_data] summary={args.summary_path}")


if __name__ == "__main__":
    main()
