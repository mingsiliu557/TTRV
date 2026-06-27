#!/usr/bin/env python
import argparse
import io
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import datasets
from PIL import Image as PILImage


OLD_PREFIX = "/home/anirban/kanksha1/realworldqa/images/"


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean_question(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<image>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Please answer directly.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Please respond.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Options are:.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Choices:.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n[A-D][\.:].*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


_OPTION_RE = re.compile(
    r"(?:^|\n)\s*([A-D])\s*[\.:)]\s*(.*?)(?=\n\s*[A-D]\s*[\.:)]|\n\s*Please|$)",
    re.DOTALL,
)


def _parse_options(prompt: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in _OPTION_RE.finditer(prompt):
        label = match.group(1).upper()
        value = re.sub(r"\s+", " ", match.group(2)).strip().strip(".")
        options[label] = value
    return options


def _normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = {"true": "yes", "false": "no"}.get(text, text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _answer_matches(local_answer: str, hf_answer: str) -> bool:
    if not local_answer or not hf_answer:
        return False
    if local_answer == hf_answer:
        return True
    return len(local_answer) > 1 and (local_answer in hf_answer or hf_answer in local_answer)


def _image_from_hf_value(value: Any) -> PILImage.Image:
    if isinstance(value, PILImage.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PILImage.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return PILImage.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(value)).convert("RGB")
    return PILImage.open(str(value)).convert("RGB")


def _resize_square(image: PILImage.Image, size: int) -> PILImage.Image:
    if image.size == (size, size):
        return image.convert("RGB")
    return image.convert("RGB").resize((size, size), PILImage.Resampling.LANCZOS)


def _target_path(original_image_path: str, image_root: Path) -> Path:
    if original_image_path.startswith(OLD_PREFIX):
        rel = original_image_path[len(OLD_PREFIX) :]
    else:
        rel = Path(original_image_path).name
    return image_root / rel


def _load_hf_rows(cache_dir: Path) -> list[dict[str, Any]]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset("xai-org/RealworldQA", split="test", cache_dir=str(cache_dir))
    ds = ds.cast_column("image", HFImage(decode=False))
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "hf_index": idx,
                "question": row["question"],
                "question_key": _clean_question(row["question"]),
                "answer": row["answer"],
                "answer_key": _normalize_answer(row["answer"]),
                "image": row["image"],
            }
        )
    return rows


def _match_row(
    row: dict[str, Any],
    hf_rows: list[dict[str, Any]],
    hf_by_question: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], str, float]:
    prompt = row["prompt"]
    question_key = _clean_question(prompt)
    answer_label = str(row["answer"]).upper()
    options = _parse_options(prompt)
    option_answer = _normalize_answer(options.get(answer_label, ""))

    candidates = hf_by_question.get(question_key, [])
    if candidates:
        answer_matched = [
            item
            for item in candidates
            if _answer_matches(option_answer, item["answer_key"]) or item["answer_key"] == answer_label.lower()
        ]
        if answer_matched:
            return answer_matched[0], "exact_question_answer", 1.0
        return candidates[0], "exact_question_only", 1.0

    best: tuple[float, float, float, bool, dict[str, Any]] | None = None
    local_tokens = set(question_key.split())
    for item in hf_rows:
        question_ratio = SequenceMatcher(None, question_key, item["question_key"]).ratio()
        hf_tokens = set(item["question_key"].split())
        jaccard = len(local_tokens & hf_tokens) / max(1, len(local_tokens | hf_tokens))
        answer_match = _answer_matches(option_answer, item["answer_key"]) or item["answer_key"] == answer_label.lower()
        score = 0.7 * max(question_ratio, jaccard) + 0.3 * float(answer_match)
        if best is None or score > best[0]:
            best = (score, question_ratio, jaccard, answer_match, item)

    if best is None:
        raise RuntimeError(f"no HF rows available for template row id={row.get('id')}")
    score, question_ratio, jaccard, answer_match, item = best
    if answer_match and (question_ratio >= 0.62 or jaccard >= 0.42):
        return item, "fuzzy_question_answer", score
    if question_ratio >= 0.93:
        return item, "fuzzy_high_question", score
    raise RuntimeError(
        "failed to align RealWorldQA row "
        f"id={row.get('id')} answer={row.get('answer')} question={question_key!r}; "
        f"best_hf_index={item['hf_index']} best_question={item['question_key']!r} "
        f"best_answer={item['answer_key']!r} score={score:.4f} "
        f"question_ratio={question_ratio:.4f} jaccard={jaccard:.4f} answer_match={answer_match}"
    )


def _write_split(
    split: str,
    rows: list[dict[str, Any]],
    hf_rows: list[dict[str, Any]],
    hf_by_question: dict[str, list[dict[str, Any]]],
    output_task_dir: Path,
    image_root: Path,
    resize_size: int,
    overwrite_images: bool,
) -> dict[str, Any]:
    output_rows = []
    method_counts: Counter[str] = Counter()
    manifest_rows = []

    for idx, row in enumerate(rows):
        hf_row, method, score = _match_row(row, hf_rows, hf_by_question)
        method_counts[method] += 1
        target = _target_path(row["image_path"], image_root)
        if overwrite_images or not target.exists():
            image = _resize_square(_image_from_hf_value(hf_row["image"]), resize_size)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, format="PNG")

        item = {
            "data_source": "GPQA-TTT",
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": row["answer"]},
            "extra_info": {
                "split": split,
                "index": f"realworldqa_20_aligned_1000-{split}-{idx}",
                "dataset": "realworldqa_20_aligned_1000",
                "id": row.get("id"),
                "source": row.get("source"),
                "image_path": str(target),
                "original_image_path": row["image_path"],
                "hf_realworldqa_index": hf_row["hf_index"],
                "hf_question": hf_row["question"],
                "hf_answer": hf_row["answer"],
                "alignment_method": method,
                "alignment_score": score,
                "resize_policy": "direct_rgb_lanczos_square_1000",
                "resize_size": resize_size,
                "image_exists": target.exists(),
            },
            "images": [{"image": str(target)}],
        }
        output_rows.append(item)
        manifest_rows.append(
            {
                "split": split,
                "row": idx,
                "id": row.get("id"),
                "answer": row["answer"],
                "image": str(target),
                "hf_realworldqa_index": hf_row["hf_index"],
                "alignment_method": method,
                "alignment_score": score,
                "prompt_question_key": _clean_question(row["prompt"]),
                "hf_question_key": hf_row["question_key"],
                "hf_answer": hf_row["answer"],
            }
        )

    output_task_dir.mkdir(parents=True, exist_ok=True)
    with (output_task_dir / f"{split}.json").open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(output_rows).to_parquet(str(output_task_dir / f"{split}.parquet"))
    with (output_task_dir / f"{split}_alignment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, ensure_ascii=False, indent=2)
    return {
        "num_examples": len(output_rows),
        "json": str(output_task_dir / f"{split}.json"),
        "parquet": str(output_task_dir / f"{split}.parquet"),
        "alignment_manifest": str(output_task_dir / f"{split}_alignment_manifest.json"),
        "alignment_methods": dict(method_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RealWorldQA data with HF-question aligned images resized to the TTRV 1000x1000 protocol."
    )
    parser.add_argument("--template-dir", default="verl/data/realworldqa_20")
    parser.add_argument("--output-dir", default="/jiigan-hp/ttrv-datasets/verl_data/realworldqa_20_aligned_1000")
    parser.add_argument("--image-root", default="/jiigan-hp/ttrv-datasets/realworldqa_aligned_1000/images")
    parser.add_argument("--cache-dir", default="/jiigan-hp/ttrv-datasets/hf_home")
    parser.add_argument("--resize-size", type=int, default=1000)
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()

    template_dir = Path(args.template_dir)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    hf_rows = _load_hf_rows(Path(args.cache_dir))
    hf_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hf_rows:
        hf_by_question[row["question_key"]].append(row)

    summary = {
        "dataset": "realworldqa_20_aligned_1000",
        "template_dir": str(template_dir),
        "output_dir": str(output_dir),
        "image_root": str(image_root),
        "hf_cache_dir": args.cache_dir,
        "resize_size": args.resize_size,
        "splits": {},
    }
    for split in ("train", "test"):
        rows = _load_json(template_dir / f"{split}.json")
        summary["splits"][split] = _write_split(
            split,
            rows,
            hf_rows,
            hf_by_question,
            output_dir,
            image_root,
            args.resize_size,
            args.overwrite_images,
        )

    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
