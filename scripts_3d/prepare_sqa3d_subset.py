#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from datasets import Dataset


PROMPT_VERSION = "sqa3d_short_answer_v3_ll3da_aligned"
PROMPT_TEMPLATE = "### human: {situation} {question} Give only the short answer. ### assistant:"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_answer(answers):
    if not answers:
        return ""
    for answer in answers:
        if str(answer.get("answer_confidence", "")).lower() == "yes":
            return answer.get("answer", "")
    return answers[0].get("answer", "")


def infer_question_type(question: str) -> str:
    first_token = str(question or "").strip().split()
    if not first_token:
        return "other"
    first = first_token[0].lower().strip("?:,.!")
    if first in {"what", "is", "how", "can", "which"}:
        return first
    return "other"


def parse_subset(value: str | int | None) -> int | None:
    if value is None:
        return 20
    if isinstance(value, int):
        return None if value <= 0 else value
    normalized = str(value).strip().lower()
    if normalized in {"all", "full", "-1", "0", "none"}:
        return None
    subset = int(normalized)
    return None if subset <= 0 else subset


def build_rows(data_dir: Path, scannet_dir: Path, subset: int | None, seed: int):
    questions_path = data_dir / "v1_balanced_questions_test_scannetv2.json"
    annotations_path = data_dir / "v1_balanced_sqa_annotations_test_scannetv2.json"
    if not questions_path.exists() or not annotations_path.exists():
        raise FileNotFoundError(
            f"Missing SQA3D files under {data_dir}. Expected {questions_path.name} and {annotations_path.name}."
        )

    questions = load_json(questions_path)["questions"]
    annotations = {item["question_id"]: item for item in load_json(annotations_path)["annotations"]}

    samples = []
    for question in questions:
        qid = question["question_id"]
        annotation = annotations.get(qid)
        if annotation is None:
            continue
        scene_id = question["scene_id"]
        pc_path = scannet_dir / f"{scene_id}_aligned_vert.npy"
        if not pc_path.exists():
            continue
        answers = [answer.get("answer", "") for answer in annotation.get("answers", []) if answer.get("answer")]
        samples.append(
            {
                "question_id": qid,
                "scene_id": scene_id,
                "situation": question.get("situation", ""),
                "question": question["question"],
                "question_type": infer_question_type(question["question"]),
                "answer": pick_answer(annotation.get("answers", [])),
                "answers": answers,
                "answer_type": annotation.get("answer_type", "other"),
                "pc_path": str(pc_path),
            }
        )

    random.seed(seed)
    random.shuffle(samples)
    if subset is None:
        return samples
    return samples[:subset]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(repo_root / "data/sqa3d_data"))
    parser.add_argument("--scannet-dir", default=str(repo_root / "data/scannet/scannet_data"))
    parser.add_argument("--output-file", default=str(repo_root / "data/sqa3d_test_subset20.parquet"))
    parser.add_argument("--subset", default="20", help="Number of rows to keep, or 'all' for every usable test row.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    scannet_dir = Path(args.scannet_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    subset = parse_subset(args.subset)
    rows = []
    for row_idx, sample in enumerate(build_rows(data_dir, scannet_dir, subset, args.seed)):
        rows.append(
            {
                "data_source": "sqa3d",
                "prompt": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(
                            situation=sample["situation"].strip(),
                            question=sample["question"].strip(),
                        ),
                    }
                ],
                "ability": "3d_situated_qa",
                "reward_model": {"style": "rule", "ground_truth": sample["answer"]},
                "extra_info": {
                    "pc_path": sample["pc_path"],
                    "question_id": int(sample["question_id"]),
                    "scene_id": sample["scene_id"],
                    "question_type": sample["question_type"],
                    "answer_type": sample["question_type"],
                    "sqa_answer_type": sample["answer_type"],
                    "answers": sample["answers"],
                    "index": f"sqa3d-{sample['question_id']}",
                    "row_index": row_idx,
                    "prompt_version": PROMPT_VERSION,
                },
            }
        )

    if subset is not None and len(rows) < subset:
        raise RuntimeError(f"Only found {len(rows)} SQA3D rows with existing point clouds; requested {subset}.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_file))
    print(f"Saved {len(rows)} SQA3D rows to {output_file}")


if __name__ == "__main__":
    main()
