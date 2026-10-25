#!/usr/bin/env python3
"""Prepare held-out real/shuffle/blank inputs for diagnostic Stage A."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    task: str
    parser: str
    labels: str


BENCHMARKS = (
    BenchmarkSpec("mme20", "mme_20", "legacy", "AB"),
    BenchmarkSpec("ai2d20", "ai2d_20", "legacy", "A-D"),
    BenchmarkSpec("seed20", "seed_20", "legacy", "A-D"),
    BenchmarkSpec("mathvista20_hq", "mathvista_20_hq_train13_repeat20", "legacy", "A-F"),
    BenchmarkSpec("mathverse20_fixed", "mathverse_20_choice_norm_promptfix", "mathvista_choice", "A-F"),
    BenchmarkSpec(
        "realworldqa20_hq",
        "realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652",
        "legacy",
        "A-D",
    ),
    BenchmarkSpec("visualsimpleqa", "visualsimpleqa_da_20_hq_major_correct", "direct_answer_short", "A-D"),
    BenchmarkSpec("ocrbench_v1", "ocrbench_v1_20_hq_major_correct", "direct_answer_short", "A-D"),
    BenchmarkSpec("aokvqa_da_val", "aokvqa_da_val_20_hq_major_correct", "direct_answer_short", "A-D"),
)
SPEC_BY_KEY = {spec.key: spec for spec in BENCHMARKS}


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_qid(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}
    for key in ("question_id", "id", "source_index", "index"):
        value = extra.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"Row has no stable question id: {extra}")


def _row_image_path(row: dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict) and first.get("image"):
            return str(first["image"])
        if first:
            return str(first)
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("image_path"):
        return str(extra["image_path"])
    raise ValueError(f"Row {_stable_qid(row)} has no image path")


def _replace_image(row: dict[str, Any], image_path: str) -> None:
    images = row.get("images")
    if isinstance(images, list) and images:
        if isinstance(images[0], dict):
            images[0]["image"] = image_path
        else:
            images[0] = image_path
    else:
        row["images"] = [{"image": image_path}]


def _condition_row(
    row: dict[str, Any],
    *,
    benchmark: str,
    qid: str,
    condition: str,
    target_image: str,
    effective_image: str,
    subset_position: int,
    subset_seed: int,
    shuffle_src_qid: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(row)
    extra = result.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}
        result["extra_info"] = extra
    extra.update(
        {
            "diagnostic_benchmark": benchmark,
            "diagnostic_question_id": qid,
            "diagnostic_condition": condition,
            "diagnostic_subset_position": int(subset_position),
            "diagnostic_subset_seed": int(subset_seed),
            "diagnostic_target_image_path": target_image,
            "diagnostic_effective_image_path": effective_image,
            "diagnostic_shuffle_src_qid": shuffle_src_qid,
        }
    )
    extra["image_path"] = effective_image
    _replace_image(result, effective_image)
    return result


def _blank_image(source: str, destination: Path) -> tuple[int, int]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as image:
        size = tuple(image.size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        Image.new("RGB", size, (255, 255, 255)).save(destination)
    else:
        with Image.open(destination) as existing:
            if tuple(existing.size) != size or existing.mode != "RGB":
                raise ValueError(f"Invalid existing blank image: {destination}")
    return int(size[0]), int(size[1])


def _derangement(rows: list[dict[str, Any]], seed: int) -> list[int]:
    qids = [_stable_qid(row) for row in rows]
    images = [_row_image_path(row) for row in rows]
    rng = random.Random(seed)
    candidate = list(range(len(rows)))
    for _ in range(10000):
        rng.shuffle(candidate)
        if all(qids[i] != qids[j] and images[i] != images[j] for i, j in enumerate(candidate)):
            return list(candidate)
    raise RuntimeError("Unable to construct a qid- and image-disjoint shuffle derangement")


def _prepare_one(
    spec: BenchmarkSpec,
    *,
    data_root: Path,
    output_root: Path,
    subset_size: int,
    subset_seed: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    task_root = data_root / spec.task
    train_path = task_root / "train.parquet"
    test_path = task_root / "test.parquet"
    train_rows = _read_parquet(train_path)
    test_rows = _read_parquet(test_path)

    train_ids = {_stable_qid(row) for row in train_rows}
    eligible = [row for row in test_rows if _stable_qid(row) not in train_ids]
    eligible.sort(key=_stable_qid)
    if len(eligible) < subset_size:
        raise ValueError(f"{spec.key}: requested {subset_size}, only {len(eligible)} held-out rows")

    selected = random.Random(subset_seed).sample(eligible, subset_size)
    selected_ids = [_stable_qid(row) for row in selected]
    if len(set(selected_ids)) != subset_size:
        raise ValueError(f"{spec.key}: sampled duplicate question ids")
    if set(selected_ids) & train_ids:
        raise ValueError(f"{spec.key}: sampled subset overlaps train ids")

    pairing = _derangement(selected, shuffle_seed)
    combined_rows: list[dict[str, Any]] = []
    pairing_records: list[dict[str, Any]] = []
    blank_records: list[dict[str, Any]] = []
    blank_root = output_root / "prepared" / "blank_images" / spec.key

    for position, (target, source_position) in enumerate(zip(selected, pairing)):
        source = selected[source_position]
        qid = _stable_qid(target)
        source_qid = _stable_qid(source)
        target_image = _row_image_path(target)
        source_image = _row_image_path(source)
        if not Path(target_image).exists() or not Path(source_image).exists():
            raise FileNotFoundError(f"{spec.key}: missing target/source image for {qid}")

        digest = hashlib.sha1(f"{spec.key}:{qid}:{target_image}".encode("utf-8")).hexdigest()[:24]
        blank_path = blank_root / f"{digest}.png"
        blank_size = _blank_image(target_image, blank_path)

        combined_rows.append(
            _condition_row(
                target,
                benchmark=spec.key,
                qid=qid,
                condition="real",
                target_image=target_image,
                effective_image=target_image,
                subset_position=position,
                subset_seed=subset_seed,
            )
        )
        combined_rows.append(
            _condition_row(
                target,
                benchmark=spec.key,
                qid=qid,
                condition="shuffle",
                target_image=target_image,
                effective_image=source_image,
                subset_position=position,
                subset_seed=subset_seed,
                shuffle_src_qid=source_qid,
            )
        )
        combined_rows.append(
            _condition_row(
                target,
                benchmark=spec.key,
                qid=qid,
                condition="blank",
                target_image=target_image,
                effective_image=str(blank_path),
                subset_position=position,
                subset_seed=subset_seed,
            )
        )
        pairing_records.append(
            {
                "target_qid": qid,
                "shuffle_src_qid": source_qid,
                "target_image_path": target_image,
                "shuffle_image_path": source_image,
            }
        )
        blank_records.append(
            {
                "question_id": qid,
                "source_image_path": target_image,
                "blank_image_path": str(blank_path),
                "size": list(blank_size),
            }
        )

    prepared_dir = output_root / "prepared" / spec.key
    prepared_dir.mkdir(parents=True, exist_ok=True)
    combined_path = prepared_dir / "combined_conditions.parquet"
    real_path = prepared_dir / "real.parquet"
    counterfactual_path = prepared_dir / "shuffle_blank.parquet"
    real_rows = [
        row for row in combined_rows if row["extra_info"]["diagnostic_condition"] == "real"
    ]
    counterfactual_rows = [
        row for row in combined_rows if row["extra_info"]["diagnostic_condition"] in {"shuffle", "blank"}
    ]
    pd.DataFrame(combined_rows).to_parquet(combined_path, index=False)
    pd.DataFrame(real_rows).to_parquet(real_path, index=False)
    pd.DataFrame(counterfactual_rows).to_parquet(counterfactual_path, index=False)

    _write_json(output_root / f"subset_{spec.key}.json", selected_ids)
    _write_json(
        output_root / f"shuffle_pairing_{spec.key}.json",
        {
            "benchmark": spec.key,
            "seed": shuffle_seed,
            "pairs": pairing_records,
        },
    )
    _write_json(prepared_dir / "blank_images.json", blank_records)

    return {
        **asdict(spec),
        "train_parquet": str(train_path),
        "source_test_parquet": str(test_path),
        "combined_parquet": str(combined_path),
        "real_parquet": str(real_path),
        "counterfactual_parquet": str(counterfactual_path),
        "train_rows": len(train_rows),
        "train_unique_ids": len(train_ids),
        "test_rows": len(test_rows),
        "eligible_rows": len(eligible),
        "selected_questions": len(selected_ids),
        "prepared_rows": len(combined_rows),
        "real_rows": len(real_rows),
        "counterfactual_rows": len(counterfactual_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/ttrv/verl_data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=250)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(SPEC_BY_KEY), default=None)
    args = parser.parse_args()

    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    keys = args.benchmarks or [spec.key for spec in BENCHMARKS]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        _prepare_one(
            SPEC_BY_KEY[key],
            data_root=args.data_root,
            output_root=args.output_root,
            subset_size=args.subset_size,
            subset_seed=args.subset_seed,
            shuffle_seed=args.shuffle_seed,
        )
        for key in keys
    ]
    _write_json(
        args.output_root / "preparation_summary.json",
        {
            "subset_size": args.subset_size,
            "subset_seed": args.subset_seed,
            "shuffle_seed": args.shuffle_seed,
            "benchmarks": summaries,
        },
    )

    manifest = args.output_root / "run_manifest.tsv"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write(
            "benchmark\ttask\tparser\tlabels\ttrain_parquet\treal_parquet"
            "\tcounterfactual_parquet\texpected_questions\n"
        )
        for item in summaries:
            handle.write(
                "\t".join(
                    [
                        item["key"],
                        item["task"],
                        item["parser"],
                        item["labels"],
                        item["train_parquet"],
                        item["real_parquet"],
                        item["counterfactual_parquet"],
                        str(item["selected_questions"]),
                    ]
                )
                + "\n"
            )

    for item in summaries:
        print(
            f"{item['key']}: selected={item['selected_questions']} "
            f"excluded_train_unique={item['train_unique_ids']} prepared={item['prepared_rows']}"
        )
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
