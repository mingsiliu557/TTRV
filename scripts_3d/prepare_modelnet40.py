#!/usr/bin/env python3
import argparse
import os
import pickle
import urllib.request
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/.cache")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/.cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/.cache/huggingface/datasets")
os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/.cache/torch")
os.environ.setdefault("PIP_CACHE_DIR", "/root/autodl-tmp/.cache/pip")
os.environ.pop("TRANSFORMERS_CACHE", None)

import numpy as np
from datasets import Dataset


MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]

DEFAULT_URL = "https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/modelnet40_test_8192pts_fps.dat"
POINT_TOKEN_LEN = 513
POINT_PROMPT = "<point_start>" + ("<point_patch>" * POINT_TOKEN_LEN) + "<point_end>\nWhat is this?"


def pc_norm(pc: np.ndarray) -> np.ndarray:
    xyz = pc[:, :3].astype(np.float32, copy=True)
    xyz -= np.mean(xyz, axis=0)
    scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if scale > 0:
        xyz /= scale
    return xyz


def load_modelnet_dat(path: Path):
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        points = next((obj[key] for key in ("data", "points", "list_of_points") if key in obj), None)
        labels = next((obj[key] for key in ("label", "labels", "list_of_labels") if key in obj), None)
    else:
        points, labels = obj
    if points is None or labels is None:
        raise ValueError(f"Could not find points/labels in {path}")
    return list(points), [int(np.asarray(label).reshape(-1)[0]) for label in labels]


def ensure_data_file(path: Path, url: str):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def choose_indices(labels: list[int], subset: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_label = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    for candidates in by_label.values():
        rng.shuffle(candidates)

    selected = []
    for label in sorted(by_label):
        if len(selected) >= subset:
            break
        selected.append(by_label[label][0])

    if len(selected) < subset:
        remaining = [idx for idx in range(len(labels)) if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: subset - len(selected)])
    return selected


def make_point_cloud(point_set: np.ndarray) -> np.ndarray:
    xyz = pc_norm(np.asarray(point_set)[:, :3])
    return np.concatenate([xyz, np.zeros_like(xyz, dtype=np.float32)], axis=1).astype(np.float32, copy=False)


def build_rows(points, labels, indices: list[int], pc_dir: Path, split: str):
    rows = []
    for row_idx, source_idx in enumerate(indices):
        label_id = labels[source_idx]
        label_name = MODELNET40_CLASSES[label_id]
        pc = make_point_cloud(points[source_idx])
        pc_path = pc_dir / f"source_{source_idx:04d}_label_{label_id:02d}.npy"
        np.save(pc_path, pc)
        rows.append(
            {
                "data_source": "modelnet40_freeform",
                "prompt": [{"role": "user", "content": POINT_PROMPT}],
                "ability": "3d_classification",
                "reward_model": {"style": "rule", "ground_truth": label_name},
                "extra_info": {
                    "pc_path": str(pc_path),
                    "label_id": label_id,
                    "label_name": label_name,
                    "index": f"modelnet40-{source_idx:04d}",
                    "source_index": int(source_idx),
                    "split": split,
                    "row_index": int(row_idx),
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-file", default=str(repo_root / "data/modelnet40_data/modelnet40_test_8192pts_fps.dat"))
    parser.add_argument("--output-dir", default=str(repo_root / "data/modelnet40_20"))
    parser.add_argument("--subset", type=int, default=20, help="Number of adaptation/train samples to draw.")
    parser.add_argument(
        "--val-subset",
        type=int,
        default=0,
        help="Number of validation samples to draw. Use 0 or a negative value for the full official test split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    data_file = Path(args.data_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    pc_dir = output_dir / "pc_npy"
    ensure_data_file(data_file, args.url)

    points, labels = load_modelnet_dat(data_file)
    train_indices = choose_indices(labels, args.subset, args.seed)
    if args.val_subset and args.val_subset > 0:
        test_indices = choose_indices(labels, args.val_subset, args.seed)
    else:
        test_indices = list(range(len(labels)))
    pc_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_rows(points, labels, train_indices, pc_dir, "train")
    test_rows = build_rows(points, labels, test_indices, pc_dir, "test")
    train_dataset = Dataset.from_list(train_rows)
    test_dataset = Dataset.from_list(test_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    train_covered = len({row["extra_info"]["label_id"] for row in train_rows})
    test_covered = len({row["extra_info"]["label_id"] for row in test_rows})
    print(f"Loaded {len(points)} ModelNet40 test samples.")
    print(f"Sampled {len(train_rows)} train/adaptation samples covering {train_covered} of 40 classes.")
    print(f"Prepared {len(test_rows)} validation samples covering {test_covered} of 40 classes.")
    print(f"Saved point cloud .npy files to {pc_dir}")
    print(f"Saved {train_path} and {test_path}")


if __name__ == "__main__":
    main()
