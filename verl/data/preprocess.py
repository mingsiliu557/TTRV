import argparse
import os
from pathlib import Path

import datasets


def resolve_image_path(image_path, image_root=None):
    image_path = Path(image_path).expanduser()
    if image_path.exists() or image_root is None:
        return str(image_path)

    image_root = Path(image_root).expanduser().resolve()
    parts = image_path.parts
    if image_root.name in parts:
        root_index = parts.index(image_root.name)
        return str(image_root.joinpath(*parts[root_index + 1:]))
    return str(image_root / image_path.name)


def make_map_fn(split, source=None, image_root=None):
    def process_fn(example, idx):
        if source is None:
            data_source = example.pop("source")
        else:
            data_source = source

        # Construct question
        question = example.pop("prompt")
        image_path = example.pop("image_path", None)  # Allow missing image_path
        solution = example.pop("answer")

        # Only include images if image_path is present and file exists
        images = None
        if image_path:
            image_path = resolve_image_path(image_path, image_root)
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image not found for {data_source}/{split} row {idx}: {image_path}. "
                    "Download the dataset images and pass --image-root or set IMAGE_ROOT."
                )
            images = [{"image": image_path}]

        data = {
            "data_source": "GPQA-TTT",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}",
            },
        }
        if images is not None:
            data["images"] = images

        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_source", nargs="?", default=os.environ.get("TASK", "dtd_20"))
    parser.add_argument("--data-dir", default=Path(__file__).resolve().parent)
    parser.add_argument("--image-root", default=os.environ.get("IMAGE_ROOT"))
    args = parser.parse_args()

    data_source = args.data_source
    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_dir = data_dir / data_source

    train_dataset = datasets.load_dataset("json", data_files=str(dataset_dir / "train.json"), split="train")
    test_dataset = datasets.load_dataset("json", data_files=str(dataset_dir / "test.json"), split="train")

    train_dataset = train_dataset.map(function=make_map_fn("train", data_source, args.image_root), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source, args.image_root), with_indices=True)

    train_dataset.to_parquet(str(dataset_dir / "train.parquet"))
    test_dataset.to_parquet(str(dataset_dir / "test.parquet"))
    print(f"Wrote parquet files to {dataset_dir}")
