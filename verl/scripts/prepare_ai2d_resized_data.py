#!/usr/bin/env python
import argparse
import copy
import json
from pathlib import Path
from typing import Any

import datasets
from PIL import Image, ImageOps


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _resize_rgb_image(src: Path, dst: Path, max_side: int) -> dict[str, Any]:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        old_size = im.size
        scale = min(1.0, float(max_side) / float(max(old_size)))
        if scale < 1.0:
            new_size = (max(1, round(old_size[0] * scale)), max(1, round(old_size[1] * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        else:
            new_size = old_size
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Use baseline RGB JPEGs to avoid EXIF/progressive/CMYK surprises in the processor.
        im.save(dst, format='JPEG', quality=95, optimize=False, progressive=False)
    return {'old_size': old_size, 'new_size': new_size, 'scale': scale}


def _rewrite_rows(rows: list[dict[str, Any]], source_image_root: Path, resized_image_root: Path, max_side: int, image_stats: dict[str, Any]) -> list[dict[str, Any]]:
    rewritten = []
    for row in rows:
        item = copy.deepcopy(row)
        src = Path(item['images'][0]['image'])
        try:
            rel = src.relative_to(source_image_root)
        except ValueError:
            rel = Path(src.name)
        dst = (resized_image_root / rel).with_suffix('.jpg')
        key = str(src)
        if key not in image_stats:
            image_stats[key] = _resize_rgb_image(src, dst, max_side)
            image_stats[key]['resized_path'] = str(dst)
        new_path = image_stats[key]['resized_path']
        item['images'][0]['image'] = new_path
        extra = item.setdefault('extra_info', {})
        extra['original_image_path_before_resize'] = extra.get('image_path', str(src))
        extra['image_path'] = new_path
        extra['ai2d_resize_max_side'] = max_side
        extra['ai2d_resize_policy'] = 'exif_transpose_rgb_jpeg_lanczos_max_side'
        rewritten.append(item)
    return rewritten


def _write_split(rows: list[dict[str, Any]], output_task_dir: Path, split: str) -> None:
    output_task_dir.mkdir(parents=True, exist_ok=True)
    with (output_task_dir / f'{split}.json').open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    datasets.Dataset.from_list(rows).to_parquet(str(output_task_dir / f'{split}.parquet'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Create an AI2D data copy with normalized/resized RGB images for InternVL/vLLM stability.')
    parser.add_argument('--source-dir', default='/jiigan-hp/ttrv-datasets/verl_data/ai2d_20')
    parser.add_argument('--source-image-root', default='/jiigan-hp/ttrv-datasets/ai2d')
    parser.add_argument('--output-root', default='/jiigan-hp/ttrv-datasets/verl_data/ai2d_20_rgb_max896')
    parser.add_argument('--resized-image-root', default='/jiigan-hp/ttrv-datasets/ai2d_rgb_max896')
    parser.add_argument('--task-name', default='ai2d_20')
    parser.add_argument('--max-side', type=int, default=896)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_task_dir = Path(args.output_root) / args.task_name
    source_image_root = Path(args.source_image_root)
    resized_image_root = Path(args.resized_image_root)
    image_stats: dict[str, Any] = {}

    train_rows = _rewrite_rows(_load_rows(source_dir / 'train.json'), source_image_root, resized_image_root, args.max_side, image_stats)
    test_rows = _rewrite_rows(_load_rows(source_dir / 'test.json'), source_image_root, resized_image_root, args.max_side, image_stats)
    _write_split(train_rows, output_task_dir, 'train')
    _write_split(test_rows, output_task_dir, 'test')

    scales = [v['scale'] for v in image_stats.values()]
    old_max = max(max(v['old_size']) for v in image_stats.values()) if image_stats else 0
    new_max = max(max(v['new_size']) for v in image_stats.values()) if image_stats else 0
    resized_count = sum(1 for v in image_stats.values() if v['scale'] < 1.0)
    summary = {
        'source_dir': str(source_dir),
        'output_task_dir': str(output_task_dir),
        'source_image_root': str(source_image_root),
        'resized_image_root': str(resized_image_root),
        'task_name': args.task_name,
        'max_side': args.max_side,
        'train_examples': len(train_rows),
        'test_examples': len(test_rows),
        'unique_images': len(image_stats),
        'resized_images': resized_count,
        'original_max_side': old_max,
        'new_max_side': new_max,
        'min_scale': min(scales) if scales else None,
        'max_scale': max(scales) if scales else None,
        'train_parquet': str(output_task_dir / 'train.parquet'),
        'test_parquet': str(output_task_dir / 'test.parquet'),
    }
    with (output_task_dir / 'resize_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
