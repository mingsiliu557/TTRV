#!/usr/bin/env python
"""Audit PAPO grounding JSONL with fallback stats and box contact sheets."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from ._common import grounding_key, image_specs, load_image_from_spec, read_jsonl, read_rows, row_data_source
except ImportError:
    from _common import grounding_key, image_specs, load_image_from_spec, read_jsonl, read_rows, row_data_source  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-jsonl", required=True)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--thumb-size", type=int, default=320)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def build_image_map(input_parquet: str, image_key: str):
    rows = read_rows(input_parquet)
    image_map = {}
    source_map = {}
    for ordinal, row in enumerate(rows):
        key = grounding_key(row, ordinal)
        specs = image_specs(row, image_key=image_key)
        image_map[key] = specs[0] if specs else None
        source_map[key] = row_data_source(row)
    return image_map, source_map


def box_list(value):
    if isinstance(value, str) and value.upper() == "WHOLE_IMAGE":
        return "WHOLE_IMAGE"
    if not isinstance(value, list):
        return []
    boxes = []
    for item in value:
        if isinstance(item, list) and len(item) == 4:
            try:
                boxes.append([float(x) for x in item])
            except Exception:
                continue
    return boxes


def collect_stats(records, source_map):
    stats = defaultdict(lambda: defaultdict(int))
    reasons = defaultdict(lambda: defaultdict(int))
    for record in records:
        key = str(record.get("grounding_key") or "")
        source = str(record.get("data_source") or source_map.get(key, "unknown"))
        stats[source]["total"] += 1
        boxes = box_list(record.get("boxes_norm"))
        fallback = bool(record.get("fallback"))
        if fallback:
            stats[source]["fallback"] += 1
            reason = str(record.get("fallback_reason") or "unknown")
            reasons[source][reason] += 1
        elif boxes:
            stats[source]["grounded"] += 1
        else:
            stats[source]["empty_boxes"] += 1
    summary = {}
    for source, values in stats.items():
        total = max(1, values["total"])
        summary[source] = {
            "total": values["total"],
            "grounded": values["grounded"],
            "fallback": values["fallback"],
            "empty_boxes": values["empty_boxes"],
            "fallback_ratio": values["fallback"] / total,
            "fallback_reasons": dict(reasons[source]),
        }
    return summary


def draw_record(record, image_spec, thumb_size: int):
    from PIL import Image, ImageDraw, ImageFont

    label_h = 56
    try:
        image = load_image_from_spec(image_spec) if image_spec is not None else Image.new("RGB", (thumb_size, thumb_size), "white")
    except Exception:
        image = Image.new("RGB", (thumb_size, thumb_size), "white")
    width, height = image.size
    scale = min(thumb_size / width, thumb_size / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(new_size)
    canvas = Image.new("RGB", (thumb_size, thumb_size + label_h), "white")
    offset = ((thumb_size - new_size[0]) // 2, (thumb_size - new_size[1]) // 2)
    canvas.paste(image, offset)
    draw = ImageDraw.Draw(canvas)
    boxes = box_list(record.get("boxes_norm"))
    fallback = bool(record.get("fallback"))
    color = "red" if fallback else "lime"
    if boxes == "WHOLE_IMAGE":
        draw.rectangle((offset[0], offset[1], offset[0] + new_size[0] - 1, offset[1] + new_size[1] - 1), outline="orange", width=4)
    elif isinstance(boxes, list):
        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle(
                (
                    offset[0] + x0 * new_size[0],
                    offset[1] + y0 * new_size[1],
                    offset[0] + x1 * new_size[0],
                    offset[1] + y1 * new_size[1],
                ),
                outline=color,
                width=3,
            )
    key = str(record.get("grounding_key") or "")
    reason = str(record.get("fallback_reason") or "grounded")
    descriptor = str(record.get("descriptor") or "")[:80]
    text = f"{key[-32:]}\n{reason}: {descriptor}"
    draw.rectangle((0, thumb_size, thumb_size, thumb_size + label_h), fill="white")
    draw.text((6, thumb_size + 4), text, fill="black")
    return canvas


def write_contact_sheet(records, image_map, out_path: Path, limit: int, thumb_size: int, columns: int):
    from PIL import Image

    selected = records[:limit]
    if not selected:
        return
    tiles = [draw_record(record, image_map.get(str(record.get("grounding_key") or "")), thumb_size) for record in selected]
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 56)), "white")
    for i, tile in enumerate(tiles):
        x = (i % columns) * thumb_size
        y = (i // columns) * (thumb_size + 56)
        sheet.paste(tile, (x, y))
    sheet.save(out_path)


def write_markdown(summary, out_path: Path):
    lines = ["# PAPO Grounding Audit", "", "| data_source | total | grounded | fallback | fallback_ratio | reasons |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for source, values in sorted(summary.items()):
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(values["fallback_reasons"].items())) or "n/a"
        lines.append(
            f"| {source} | {values['total']} | {values['grounded']} | {values['fallback']} | {values['fallback_ratio']:.4f} | {reasons} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.grounding_jsonl)
    image_map, source_map = build_image_map(args.input_parquet, args.image_key)
    summary = collect_stats(records, source_map)
    (out_dir / "fallback_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, out_dir / "fallback_stats.md")
    write_contact_sheet(records, image_map, out_dir / "contact_sheet.jpg", args.limit, args.thumb_size, args.columns)
    print(f"wrote audit artifacts to {out_dir}")


if __name__ == "__main__":
    main()
